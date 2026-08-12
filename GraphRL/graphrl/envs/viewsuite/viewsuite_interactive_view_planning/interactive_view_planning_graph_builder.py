"""
Graph builder for ViewSuite Interactive View Planning.

Extends VagenGraphBuilder — implements traj_to_transitions() for extracting
camera-pose transitions from VAGEN active exploration rollouts.

Overrides convert_files() to add image quality filtering after graph building.

Node = camera pose (6-DoF) + observation image.
Edge = action sequence (turn_left, move_forward, etc.).

Registered as "viewsuite_interactive_view_planning" in graph_builder_registry.
"""

import hashlib
import logging
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from graphrl.traj_to_sft.utils.base_graph import BaseGraph, NodeData, EdgeData
from graphrl.traj_to_sft.utils.graph_builder import VagenGraphBuilder, VagenNodeData, VagenEdgeData

logger = logging.getLogger(__name__)

# ── regex patterns ────────────────────────────────────────────────────────────

_SCENE_ID_RE = re.compile(r"scene(\d+_\d+)")
_POSE_RE = re.compile(
    r"\[tx=([\d.e+-]+),\s*ty=([\d.e+-]+),\s*tz=([\d.e+-]+),\s*"
    r"rx=([\d.e+-]+)°?,\s*ry=([\d.e+-]+)°?,\s*rz=([\d.e+-]+)°?\]"
)
_ACTION_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL)
_GROUND_DESC_RE = re.compile(r"<description>(.*?)</description>", re.DOTALL | re.IGNORECASE)
_GROUND_PRED_RE = re.compile(r"<prediction>(.*?)</prediction>", re.DOTALL | re.IGNORECASE)
_ANSWER_RE = re.compile(r"^answer\s*\(", re.IGNORECASE)
_IMAGE_PLACEHOLDER = "<image>"

_VALID_ACTIONS = frozenset({
    "move_forward", "move_backward", "move_left", "move_right",
    "move_up", "move_down", "turn_left", "turn_right",
    "look_up", "look_down", "rotate_cw", "rotate_ccw",
})


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_pose(text: str) -> Optional[Dict[str, float]]:
    """Extract 6-DoF camera pose from text like [tx=1.23, ty=4.56, ...]."""
    m = _POSE_RE.search(text)
    if not m:
        return None
    return {
        "tx": float(m.group(1)),
        "ty": float(m.group(2)),
        "tz": float(m.group(3)),
        "rx": float(m.group(4)),
        "ry": float(m.group(5)),
        "rz": float(m.group(6)),
    }


def _parse_action(text: str) -> Optional[str]:
    """Extract action from <action>...</action> tags. Returns None for answer(...)."""
    m = _ACTION_RE.search(text)
    if not m:
        return None
    action_text = m.group(1).strip()
    if _ANSWER_RE.match(action_text):
        return None  # final answer, not a movement action
    return action_text


def _clean_action(action: str) -> Optional[str]:
    """Filter out noisy actions, keeping only valid ones.

    Splits by ``|``, normalises each sub-action to lowercase and strips
    whitespace, then keeps only those in ``_VALID_ACTIONS``.
    Returns the cleaned ``|``-joined string, or ``None`` if nothing valid.
    """
    parts = [a.strip().lower() for a in action.split("|")]
    valid = [a for a in parts if a in _VALID_ACTIONS]
    return " | ".join(valid) if valid else None


def _pose_to_text(pose: Dict[str, float]) -> str:
    """Format pose as human-readable string."""
    return (
        f"[tx={pose['tx']:.4f}, ty={pose['ty']:.4f}, tz={pose['tz']:.4f}, "
        f"rx={pose['rx']:.2f}°, ry={pose['ry']:.2f}°, rz={pose['rz']:.2f}°]"
    )


def _count_images(content: str) -> int:
    """Count <image> placeholders in a message."""
    return content.count(_IMAGE_PLACEHOLDER)


def _action_count(obs_str: str) -> int:
    """Count individual actions in a pipe-separated action string."""
    return len([a for a in obs_str.split("|") if a.strip()])


def _image_passes_filter(
    image_path: str,
    void_threshold: float = 0.7,
    std_threshold: float = 10.0,
    void_color: int = 255,
    void_tolerance: int = 5,
) -> bool:
    """
    Check image quality. Returns False if:
    - More than void_threshold of pixels are void (near void_color)
    - Standard deviation < std_threshold (nearly uniform)
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if np.std(gray) < std_threshold:
            return False
        void_mask = np.abs(gray.astype(np.float32) - void_color) <= void_tolerance
        if np.sum(void_mask) / gray.size > void_threshold:
            return False
        return True
    except Exception:
        return False


# ── Union-Find for edge refinement ───────────────────────────────────────────

class _UnionFind:
    """Union-Find with union-by-rank, path compression, and real-node preference."""

    __slots__ = ("parent", "rank", "is_real")

    def __init__(self) -> None:
        self.parent: Dict[str, str] = {}
        self.rank: Dict[str, int] = {}
        self.is_real: Dict[str, bool] = {}

    def add(self, x: str, real: bool = False) -> None:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            self.is_real[x] = real

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: str, y: str) -> bool:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        # Prefer real node as representative
        if self.is_real[ry] and not self.is_real[rx]:
            rx, ry = ry, rx
        elif self.rank[rx] < self.rank[ry] and not (self.is_real[rx] and not self.is_real[ry]):
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        self.is_real[rx] = self.is_real[rx] or self.is_real[ry]
        return True


# ── ViewSuite node data (Layer 2) ────────────────────────────────────────────

class ViewSuiteNodeData(VagenNodeData):
    """
    Node data for ViewSuite active exploration.

    state = {"scene_id": str, "pose": {"tx", "ty", "tz", "rx", "ry", "rz"}}

    Dedup:
        unique_key  — scene_id + "_" + md5(scene_id + pose at 4dp)
        bucket_key  — scene_id (all nodes in a scene are compared)
        is_similar_to — each position axis < 1e-3 m and each angle < 1e-3°
    """

    POSITION_TOL = 1e-3   # metres, per axis
    ANGLE_TOL = 1e-3      # degrees, per axis

    def unique_key(self) -> str:
        p = self.state["pose"]
        pose_str = (
            f"{p['tx']:.4f}_{p['ty']:.4f}_{p['tz']:.4f}_"
            f"{p['rx']:.4f}_{p['ry']:.4f}_{p['rz']:.4f}"
        )
        raw = f"{self.state['scene_id']}|{pose_str}"
        pose_hash = hashlib.md5(raw.encode()).hexdigest()[:12]
        return f"{self.state['scene_id']}_{pose_hash}"

    def bucket_key(self) -> str:
        return self.state["scene_id"]

    def is_similar_to(self, other: "NodeData") -> bool:
        if not isinstance(other, ViewSuiteNodeData):
            return False
        pa, pb = self.state["pose"], other.state["pose"]
        for k in ("tx", "ty", "tz"):
            if abs(pa[k] - pb[k]) > self.POSITION_TOL:
                return False
        for k in ("rx", "ry", "rz"):
            if abs(pa[k] - pb[k]) > self.ANGLE_TOL:
                return False
        return True


# ── graph builder ─────────────────────────────────────────────────────────────

class InteractiveViewPlanningGraphBuilder(VagenGraphBuilder):
    """
    Converts VAGEN active exploration rollouts into a BaseGraph.

    Parses multi-turn conversations where:
    - User messages contain camera poses and <image> placeholders
    - Assistant messages contain <action>...</action> movement commands

    Config keys:
        num_workers (int, default 1): parallel worker processes.
        filter:
            void_threshold (float, default 0.7): max void pixel ratio.
            std_threshold (float, default 10.0): min image std deviation.
    """

    def __init__(self, config):
        super().__init__(config)
        # Idea-5: coarser node-merge tolerance unlocks cross-trajectory merge.
        # The agent moves on a 0.5m/30deg grid, but the default 1e-3 tolerance is
        # ~exact-match, so float drift keeps grid-identical poses from different
        # trajectories from ever merging -> the graph barely beats no-graph.
        # Set the tolerance BELOW half a step (< 0.25m / < 15deg) so distinct
        # ADJACENT grid cells never merge. Configurable via graph_builder.merge_tol.
        mt = (config.get("merge_tol") or {})
        if mt.get("position") is not None:
            ViewSuiteNodeData.POSITION_TOL = float(mt["position"])
        if mt.get("angle") is not None:
            ViewSuiteNodeData.ANGLE_TOL = float(mt["angle"])

    # ── factory override ──────────────────────────────────────────────────────

    def _make_node_data(self, ndata: Dict[str, Any]) -> NodeData:
        return ViewSuiteNodeData(
            state=ndata["state"],
            obs_str=ndata.get("obs_str"),
            image_paths=ndata.get("image_paths", []),
            extra=ndata.get("extra", {}),
        )

    # ── edge: shorter wins ───────────────────────────────────────────────────

    def _apply_transitions(
        self, graph: BaseGraph,
        transitions: List[Tuple[NodeData, EdgeData, NodeData]],
        images_dir: Path,
    ) -> None:
        """Apply transitions with shorter-wins edge policy.

        For a given (src, dst) node pair, only the edge with the fewest
        actions is kept (matching ViewSuite DiGraph behaviour).
        """
        for src, edge, dst in transitions:
            # Phase 1: resolve nodes (same as base)
            src_uid = self._resolve_node(src, graph)
            src_imgs = self._copy_images(src, src_uid, images_dir)
            graph._upsert_node(
                src_uid, src.state,
                obs_str=src.obs_str, extra=src.extra, image_paths=src_imgs,
            )

            dst_uid = self._resolve_node(dst, graph)
            dst_imgs = self._copy_images(dst, dst_uid, images_dir)
            graph._upsert_node(
                dst_uid, dst.state,
                obs_str=dst.obs_str, extra=dst.extra, image_paths=dst_imgs,
            )

            if src_uid == dst_uid:
                continue

            # Phase 2: shorter-wins edge logic
            new_count = _action_count(edge.obs_str)
            g = graph._g

            existing = g.get_edge_data(src_uid, dst_uid)
            if existing:
                # Find the shortest existing edge
                best_eid, best_count = None, float("inf")
                for eid, edata in existing.items():
                    c = _action_count(edata["obs_str"])
                    if c < best_count:
                        best_eid, best_count = eid, c
                if new_count < best_count:
                    # New edge is shorter — remove all existing, add new
                    for eid in list(existing.keys()):
                        g.remove_edge(src_uid, dst_uid, key=eid)
                    g.add_edge(
                        src_uid, dst_uid, key=repr(edge.obs_str),
                        obs_str=edge.obs_str,
                        image_paths=edge.image_paths,
                        extra=edge.extra,
                    )
                # else: existing is shorter or equal — keep it
            else:
                # No existing edge — add directly
                g.add_edge(
                    src_uid, dst_uid, key=repr(edge.obs_str),
                    obs_str=edge.obs_str,
                    image_paths=edge.image_paths,
                    extra=edge.extra,
                )

    def _merge_graph(self, target: BaseGraph, source: BaseGraph) -> None:
        """Merge source into target with shorter-wins edge policy."""
        key_map: Dict[str, str] = {}
        for old_key, ndata in source._g.nodes(data=True):
            node = self._make_node_data(ndata)
            resolved = self._resolve_node(node, target)
            key_map[old_key] = resolved
            target._upsert_node(
                resolved, ndata["state"],
                obs_str=ndata.get("obs_str"),
                extra=ndata.get("extra", {}),
                image_paths=ndata.get("image_paths", []),
            )
        for u, v, _eid, data in source._g.edges(data=True, keys=True):
            new_u, new_v = key_map.get(u, u), key_map.get(v, v)
            if new_u == new_v:
                continue
            new_count = _action_count(data["obs_str"])
            existing = target._g.get_edge_data(new_u, new_v)
            if existing:
                best_count = min(_action_count(ed["obs_str"]) for ed in existing.values())
                if new_count < best_count:
                    for eid in list(existing.keys()):
                        target._g.remove_edge(new_u, new_v, key=eid)
                    target._g.add_edge(
                        new_u, new_v, key=repr(data["obs_str"]),
                        obs_str=data["obs_str"],
                        image_paths=data.get("image_paths", []),
                        extra=data.get("extra", {}),
                    )
            else:
                target._g.add_edge(
                    new_u, new_v, key=repr(data["obs_str"]),
                    obs_str=data["obs_str"],
                    image_paths=data.get("image_paths", []),
                    extra=data.get("extra", {}),
                )

    # ── transition extraction ────────────────────────────────────────────────

    def traj_to_transitions(
        self,
        messages: List[Dict[str, str]],
        rollout_dir: Path,
        step_idx: int,
        line_idx: int,
    ) -> List[Tuple[NodeData, EdgeData, NodeData]]:
        """
        Extract (src_pose, action, dst_pose) transitions from one episode.

        Conversation structure (multi-turn active exploration):
          user[0]:  initial view <image>, top-down <image>, target <image>, pose
          asst[0]:  <action>turn_left | move_forward</action>
          user[1]:  new pose, <image>
          asst[1]:  <action>...</action>
          ...
          asst[N]:  <action>answer(tx, ty, tz, rx, ry, rz)</action>  (skipped)
        """
        image_base = rollout_dir / f"image_{step_idx}" / f"images_{line_idx}"

        # Extract scene_id from the first user message (e.g. "You're in the scene scene0353_02.")
        scene_id = "unknown"
        for msg in messages:
            if msg["role"] == "user":
                m = _SCENE_ID_RE.search(msg["content"])
                if m:
                    scene_id = f"scene{m.group(1)}"
                break

        # First pass: collect all states (pose + image) and actions
        states: List[Tuple[Dict[str, float], Optional[str]]] = []  # (pose, image_path)
        actions: List[Optional[str]] = []  # action text after each state
        grounding_txt: List[Dict[str, str]] = []  # per-state {"desc","pred"} (grounding fmt)
        global_img_idx = 0

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "user":
                pose = _parse_pose(content)
                if pose is None:
                    global_img_idx += _count_images(content)
                    continue

                num_images = _count_images(content)
                obs_img_path = None
                if num_images > 0:
                    obs_img_idx = global_img_idx
                    for suffix in (".png", ".jpg"):
                        candidate = image_base / f"{obs_img_idx}{suffix}"
                        if candidate.exists():
                            obs_img_path = str(candidate)
                            break
                    global_img_idx += num_images
                else:
                    global_img_idx += num_images

                states.append((pose, obs_img_path))
                actions.append(None)
                grounding_txt.append({})

            elif role == "assistant":
                action = _parse_action(content)
                if action is not None:
                    action = _clean_action(action)
                if action and actions:
                    actions[-1] = action
                # Grounding format: cache the goal-AGNOSTIC visual text so it can be
                # reused as SFT supervision after a hindsight goal relabel.
                #   <description> describes the CURRENT view  -> belongs to state i
                #   <prediction>  describes the NEXT view     -> belongs to state i+1
                # (<think> is goal-dependent and deliberately NOT cached here.)
                if grounding_txt:
                    d = _GROUND_DESC_RE.search(content)
                    p = _GROUND_PRED_RE.search(content)
                    if d and grounding_txt:
                        grounding_txt[-1]["desc"] = d.group(1).strip()
                    if p and grounding_txt:
                        grounding_txt[-1]["pred"] = p.group(1).strip()

        # Build transitions: (state_i, action_i, state_{i+1})
        transitions: List[Tuple[NodeData, EdgeData, NodeData]] = []
        for i in range(len(states) - 1):
            pose_src, img_src = states[i]
            pose_dst, img_dst = states[i + 1]
            action = actions[i]

            if action is None:
                continue  # no valid action between these states

            g_src = grounding_txt[i] if i < len(grounding_txt) else {}
            g_dst = grounding_txt[i + 1] if i + 1 < len(grounding_txt) else {}
            src_extra = {"scene_id": scene_id}
            if g_src.get("desc"):
                src_extra["view_desc"] = g_src["desc"]
            dst_extra = {"scene_id": scene_id}
            # prefer the NEXT state's own description; fall back to this step's prediction
            if g_dst.get("desc"):
                dst_extra["view_desc"] = g_dst["desc"]
            elif g_src.get("pred"):
                dst_extra["view_desc"] = g_src["pred"]

            src = ViewSuiteNodeData(
                state={"scene_id": scene_id, "pose": pose_src},
                obs_str=_pose_to_text(pose_src),
                source_images=[img_src] if img_src else [],
                extra=src_extra,
            )
            dst = ViewSuiteNodeData(
                state={"scene_id": scene_id, "pose": pose_dst},
                obs_str=_pose_to_text(pose_dst),
                source_images=[img_dst] if img_dst else [],
                extra=dst_extra,
            )
            edge_extra = {}
            if g_src.get("pred"):
                edge_extra["pred_desc"] = g_src["pred"]
            transitions.append((src, VagenEdgeData(obs_str=action, extra=edge_extra), dst))

        return transitions

    # ── graph building with filter ───────────────────────────────────────────

    def convert_files(
        self,
        files: List[Path],
        rollout_dir: Path,
        graph_dir: Path,
    ) -> None:
        if not files:
            return

        graph_dir = Path(graph_dir)
        graph_dir.mkdir(parents=True, exist_ok=True)
        images_dir = graph_dir / "images"
        images_dir.mkdir(exist_ok=True)

        num_workers = int(self.config.get("num_workers", 1))
        if num_workers > 1:
            graph = self._build_parallel(files, rollout_dir, images_dir, num_workers)
        else:
            graph = self._build_sequential(files, rollout_dir, images_dir)

        # Apply image quality filter
        filter_cfg = self.config.get("filter", {})
        removed = self._filter_graph(
            graph,
            images_dir,
            void_threshold=filter_cfg.get("void_threshold", 0.7),
            std_threshold=filter_cfg.get("std_threshold", 10.0),
        )
        if removed:
            logger.info(
                "[InteractiveViewPlanningGraphBuilder] Filtered %d low-quality nodes", removed,
            )

        # Split multi-action edges and merge equivalent nodes.
        # Idea-1: if atomize is enabled, render intermediate views so the graph
        # becomes 100% single-action (multi-action edges whose intermediates fail
        # to render are dropped), instead of collapsing back to "a | b | c" edges.
        atom_cfg = self.config.get("atomize", {}) or {}
        if atom_cfg.get("enabled"):
            from .utils.graph_atomize import atomize_graph
            try:
                stats = atomize_graph(self, graph, images_dir, atom_cfg)
                logger.info(
                    "[InteractiveViewPlanningGraphBuilder] Atomize: %s", stats,
                )
            except Exception as e:
                # Fail-safe: never crash the pipeline. Still guarantee a
                # single-action graph by pruning any multi-action edges.
                logger.error(
                    "[atomize] FAILED (%s) -> pruning multi-action edges as fallback",
                    e, exc_info=True,
                )
                pruned = 0
                for u, v, eid, data in list(graph._g.edges(data=True, keys=True)):
                    if len([a for a in data["obs_str"].split("|") if a.strip()]) > 1:
                        graph._g.remove_edge(u, v, key=eid)
                        pruned += 1
                logger.info("[atomize] fallback pruned %d multi-action edges", pruned)
        else:
            n_virt, n_merged = self._refine_graph(graph)
            if n_virt:
                logger.info(
                    "[InteractiveViewPlanningGraphBuilder] Refine: %d virtual nodes created, "
                    "%d nodes merged",
                    n_virt, n_merged,
                )

        # Remove redundant edges (Dijkstra-based)
        n_redundant = self._remove_redundant_edges(graph)
        if n_redundant:
            logger.info(
                "[InteractiveViewPlanningGraphBuilder] Removed %d redundant edges", n_redundant,
            )

        graph.save(graph_dir)
        logger.info(
            "[%s] %d file(s) → graph (%s): %d nodes, %d edges",
            self.__class__.__name__, len(files), graph_dir,
            graph.num_nodes, graph.num_edges,
        )

    def _refine_graph(self, graph: BaseGraph) -> Tuple[int, int]:
        """Split multi-action edges into single-action chains, then merge
        equivalent nodes using Union-Find.

        Algorithm:
          A. For each edge with k actions (k>1), create k-1 virtual nodes
             and replace the edge with a chain of single-action edges.
          B. Iteratively apply two merge rules until convergence:
             Rule 1: p --[a]--> x AND p --[a]--> y  ⇒  union(x, y)
             Rule 2: x --[a]--> s AND y --[a]--> s  ⇒  union(x, y)
          C. Rebuild graph using Union-Find representatives.

        Returns (num_virtual_created, num_nodes_merged).
        """
        g = graph._g

        # ── Step A: Split multi-action edges ──
        virt_counter = 0
        edges_to_remove: List[Tuple[str, str, str]] = []
        new_nodes: List[Tuple[str, Dict]] = []
        new_edges: List[Tuple[str, str, str]] = []  # (from, to, action)

        for u, v, eid, data in list(g.edges(data=True, keys=True)):
            actions = [a.strip() for a in data["obs_str"].split("|") if a.strip()]
            if len(actions) <= 1:
                continue
            edges_to_remove.append((u, v, eid))
            scene_id = g.nodes[u].get("extra", {}).get("scene_id", "unknown")
            prev = u
            for i, action in enumerate(actions):
                if i < len(actions) - 1:
                    vid = f"__virt_{virt_counter}"
                    virt_counter += 1
                    new_nodes.append((vid, {
                        "state": None, "obs_str": "",
                        "image_paths": [],
                        "extra": {"virtual": True, "scene_id": scene_id},
                    }))
                    nxt = vid
                else:
                    nxt = v
                new_edges.append((prev, nxt, action))
                prev = nxt

        if virt_counter == 0:
            return 0, 0

        for u, v, eid in edges_to_remove:
            g.remove_edge(u, v, key=eid)
        for vid, attrs in new_nodes:
            g.add_node(vid, **attrs)
        for u, v, action in new_edges:
            key = repr(action)
            if not g.has_edge(u, v, key=key):
                g.add_edge(u, v, key=key, obs_str=action, image_paths=[], extra={})

        # ── Step B: Union-Find merge ──
        uf = _UnionFind()
        for nid in g.nodes():
            uf.add(nid, real=bool(g.nodes[nid].get("image_paths")))

        iteration = 0
        while True:
            iteration += 1
            changed = False

            # Collect edges with canonical endpoints
            canon_edges = [
                (uf.find(u), uf.find(v), data["obs_str"])
                for u, v, _eid, data in g.edges(data=True, keys=True)
            ]

            # Rule 1: same source + same action → union targets
            out_groups: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
            for fu, fv, action in canon_edges:
                out_groups[(fu, action)].add(fv)
            for targets in out_groups.values():
                reps = list({uf.find(t) for t in targets})
                for i in range(1, len(reps)):
                    if uf.union(reps[0], reps[i]):
                        changed = True

            # Rule 2: same target + same action → union sources
            in_groups: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
            for fu, fv, action in canon_edges:
                in_groups[(action, uf.find(fv))].add(uf.find(fu))
            for sources in in_groups.values():
                reps = list({uf.find(s) for s in sources})
                for i in range(1, len(reps)):
                    if uf.union(reps[0], reps[i]):
                        changed = True

            if not changed:
                break

        # ── Step C: Rebuild graph ──
        # Pick best node attributes per representative (prefer real nodes)
        rep_attrs: Dict[str, Dict] = {}
        for nid in list(g.nodes()):
            rep = uf.find(nid)
            attrs = dict(g.nodes[nid])
            if rep not in rep_attrs:
                rep_attrs[rep] = attrs
            elif attrs.get("image_paths") and not rep_attrs[rep].get("image_paths"):
                rep_attrs[rep] = attrs

        # Collect canonical edges (dedup, skip self-loops)
        seen_edges: Set[Tuple[str, str, str]] = set()
        edge_data_map: Dict[Tuple[str, str, str], Dict] = {}
        for u, v, _eid, data in g.edges(data=True, keys=True):
            ru, rv = uf.find(u), uf.find(v)
            if ru == rv:
                continue
            key = (ru, rv, data["obs_str"])
            if key not in seen_edges:
                seen_edges.add(key)
                edge_data_map[key] = data

        # Build new graph
        new_g = g.__class__()
        for rep, attrs in rep_attrs.items():
            new_g.add_node(rep, **attrs)
        for (ru, rv, action), data in edge_data_map.items():
            new_g.add_edge(
                ru, rv, key=repr(action),
                obs_str=action,
                image_paths=data.get("image_paths", []),
                extra=data.get("extra", {}),
            )

        num_before = g.number_of_nodes()
        graph._g = new_g
        num_merged = num_before - new_g.number_of_nodes()

        # ── Step D: Collapse remaining virtual nodes into multi-action edges ──
        n_collapsed = self._collapse_virtual_nodes(graph)

        logger.debug(
            "[refine] %d iterations, %d virtual created, %d merged, "
            "%d collapsed, %d nodes / %d edges remaining",
            iteration, virt_counter, num_merged, n_collapsed,
            graph.num_nodes, graph.num_edges,
        )
        return virt_counter, num_merged

    def _collapse_virtual_nodes(self, graph: BaseGraph) -> int:
        """Remove remaining virtual nodes by collapsing chains into multi-action edges.

        For each chain  real_A --[a1]--> virt_0 --[a2]--> ... --[aN]--> real_B,
        creates a single edge  real_A --[a1 | a2 | ... | aN]--> real_B
        and removes all virtual nodes in the chain.

        Handles branching virtual nodes by tracing each outgoing branch
        as a separate DFS path from the original real start node.

        Returns count of collapsed virtual nodes.
        """
        g = graph._g
        virtual_ids = {
            nid for nid, attrs in g.nodes(data=True)
            if attrs.get("extra", {}).get("virtual", False)
        }
        if not virtual_ids:
            return 0

        edges_to_add: List[Tuple[str, str, str]] = []  # (from, to, combined_action)

        def _trace_to_real(start: str, actions: List[str], cur: str, visited: Set[str]) -> None:
            """DFS: follow virtual nodes, branching at forks, until real nodes."""
            if cur not in virtual_ids:
                # Reached a real node — record the combined edge
                edges_to_add.append((start, cur, " | ".join(actions)))
                return
            if cur in visited:
                return  # cycle guard
            visited.add(cur)
            out = list(g.out_edges(cur, data=True, keys=True))
            if not out:
                return  # dead-end virtual
            for _, nxt, _eid, edata in out:
                _trace_to_real(start, actions + [edata["obs_str"]], nxt, visited)

        # Find all real→virtual edges and trace each branch
        for u, v, _eid, data in list(g.edges(data=True, keys=True)):
            if u not in virtual_ids and v in virtual_ids:
                _trace_to_real(u, [data["obs_str"]], v, set())

        # Remove all virtual nodes (and their incident edges)
        for vid in list(virtual_ids):
            if vid in g:
                g.remove_node(vid)

        # Add combined edges with shorter-wins policy
        for u, v, combined_action in edges_to_add:
            if u in g and v in g:
                existing = g.get_edge_data(u, v)
                new_count = _action_count(combined_action)
                if existing:
                    best_count = min(_action_count(ed["obs_str"]) for ed in existing.values())
                    if new_count < best_count:
                        for eid in list(existing.keys()):
                            g.remove_edge(u, v, key=eid)
                        g.add_edge(
                            u, v, key=repr(combined_action),
                            obs_str=combined_action, image_paths=[], extra={},
                        )
                else:
                    g.add_edge(
                        u, v, key=repr(combined_action),
                        obs_str=combined_action, image_paths=[], extra={},
                    )

        return len(virtual_ids)

    def _remove_redundant_edges(self, graph: BaseGraph) -> int:
        """Remove edges that can be replaced by an alternative path of equal or shorter cost.

        For each edge u→v with cost = action_count, temporarily remove it and
        check via Dijkstra whether an alternative path u→...→v exists with
        total cost ≤ direct cost.  If so, the edge is redundant and removed.

        Returns count of removed edges.
        """
        import networkx as nx

        g = graph._g

        def _weight(u: str, v: str, data: Dict) -> int:
            # MultiDiGraph: data is {eid: {attrs}} — use shortest edge
            if "obs_str" in data:
                return _action_count(data["obs_str"])
            return min(_action_count(ed["obs_str"]) for ed in data.values())

        edges_to_remove: List[Tuple[str, str, str]] = []

        for u, v, eid, data in list(g.edges(data=True, keys=True)):
            direct_cost = _action_count(data["obs_str"])
            # Temporarily remove the direct edge
            g.remove_edge(u, v, key=eid)
            try:
                alt_cost = nx.dijkstra_path_length(g, u, v, weight=_weight)
                if alt_cost <= direct_cost:
                    edges_to_remove.append((u, v, eid))
                else:
                    # Restore — not redundant
                    g.add_edge(u, v, key=eid, **data)
            except nx.NetworkXNoPath:
                # No alternative — restore
                g.add_edge(u, v, key=eid, **data)

        return len(edges_to_remove)

    def _filter_graph(
        self,
        graph: BaseGraph,
        images_dir: Path,
        void_threshold: float = 0.7,
        std_threshold: float = 10.0,
    ) -> int:
        """
        Remove nodes whose images all fail quality check.
        Directly removes nodes and their incident edges (no bypass).
        Returns count of removed nodes.
        """
        g = graph._g
        nodes_to_remove: Set[str] = set()

        for nid, attrs in list(g.nodes(data=True)):
            img_paths = attrs.get("image_paths", [])
            if not img_paths:
                nodes_to_remove.add(nid)
                continue
            # Keep node if ANY image passes
            keep = False
            for img_rel in img_paths:
                img_abs = images_dir.parent / img_rel  # images/ is relative to graph_dir
                if _image_passes_filter(
                    str(img_abs),
                    void_threshold=void_threshold,
                    std_threshold=std_threshold,
                ):
                    keep = True
                    break
            if not keep:
                nodes_to_remove.add(nid)

        if not nodes_to_remove:
            return 0

        # Remove bad nodes (and their edges) + clean up orphaned image files
        orphaned_images: List[Path] = []
        for nid in nodes_to_remove:
            for img_rel in g.nodes[nid].get("image_paths", []):
                img_file = images_dir.parent / img_rel
                if img_file.exists():
                    orphaned_images.append(img_file)
            g.remove_node(nid)

        for img_file in orphaned_images:
            img_file.unlink(missing_ok=True)

        return len(nodes_to_remove)
