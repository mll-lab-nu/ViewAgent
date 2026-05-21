# ViewSuite Interactive View Planning Environment

Multimodal active-exploration env for GraphRL. VAGEN agent navigates 3D
ScanNet scenes; the TrajToSFT phase converts those rollouts into a
camera-pose graph and emits seven LLaMA-Factory datasets.

## Inheritance

```
Layer 0 (ABC)         Layer 1 (VAGEN defaults)        Layer 2 (env-specific)
─────────────         ────────────────────────        ──────────────────────
NodeData          ◄── VagenNodeData               ◄── ViewSuiteNodeData
EdgeData          ◄── VagenEdgeData                   (default — no override)
VagenGraphBuilder                                  ◄── InteractiveViewPlanningGraphBuilder
TrajToSFTGraphBase                                 ◄── InteractiveViewPlanningTrajToSFT
```

Differences from the text-only
[sokoban_text](../../sokoban/sokoban_text/README.md) reference:

- **Multimodal** — nodes carry observation images (`source_images` →
  `image_paths`).
- **Custom node dedup** — `ViewSuiteNodeData` overrides all three dedup
  methods so two close camera poses in the same scene collapse to one node.
- **Image-quality filter** — overrides `convert_files()` to remove void /
  uniform-image nodes after graph build.
- **Scene-scoped MCQ** — forward-dynamics negatives are drawn from the same
  scene only.

---

## Graph builder ([`interactive_view_planning_graph_builder.py`](interactive_view_planning_graph_builder.py))

### Node design

```python
class ViewSuiteNodeData(VagenNodeData):
    state = {"scene_id": "scene0030_00", "pose": {"tx": 2.96, "ty": 2.88, ...}}
    obs_str = "[tx=2.9622, ty=2.8836, tz=1.4770, rx=-120.00°, ry=0.00°, rz=150.00°]"
    source_images = ["/abs/rollout/image_0/images_5/0.png"]   # before copy
    image_paths   = ["images/abc123_0.png"]                    # after copy
    extra = {"scene_id": "scene0030_00"}
```

### Custom dedup (Layer 2)

```python
class ViewSuiteNodeData(VagenNodeData):
    def unique_key(self):
        # md5(scene_id + pose_at_4_decimal_places)
        return hashlib.md5(f"{scene_id}|{pose_4dp}".encode()).hexdigest()[:16]

    def bucket_key(self):
        return self.state["scene_id"]                           # scene-scoped bucket

    def is_similar_to(self, other):
        # position distance < 0.05m AND each angle delta < 5°
        return pos_err < 0.05 and all(angle_err < 5.0)
```

Because `ViewSuiteNodeData` has a custom dedup, the builder overrides
`_make_node_data()` so merge/dedup reconstructs the correct class:

```python
class InteractiveViewPlanningGraphBuilder(VagenGraphBuilder):
    def _make_node_data(self, ndata):
        return ViewSuiteNodeData(
            state=ndata["state"], obs_str=ndata.get("obs_str"), ...,
        )
```

### Edge design

Default `VagenEdgeData` works as-is — the action text (e.g. `"turn_left"`
or `"turn_left | move_forward"`) is the dedup key.

### Conversation parsing (`traj_to_transitions`)

Multi-turn rollout shape:

```
user[0]:  initial view <image>, top-down <image>, target <image>, pose [tx=..., ry=...]
asst[0]:  <action>turn_left</action>
user[1]:  new pose [tx=..., ...], <image>
asst[1]:  <action>move_forward</action>
…
asst[N]:  <action>answer(tx, ty, tz, rx, ry, rz)</action>      # skipped (not a movement)
```

Parser:
1. Counts `<image>` placeholders globally to map indices to image files at
   `rollout_dir/image_{step_idx}/images_{line_idx}/{idx}.{png,jpg}`.
2. Pulls 6-DoF pose from each user message via regex.
3. Pulls actions from `<action>...</action>` tags, skipping `answer(...)`.
4. Returns `[(ViewSuiteNodeData, VagenEdgeData, ViewSuiteNodeData), ...]`.

### Image quality filter (`convert_files` override)

After `_build_sequential` / `_build_parallel`:

1. **Per-node check** — drop a node if ALL its images fail:
   - void-pixel ratio > 70% (pixels near gray=235), or
   - grayscale std-dev < 10 (nearly uniform).
2. **Bypass edges** — for each removed node N, connect each predecessor to
   each successor with combined action text.
3. Save filtered graph.

```yaml
graph_builder:
  filter:
    void_threshold: 0.7    # max ratio of void pixels
    std_threshold: 10.0    # min grayscale std-dev
```

---

## TrajToSFT subclass ([`traj_to_sft.py`](traj_to_sft.py))

Standard `TrajToSFTGraphBase` subclass — supplies the builder and the
dataset generators:

```python
from graphrl import TrajToSFTGraphBase

class InteractiveViewPlanningTrajToSFT(TrajToSFTGraphBase):
    name = "TrajToSFT(viewsuite_interactive_view_planning)"

    def graph_builder_class(self):
        return InteractiveViewPlanningGraphBuilder

    def _dataset_cfg(self):
        # Override hook so RandomActionTrajToSFT can re-target self.config["sft"]
        return self.config

    def generate_datasets(self, graph, images_dir):
        # …call seven sft-generator helpers with cfg + child_rng()
```

The subclassed `_dataset_cfg()` hook lets `viewsuite_random_sft_rl`
inherit and re-target the config dict (it nests dataset config under
`self.config["sft"]` because the eval-collection knobs sit at the top
level of `traj_to_sft:`).

### Datasets ([`utils/sft_generators/`](utils/sft_generators/))

Seven datasets, all in ShareGPT message format with image references:

| Dataset | Task | Images |
|---|---|---|
| `action_gen` | given initial + target views, predict action sequence | from, to, (top_down) |
| `path_to_view` | given initial view + actions, pick result MCQ A–D | from, (top_down), 4 options |
| `multi_turn_action_gen` | step-by-step navigation from initial → target | initial, target, (top_down), per-step |
| `multi_turn_action_gen_mcq` | per-step MCQ over actions | initial + per-step + options |
| `multi_turn_action_gen_mix` | mixed open-ended / MCQ per turn | … |
| `view_difference` | numerical: predict #actions between two views | from, to |
| `view_difference_mcq` | MCQ over #actions | from, to + options |

All datasets use 5 diversified system-prompt variants (randomly sampled
per record). Forward-dynamics negatives use
`graph.get_random_nodes(filter_fn=...)` to draw from the same scene.

---

## Pipeline.yaml

```yaml
general_overrides:
  rl:
    training_steps: 600
    vagen_dir: VAGEN
    hydra_overrides:
      data: { train_files: ..., val_files: ... }
      # …VAGEN/verl knobs…

  traj_to_sft:
    module: graphrl.envs.viewsuite.viewsuite_interactive_view_planning.InteractiveViewPlanningTrajToSFT
    viewsuite_15k_dir: ${oc.env:HOME}/projects/viewsuite/data/viewsuite_15k
    graph_builder:
      num_workers: 4
      filter: { void_threshold: 0.7, std_threshold: 10.0 }
    generators:
      - action_gen
      - path_to_view
      - multi_turn_action_gen
      - multi_turn_action_gen_mcq
      - multi_turn_action_gen_mix
      - view_difference
      - view_difference_mcq
    action_gen:                { min_path_len: 1, max_path_len: 3, sample_per_scene: 15 }
    path_to_view:          { min_path_len: 1, max_path_len: 3, sample_per_scene: 15 }
    multi_turn_action_gen:     { min_path_len: 3, max_path_len: 5, sample_per_scene: 20, oversample: 10 }
    multi_turn_action_gen_mcq: { min_path_len: 1, max_path_len: 3, sample_per_scene: 15 }
    multi_turn_action_gen_mix: { min_path_len: 3, max_path_len: 5, sample_per_scene: 20, mcq_prob: 0.5 }
    view_difference:           { min_path_len: 2, max_path_len: 5, sample_per_scene: 15 }
    view_difference_mcq:       { min_path_len: 2, max_path_len: 5, sample_per_scene: 15 }
    seed: 42

  sft:
    n_gpus: 8
    hydra_overrides:
      stage: sft
      template: qwen2_vl
      # …LLaMA-Factory knobs…
```

---

## File structure

```
graphrl/envs/viewsuite/viewsuite_interactive_view_planning/
├── __init__.py                       # re-exports InteractiveViewPlanningTrajToSFT for the dotted-path
├── README.md                         # this file
├── interactive_view_planning_graph_builder.py   # InteractiveViewPlanningGraphBuilder + ViewSuiteNodeData
├── traj_to_sft.py                    # InteractiveViewPlanningTrajToSFT
└── utils/
    ├── __init__.py
    └── sft_generators/
        ├── action_gen.py
        ├── path_to_view.py
        ├── helpers.py
        ├── multi_turn_action_gen.py
        ├── multi_turn_action_gen_mcq.py
        ├── multi_turn_action_gen_mix.py
        ├── view_difference.py
        └── view_difference_mcq.py
```
