# gym_mindcube_no_tool.py
# -*- coding: utf-8 -*-
"""
Single-turn QA environment for the MindCube spatial-reasoning dataset.

Uses raw MindCube data (MindCube_train.jsonl / MindCube_tinybench.jsonl).

Response format (free_think):
    <think>...</think><action>answer(x)</action>

In eval_mode only the <action> tag is required (lenient parsing).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from view_suite.envs.base.view_base_env import ViewBaseEnv
from view_suite.envs.mindcube.prompt import (
    build_system_prompt,
    build_example_prompt,
    build_reset_prompt,
)
from view_suite.envs.utils.jsonl_utils import (
    count_lines,
    read_jsonl_line_by_index,
    resolve_rel_image,
)
from view_suite.envs.utils.image_utils import safe_open_rgb
from view_suite.envs.utils.parse_utils import parse_no_tool_action_str


# --------------- JSONL file resolution ---------------

_SPLIT_FILE = {
    "train": "raw/MindCube_train.jsonl",
    "dev":   "raw/MindCube_tinybench.jsonl",
    "test":  "raw/MindCube_tinybench.jsonl",
}


# --------------- Gym environment ---------------


class MindCubeNoToolGym(ViewBaseEnv):
    """
    Single-turn QA environment for MindCube (raw data).

    Response format (free_think standard)::

        <think>...</think><action>answer(x)|</action>

    In ``eval_mode`` only the ``<action>`` tag is checked (lenient).

    Config keys
    -----------
    data_path : str
        Root directory of the downloaded MindCube data
        (e.g. ``/root/projects/viewsuite/data/mindcube``).
    split : str, default ``"train"``
        ``"train"`` | ``"dev"`` / ``"test"`` (= tinybench).
    jsonl_path : str, optional
        Explicit path to a JSONL file; overrides ``data_path`` / ``split``.
    total_lines : int, optional
        Pre-computed line count (avoids re-counting in parallel workers).
    dataset_root : str, optional
        Base directory for resolving relative image paths.
        Defaults to ``data_path``.
    format_reward : float, default 0.2
        Reward for valid response format.
    answer_reward : float, default 0.8
        Reward for correct answer.
    image_size : tuple[int, int] | None, default (512, 512)
        Resize images; ``None`` keeps originals.
    example_count : int, default 0
        Number of few-shot examples in the system prompt (0-4).
        0 means no in-context learning examples.
    format : str, default "free_think"
        Response format: "free_think" (strict), "eval_mode" (lenient), or "no_think".
    """

    def __init__(self, env_config: Dict[str, Any]):
        super().__init__(env_config)

        # Resolve data file
        self.data_path: str = self.config.get("data_path", "")
        self.split: str = self.config.get("split", "train")

        jsonl_override = self.config.get("jsonl_path")
        if jsonl_override:
            self.jsonl_path = Path(jsonl_override)
        else:
            if self.split not in _SPLIT_FILE:
                raise ValueError(
                    f"Unknown split={self.split!r}. "
                    f"Choose from {list(_SPLIT_FILE.keys())}"
                )
            self.jsonl_path = Path(self.data_path) / _SPLIT_FILE[self.split]

        assert self.jsonl_path.is_file(), f"JSONL not found: {self.jsonl_path}"

        self.total_lines: Optional[int] = self.config.get("total_lines", None)
        if self.total_lines is None:
            self.total_lines = count_lines(self.jsonl_path)
        assert self.total_lines > 0, "empty JSONL"

        # Image root
        _ds_root = self.config.get("dataset_root", None)
        if _ds_root is None and self.data_path:
            _ds_root = self.data_path
        self.dataset_root: Optional[Path] = Path(_ds_root) if _ds_root else None

        # Reward weights
        self.format_reward: float = float(self.config.get("format_reward", 0.2))
        self.answer_reward: float = float(self.config.get("answer_reward", 0.8))

        # Display
        self.image_size = self.config.get("image_size", (512, 512))
        self.example_count: int = int(self.config.get("example_count", 0))
        self.format: str = self.config.get("format", "free_think")

        # Per-episode state
        self.current_item: Optional[Dict[str, Any]] = None
        self.images: List[Image.Image] = []
        self.episode_done: bool = False

    # -------------------- Lifecycle --------------------

    async def close(self) -> None:
        return None

    async def system_prompt(self) -> Dict[str, Any]:
        """One-time global instruction."""
        parts = [build_system_prompt(format_name=self.format)]
        examples = build_example_prompt(self.example_count)
        if examples:
            parts.append(examples)
        return {"obs_str": "\n\n".join(parts) + "\n"}

    async def reset(self, seed: int):
        """
        Pick one QA item by ``idx = seed % total_lines``, load images, return obs.
        """
        if seed is None:
            raise ValueError("reset(seed) requires a seed")

        idx = seed % self.total_lines
        self.episode_done = False
        self.images.clear()
        self.current_item = read_jsonl_line_by_index(self.jsonl_path, idx)
        item = self.current_item

        # Load images  (raw field: "images")
        for rel in item.get("images") or []:
            img = safe_open_rgb(
                resolve_rel_image(self.jsonl_path, rel, self.dataset_root)
            )
            if img is not None:
                self.images.append(img)

        # Build question prompt  (raw field: "question" already contains choices)
        question = (item.get("question") or "").strip()
        prompt = build_reset_prompt(question, num_images=len(self.images))

        obs = {
            "obs_str": prompt,
            "multi_modal_input": {"<image>": self._resize_image(self.images)},
        }
        info = {
            "id": item.get("id"),
            "category": item.get("category"),
            "type": item.get("type"),
            "jsonl_idx": idx,
        }
        return obs, info

    # -------------------- Single Step --------------------

    async def step(self, action_str: str):
        """
        Parse ``<think>...</think><action>answer(x)|</action>``,
        compute reward, end episode.
        """
        if self.episode_done:
            return (
                {"obs_str": "Episode done", "multi_modal_input": {"<image>": []}},
                0.0,
                True,
                {"error": "episode_done"},
            )

        assert self.current_item is not None, "reset() must be called before step()."
        item = self.current_item

        format_is_valid, parsed_answer = parse_no_tool_action_str(
            action_str, format=self.format
        )

        reward = 0.0
        format_score = self.format_reward if format_is_valid else 0.0
        reward += format_score

        answer_is_correct = False
        if format_is_valid:
            gold = (item.get("gt_answer") or "").strip()
            answer_is_correct = (
                bool(parsed_answer) and bool(gold)
                and parsed_answer.strip().lower()[0] == gold.strip().lower()[0]
            )
            if answer_is_correct:
                reward += self.answer_reward

        self.episode_done = True

        status_parts = ["format: ok" if format_is_valid else "format: error"]
        if format_is_valid:
            status_parts.append(
                "answer: correct" if answer_is_correct else "answer: wrong"
            )

        obs = {"obs_str": " | ".join(status_parts)}
        info = {
            "raw_response": action_str,
            "parsed_answer": parsed_answer,
            "answer_correct": answer_is_correct,
            "format_reward": format_score,
            "answer_reward": self.answer_reward if answer_is_correct else 0.0,
            "total_reward": reward,
            "success": answer_is_correct,
        }
        return obs, reward, True, info


# -------------------- CLI smoke-test --------------------

if __name__ == "__main__":
    import asyncio
    import os
    import fire

    async def _main_async(
        data_path: str = "/root/projects/viewsuite/data/viewagent_mindcube",
        split: str = "dev",
        jsonl_path: str = "",
        save_path: str = "./test_mindcube",
        image_size: Optional[Tuple[int, int]] = (512, 512),
        format: str = "free_think",
    ):
        Path(save_path).mkdir(parents=True, exist_ok=True)

        cfg: Dict[str, Any] = {
            "data_path": data_path,
            "split": split,
            "image_size": image_size,
            "format": format,
        }
        if jsonl_path:
            cfg["jsonl_path"] = jsonl_path

        env = MindCubeNoToolGym(cfg)

        # System prompt
        sys_obs = await env.system_prompt()
        print("=== System Prompt ===")
        print(sys_obs["obs_str"])

        # Reset
        obs, info = await env.reset(seed=0)
        print("\n=== Reset Observation ===")
        print(obs["obs_str"])
        print(f"\nInfo: {info}")

        if obs.get("multi_modal_input", {}).get("<image>"):
            for i, img in enumerate(obs["multi_modal_input"]["<image>"]):
                img.save(os.path.join(save_path, f"initial_{i}.png"))
            print(
                f"Saved {len(obs['multi_modal_input']['<image>'])} images "
                f"to {save_path}"
            )

        # Interactive step
        user_input = input("\nEnter answer letter (e.g. A) or 'quit': ").strip()
        if user_input.lower() == "quit":
            return
        action = (
            f"<think>Based on the images, I believe the answer is "
            f"{user_input}.</think><action>answer({user_input})|</action>"
        )
        print(f"\nAction: {action}")

        obs, reward, done, info = await env.step(action)
        print(f"\nReward: {reward}, Done: {done}")
        print(f"Info: {info}")
        print(f"Status: {obs['obs_str']}")

    def main(
        data_path: str = "/root/projects/viewsuite/data/viewagent_mindcube",
        split: str = "dev",
        jsonl_path: str = "",
        save_path: str = "./test_mindcube",
        format: str = "free_think",
    ):
        asyncio.run(
            _main_async(data_path, split, jsonl_path, save_path, format=format)
        )

    fire.Fire(main)
