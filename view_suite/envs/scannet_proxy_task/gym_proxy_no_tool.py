# gym_path_to_view.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union,Sequence

from PIL import Image
from view_suite.envs.base.view_base_env import ViewBaseEnv  # type: ignore

from view_suite.envs.utils.jsonl_utils import (
    resolve_rel_image,
    count_lines,
    read_jsonl_line_by_index,
)
from view_suite.envs.utils.image_utils import safe_open_rgb
from view_suite.envs.utils.parse_utils import parse_no_tool_action_str, get_format_instruction


def build_system_prompt(format_name: str = "free_think") -> str:
    """Global system prompt for the no-tool QA environment."""
    fmt_instr = get_format_instruction(format_name, action_example="answer(x)", action_description="where x is A, B, C, or D.")
    return (
        "You are a spatial reasoning agent. You are given a question and a set of images. "
        "You need to answer the question based on the images.\n"
        f"{fmt_instr}"
    ).strip()




class GymProxyNoTool(ViewBaseEnv):
    """
    Single-turn multiple-choice environment (forward-dynamics flavor).

    Same free_think I/O contract as ViewSpatialNoToolGym:
        <think>...</think><action>answer(xxx)|</action>

    Reward calculation:
      - format_reward (default 0.2): correct free_think format + valid action list +
        first action is answer(x)
      - answer_reward (default 0.8): answer(x) matches gold (supports text / letter / index)

    Config:
      - jsonl_path: str (required)
      - total_lines: Optional[int]
      - dataset_root: Optional[str]
      - format_reward: float = 0.2
      - answer_reward: float = 0.8
      - image_size: Optional[List[int, int] | Tuple[int, int]] = (512, 512)   # if None, no resizing

    Notes:
      - Mirrors ViewSpatialNoToolGym's behavior (first action must be answer(...)),
        but allows customizing prompts or logging separately.
    """

    def __init__(self, env_config: Dict[str, Any]):
        super().__init__(env_config)

        self.jsonl_path = Path(self.config["jsonl_path"])
        assert self.jsonl_path.is_file(), f"jsonl not found: {self.jsonl_path}"

        # Optional: caller can pass precomputed total_lines to avoid recounting
        self.total_lines: Optional[int] = self.config.get("total_lines", None)
        if self.total_lines is None:
            self.total_lines = count_lines(self.jsonl_path)
        assert self.total_lines and self.total_lines > 0, "empty jsonl"

        self.dataset_root: Optional[str] = self.config.get("dataset_root", None)

        self.format_reward: float = float(self.config.get("format_reward", 0.1))
        self.answer_reward: float = float(self.config.get("answer_reward", 1.0))
        self.image_size: Optional[Sequence[int]] = self.config.get("image_size", None)
        self.is_snap_every_step = bool(env_config.get("is_snap_every_step", True))

        # Per-episode state
        self.current_item: Optional[Dict[str, Any]] = None
        self.images: List[Image.Image] = []
        self.episode_done: bool = False
        self.example_count: int = int(self.config.get("example_count", 5))
        self.no_image_mode: bool = self.config.get("no_image_mode", False)
        self.format: str = self.config.get("format", "free_think")
        

    # ---------------- Internal Helpers ----------------


    # ---------------- Lifecycle ----------------

    async def close(self) -> None:
        return None
    def _build_example_prompt(self, example_count: int) -> str:
        
        examples = [
            """Example 1:
    USER: Given the initial view [image] and a top-down reference [image], which action sequence will reach the target view [image]?
    (Action semantics: translation step = 0.5 m; rotation step = 30.0 degrees per step.)
    A. [turn_left, turn_left, move_backward, move_right, move_right]
    B. [turn_left, turn_left, move_down, move_right]
    C. [turn_left, turn_right, move_backward, move_right, move_backward]
    D. [turn_left, turn_left, move_right, move_forward, move_right]

    ASSISTANT: <think>The initial view shows a bed on the right and a nightstand in the middle. The target view shows the nightstand centered between two beds. From the top-down reference, the camera needs to move to the right and slightly backward to center the nightstand between the beds. The correct sequence should involve turning to face the nightstand directly and then moving to position it centrally. Option A involves turning left twice, which aligns the view towards the nightstand, and then moving backward and right to center it. This matches the target view.</think><action>answer(A)|</action>""",

            """Example 2:
    USER: Given the initial view [image] and a top-down reference [image], after you execute the following action sequence (translation step = 0.5 m; rotation step = 30.0 degrees per step):
    [turn_right, move_forward, move_left, move_left, move_left, move_up]
    which of the following images corresponds to the result?
    A. [image]
    B. [image]
    C. [image]
    D. [image]

    ASSISTANT: <think>The initial view shows a bathroom stall with a toilet and a partition. The top-down reference shows the layout of the bathroom.

    1. turn_right: The view rotates 30 degrees to the right.
    2. move_forward: Move 0.5 meters forward, likely towards the toilet.
    3. move_left (*3): Move 1.5 meters left total, likely facing the wall.
    4. move_up: Move upwards, possibly looking over the partition.

    The resulting view should be of the toilet area from a higher perspective. Image B matches this expected result.</think><action>answer(B)|</action>""",

            """Example 3:
    SYSTEM: You are a spatial reasoning agent. You are given a question and a set of images. You need to answer the question based on the images.
    Given x be one of the choices A|B|C|D, your answer should be in the format of: <think>...</think><action>answer(x)|</action>

    USER: Given the initial view [image] and a top-down reference [image], which action sequence will reach the target view [image]?
    (Action semantics: translation step = 0.5 m; rotation step = 30.0 degrees per step.)
    A. [turn_left, turn_right, turn_left, look_down, move_right]
    B. [turn_left, look_up, turn_left, move_right]
    C. [turn_right, turn_right, turn_left, turn_left, move_right]
    D. [turn_left, turn_left, turn_left, turn_left, move_right]

    ASSISTANT: <think>The initial view shows a corner with a dispenser and a chair. The target view shows a desk with items on it. In the top-down reference, the desk is located opposite the initial view's corner. To reach the desk, a 180-degree turn is needed, followed by moving to the right. Option D involves turning left four times (a 360-degree turn) and then moving right, which aligns with reaching the desk.</think><action>answer(D)|</action>""",

            """Example 4:
    USER: Given the initial view [image] and a top-down reference [image], after you execute the following action sequence (translation step = 0.5 m; rotation step = 30.0 degrees per step):
    [turn_left, turn_left, move_forward, move_forward, move_right, move_right]
    which of the following images corresponds to the result?
    A. [image]
    B. [image]
    C. [image]
    D. [image]

    ASSISTANT: <think>The initial view shows a doorway. The top-down reference indicates the layout of the room. Starting from the doorway, the sequence of actions is: turn left twice (facing the opposite direction), move forward twice (1 meter forward), and move right twice (1 meter to the right). This sequence likely places the viewer in the bathroom area. Images C and D show a bathroom, but D is closer. The correct result after the sequence is C, as it shows the bathroom from a slightly further distance.</think><action>answer(C)|</action>"""
        ]

        return "\n\n".join(examples[:example_count])
    async def system_prompt(self) -> Dict[str, Any]:
        """One-time global instruction."""
        # Currently no images are returned here. If future versions include images,
        # they should be passed through _resize_image before returning.
        system_prompt= build_system_prompt(format_name=self.format)
        example_prompt=self._build_example_prompt(self.example_count)
        return {"obs_str": system_prompt + "\n\n" + example_prompt}

    async def reset(self, seed: int):
        """
        Deterministically pick one item by idx = seed % total_lines,
        read only that JSONL line, load its images, and return prompt + images.
        """
        if seed is None:
            raise ValueError("reset(seed) requires a seed")
        idx = seed % self.total_lines

        self.episode_done = False
        self.images.clear()
        self.current_item = read_jsonl_line_by_index(self.jsonl_path, idx)
        item = self.current_item

        # Load original images
        for rel in (item.get("image_path") or []):
            img = safe_open_rgb(resolve_rel_image(self.jsonl_path, rel, self.dataset_root))
            if img is not None:
                self.images.append(img)

        # Apply resize policy to outputs (do not overwrite internal originals)
        out_imgs = self._resize_image(self.images)

        prompt_str = item["prompt"]
        if self.is_snap_every_step:
            prompt_str += (
                "\n(Note: after every rotation, the Euler angles (rx, ry, rz) are "
                "rounded to the nearest integer multiples of the rotation step along each axis.)"
            )

        obs = {
            "obs_str": prompt_str,
            "multi_modal_input": {"<image>": out_imgs},
        }
        if self.no_image_mode:
            obs["multi_modal_input"]["<image>"] = []
            obs["obs_str"] = obs["obs_str"].replace("<image>", "")
        info = {
            "scene_id": item["scene_id"],
            "sample_id": item["sample_id"],
            "jsonl_idx": idx,
        }
        return obs, info

    # ---------------- Single Step ----------------

    async def step(self, action_str: str):
        """
        Single-turn evaluation:
          1) Parse free_think container
          2) Parse action string (action1|action2|...|)
          3) Verify first action is answer(x)
          4) Compute format reward + answer reward
          5) Terminate episode
        """
        if self.episode_done:
            # If future versions return images here, use _resize_image
            return {"obs_str": "Episode done", "multi_modal_input": {"<image>": []}}, 0.0, True, {"error": "episode_done"}

        assert self.current_item is not None, "reset() must be called before step()."
        item = self.current_item

        format_ok, parsed_answer = parse_no_tool_action_str(action_str, format=self.format)
        reward = 0.0
        fmt_score = self.format_reward if format_ok else 0.0
        reward += fmt_score

        ans_correct = False
        if format_ok:
            gold = item["gt_answer"]
            ans_correct = (
                bool(parsed_answer) and bool(gold)
                and parsed_answer.strip().lower()[0] == gold.strip().lower()[0]
            )
            if ans_correct:
                reward += self.answer_reward

        self.episode_done = True
        done = True

        status = []
        status.append("format: ok" if format_ok else "format: error")
        if format_ok:
            status.append("answer: correct" if ans_correct else "answer: wrong")

        obs = {"obs_str": " | ".join(status)}
        info = {
            "raw_response": action_str,
            "parsed_answer": parsed_answer,
            "answer_correct": ans_correct,
            "format_reward": fmt_score,
            "answer_reward": self.answer_reward if ans_correct else 0.0,
            "total_reward": reward,
            "success": ans_correct,
        }
        return obs, reward, done, info


if __name__ == "__main__":
    import fire
    import os
    import asyncio

    async def main_async(jsonl_path: str, save_path: str = "./test_fwd", resize: Optional[str] = None, format: str = "free_think"):
        """
        resize argument:
          - None / "none" -> no resizing
          - "300,300"     -> parsed as (300,300)
        """
        from pathlib import Path
        Path(save_path).mkdir(parents=True, exist_ok=True)

        env = GymProxyNoTool({"jsonl_path": jsonl_path, "resize": resize if resize is not None else (300, 300), "format": format})
        obs = await env.system_prompt()
        print("---- System Prompt ----")
        print(obs["obs_str"])

        obs, info = await env.reset(seed=0)
        print("---- Reset Prompt ----")
        print(obs["obs_str"])

        # Save (possibly resized) images for inspection
        imgs = (obs.get("multi_modal_input") or {}).get("<image>", [])
        for i, img in enumerate(imgs or []):
            img.save(os.path.join(save_path, f"initial_{i}.png"))

        user = input("Enter your choice (e.g., A): ").strip()
        action_input = f"<think>...</think><action>answer({user})|</action>"
        print("\n---- Submitted ----")
        print(action_input)

        obs, reward, done, info = await env.step(action_input)

        print("\n---- Result ----")
        print(f"Reward: {reward}, Done: {done}")
        print("Info:", info)
        print("Observation:")
        print(obs['obs_str'])

    def main(
        jsonl_path: str = "data/viewsuite_15k/path_to_view_dev_filter.jsonl",
        save_path: str = "./test_fwd",
        resize: Optional[str] = None,
        format: str = "free_think",
    ):
        asyncio.run(main_async(jsonl_path, save_path, resize, format))

    fire.Fire(main)
