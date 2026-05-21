from __future__ import annotations
from typing import Any, Dict

from view_suite.envs.scannet_proxy_task.gym_proxy_no_tool import GymProxyNoTool


class View2Path(GymProxyNoTool):

    async def system_prompt(self) -> Dict[str, Any]:
        """Simple one-line system prompt."""
        return {"obs_str": "You're a spatial reasoning agent."}

    async def reset(self, seed: int):
        """
        Override reset to include task instructions and examples in user prompt.
        """
        obs, info = await super().reset(seed)

        # Build the instruction that was originally in system prompt
        instruction = (
            "You are given a question and a set of images. You need to answer the question based on the images.\n"
            "Given x be one of the choices A|B|C|D, your answer should be in the format of: <think>...</think><action>answer(x)|</action>"
        )

        # Get examples
        example_prompt = self._build_example_prompt(self.example_count)

        # Prepend instruction and examples to the original user prompt
        obs["obs_str"] = instruction + "\n\n" + example_prompt + "\n\n" + obs["obs_str"]

        return obs, info
    def _build_example_prompt(self, example_count: int) -> str:
        
        examples = [
            """Example 1:
    USER: Given the initial view [image] and a top-down reference [image], which action sequence will reach the target view [image]?
    (Action semantics: translation step = 0.5 m; yaw step = 30.0°; pitch step = 15.0°.)
    A. [turn_left, move_forward, move_forward, move_right]
    B. [move_right, move_forward, turn_left, move_forward]
    C. [turn_right, move_forward, move_forward, move_left]
    D. [move_left, move_forward, turn_right, move_forward]

    ASSISTANT: <think>The target object is slightly right of center in the initial view but centered and larger in the target. The top-down reference shows the agent starts left of the object’s centerline and facing slightly past it. We must first move right to align laterally, then move forward to reduce distance. After approaching, a small left yaw correction centers the object. Option B performs right translation, forward motion, yaw correction, then final approach. Other options either rotate too early or move in the wrong lateral direction.</think><action>answer(B)|</action>""",

            """Example 2:
    USER: Given the initial view [image] and a top-down reference [image], which action sequence will reach the target view [image]?
    (Action semantics: translation step = 0.5 m; yaw step = 30.0°; pitch step = 15.0°.)
    A. [look_down, move_forward, move_forward, look_down]
    B. [look_up, move_forward, move_forward, look_up]
    C. [move_forward, move_forward, look_down, look_down]
    D. [move_forward, look_up, move_forward, look_up]

    ASSISTANT: <think>The target emphasizes a floor object with less ceiling visible, indicating downward pitch. The scale increase suggests forward motion. Applying look_down before and after forward steps ensures proper vertical framing throughout the approach. Option A matches both the scale change and the downward tilt. Other options either tilt upward or delay pitch excessively.</think><action>answer(A)|</action>""",

            """Example 3:
    USER: Given the initial view [image] and a top-down reference [image], which action sequence will reach the target view [image]?
    (Action semantics: translation step = 0.5 m; yaw step = 30.0°; pitch step = 15.0°.)
    A. [move_up, move_up, move_forward, look_down]
    B. [move_down, move_down, move_forward, look_up]
    C. [move_up, move_forward, move_forward, look_up]
    D. [move_down, move_forward, move_forward, look_down]

    ASSISTANT: <think>The initial view is partially occluded by a low object. In the target, more surface area is visible and occlusion is reduced, implying increased camera height. After raising height, a slight look_down is required to keep the surface centered. Forward motion accounts for the larger object scale. Option A increases height, approaches, then adjusts pitch downward. Other options either reduce height or tilt incorrectly.</think><action>answer(A)|</action>""",

            """Example 4:
    USER: Given the initial view [image] and a top-down reference [image], which action sequence will reach the target view [image]?
    (Action semantics: translation step = 0.5 m; yaw step = 30.0°; pitch step = 15.0°.)
    A. [turn_right, turn_right, move_forward, move_forward]
    B. [turn_left, turn_left, move_forward, move_forward]
    C. [turn_right, move_forward, turn_right, move_forward]
    D. [turn_left, move_forward, turn_left, move_forward]

    ASSISTANT: <think>The target shows a doorway previously behind-right now centered and closer. The top-down reference indicates a 60° right yaw aligns the forward direction with the doorway. After completing yaw adjustment, forward translation increases proximity. Completing rotation before translation avoids lateral drift. Option A satisfies this alignment and approach.</think><action>answer(A)|</action>""",

            """Example 5:
    USER: Given the initial view [image] and a top-down reference [image], which action sequence will reach the target view [image]?
    (Action semantics: translation step = 0.5 m; yaw step = 30.0°; pitch step = 15.0°.)
    A. [move_left, move_left, move_forward]
    B. [move_right, move_right, move_forward]
    C. [move_forward, move_right, move_right]
    D. [move_forward, move_left, move_left]

    ASSISTANT: <think>The painting appears right-of-center initially and centered in the target. This implies leftward lateral alignment without yaw change. The top-down map confirms the agent starts right of the painting’s normal axis. Two left steps align horizontally, and one forward step increases scale. Option A preserves symmetry and matches the final composition.</think><action>answer(A)|</action>""",

            """Example 6:
    USER: Given the initial view [image] and a top-down reference [image], which action sequence will reach the target view [image]?
    (Action semantics: translation step = 0.5 m; yaw step = 30.0°; pitch step = 15.0°.)
    A. [look_up, turn_left, move_forward, move_forward]
    B. [turn_left, move_forward, move_forward, look_up]
    C. [look_down, turn_left, move_forward, move_forward]
    D. [turn_right, move_forward, move_forward, look_up]

    ASSISTANT: <think>The bookshelf top becomes visible in the target while ceiling visibility increases slightly. This implies pitching up after navigation. The top-down reference shows the bookshelf lies front-left. A left yaw aligns toward it, forward motion reduces distance, and a final look_up matches vertical framing. Option B performs navigation first then pitch adjustment, which best matches final framing.</think><action>answer(B)|</action>""",

            """Example 7:
    USER: Given the initial view [image] and a top-down reference [image], which action sequence will reach the target view [image]?
    (Action semantics: translation step = 0.5 m; yaw step = 30.0°; pitch step = 15.0°.)
    A. [move_forward, move_forward, move_forward]
    B. [move_backward, move_backward, move_forward]
    C. [move_forward, move_right, move_forward]
    D. [move_forward, move_left, move_forward]

    ASSISTANT: <think>The target maintains identical orientation and symmetry but increases object scale. No yaw or lateral translation is needed. Pure forward motion preserves perspective consistency while reducing distance. Option A performs three forward steps and best matches the target proximity.</think><action>answer(A)|</action>""",

            """Example 8:
    USER: Given the initial view [image] and a top-down reference [image], which action sequence will reach the target view [image]?
    (Action semantics: translation step = 0.5 m; yaw step = 30.0°; pitch step = 15.0°.)
    A. [move_right, move_forward, move_forward, turn_left]
    B. [move_right, move_forward, move_forward, turn_right]
    C. [move_left, move_forward, move_forward, turn_left]
    D. [move_left, move_forward, move_forward, turn_right]

    ASSISTANT: <think>The agent starts too close to the right wall and slightly misaligned with the doorway axis. The target shows better centering and reduced right-wall dominance. Moving left re-centers the agent in the corridor. Forward motion approaches the doorway. A slight right yaw at the end corrects final alignment. Option D matches this sequence precisely.</think><action>answer(D)|</action>""",

            """Example 9:
    USER: Given the initial view [image] and a top-down reference [image], which action sequence will reach the target view [image]?
    (Action semantics: translation step = 0.5 m; yaw step = 30.0°; pitch step = 15.0°.)
    A. [move_down, move_forward, move_forward, look_up]
    B. [move_up, move_forward, move_forward, look_down]
    C. [move_down, move_forward, move_forward, look_down]
    D. [move_up, move_forward, move_forward, look_up]

    ASSISTANT: <think>The target reveals more tabletop surface and reduces near-edge occlusion, suggesting increased height. After raising height, forward movement reduces distance. A downward pitch restores proper framing of the table surface. Option B increases height, approaches, and then pitches downward to match the target view.</think><action>answer(B)|</action>""",

            """Example 10:
    USER: Given the initial view [image] and a top-down reference [image], which action sequence will reach the target view [image]?
    (Action semantics: translation step = 0.5 m; yaw step = 30.0°; pitch step = 15.0°.)
    A. [turn_left, move_forward, move_right, move_forward]
    B. [turn_right, move_forward, move_left, move_forward]
    C. [turn_left, move_forward, move_left, move_forward]
    D. [turn_right, move_forward, move_right, move_forward]

    ASSISTANT: <think>The cabinet lies front-right in the initial pose. The target centers and enlarges it. A right yaw aligns heading toward the cabinet. A forward step reduces distance. A right translation aligns laterally with its centerline. A final forward step matches scale. Option D performs exactly this alignment-approach sequence.</think><action>answer(D)|</action>""",
        ]
           
        return "\n\n".join(examples[:example_count])


if __name__ == "__main__":
    # Interactive demo -- play the V2P (View-to-Path) task yourself.
    #
    # V2P needs NO render service: it is a single-turn multiple-choice task
    # whose images are pre-rendered in the jsonl. Run (from the repo root, or
    # with VIEWSUITE_ROOT exported):
    #     python view_suite/envs/scannet_proxy_task/view_to_path.py
    #
    # Each question's images are saved under --save_dir so you can look at
    # them; then type your answer -- only A, B, C or D is accepted.
    import asyncio
    import os

    import fire

    async def _play(
        jsonl_path: str = "",
        save_dir: str = "",
        seed: int = 0,
        format: str = "free_think",
    ) -> None:
        jsonl_path = jsonl_path or os.path.join(
            os.environ.get("VIEWSUITE_ROOT", "."),
            "data/viewsuite_15k/view_to_path_test.jsonl",
        )
        save_dir = os.path.abspath(save_dir or os.path.join(
            os.environ.get("VIEWSUITE_ROOT", "."), "tests/v2p_play"))
        os.makedirs(save_dir, exist_ok=True)
        # example_count=0: keep the play screen clean (no in-prompt examples).
        env = View2Path(
            {"jsonl_path": jsonl_path, "format": format, "example_count": 0}
        )

        print(f"[V2P demo] jsonl={jsonl_path}")
        print(f"[V2P demo] question images are saved under: {save_dir}/")
        print("[V2P demo] answer with A / B / C / D  (or 'quit' to exit)\n")

        while True:
            obs, _ = await env.reset(seed=seed)
            imgs = (obs.get("multi_modal_input") or {}).get("<image>", [])
            for i, img in enumerate(imgs):
                img.save(os.path.join(save_dir, f"q{seed}_img{i}.png"))

            print("=" * 64)
            print(obs["obs_str"])
            print(f"[{len(imgs)} image(s) saved: {save_dir}/q{seed}_img*.png]")

            try:
                ans = input("Your answer [A/B/C/D, or quit]: ").strip()
            except EOFError:
                print()
                break
            if ans.lower() in {"quit", "exit"}:
                break
            letter = ans.upper()[:1]
            if letter not in {"A", "B", "C", "D"}:
                print(f"  '{ans}' is invalid -- only A, B, C or D is allowed.\n")
                continue

            obs, reward, done, info = await env.step(
                f"<think>play</think><action>answer({letter})|</action>"
            )
            verdict = "CORRECT" if info.get("success") else "WRONG"
            print(f"  -> you answered {letter}: {verdict}  (reward={reward})\n")
            seed += 1

        await env.close()
        print("[V2P demo] bye.")

    fire.Fire(lambda **kw: asyncio.run(_play(**kw)))
