from __future__ import annotations
from view_suite.envs.scannet_proxy_task.gym_proxy_no_tool import GymProxyNoTool


class Path2View(GymProxyNoTool):
    pass


if __name__ == "__main__":
    # Interactive demo -- play the P2V (Path-to-View) task yourself.
    #
    # P2V needs NO render service: it is a single-turn multiple-choice task
    # whose images are pre-rendered in the jsonl. Run (from the repo root, or
    # with VIEWSUITE_ROOT exported):
    #     python view_suite/envs/scannet_proxy_task/path_to_view.py
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
            "data/viewsuite_15k/path_to_view_test.jsonl",
        )
        save_dir = os.path.abspath(save_dir or os.path.join(
            os.environ.get("VIEWSUITE_ROOT", "."), "tests/p2v_play"))
        os.makedirs(save_dir, exist_ok=True)
        env = Path2View({"jsonl_path": jsonl_path, "format": format})

        print(f"[P2V demo] jsonl={jsonl_path}")
        print(f"[P2V demo] question images are saved under: {save_dir}/")
        print("[P2V demo] answer with A / B / C / D  (or 'quit' to exit)\n")

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
        print("[P2V demo] bye.")

    fire.Fire(lambda **kw: asyncio.run(_play(**kw)))
