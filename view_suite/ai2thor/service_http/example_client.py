#!/usr/bin/env python3
"""
Example client for AI2-THOR HTTP render service (AsyncUnifiedClient).

Demonstrates how to send a render request and save returned images.
"""
import asyncio
import numpy as np

from view_suite.service_http.async_client import AsyncUnifiedClient


async def main():
    # Initialize client
    client = AsyncUnifiedClient(
        base_url="http://localhost:8765",
        timeout=120.0,
    )

    try:
        # Example: Render a single frame from FloorPlan1
        scene_id = "FloorPlan1"

        # Camera pose (position and rotation in AI2-THOR format)
        pose = {
            "position": {"x": -1.25, "y": 1.0, "z": -1.0},
            "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
        }

        # Prepare request metadata
        meta = {
            "scene_id": scene_id,
            "tasks": [
                {
                    "pose": pose,
                    "width": 640,
                    "height": 480,
                    "fov": 90.0,
                },
                # You can add multiple poses to render in one request
                {
                    "pose": {
                        "position": {"x": -1.25, "y": 1.0, "z": -1.0},
                        "rotation": {"x": 0.0, "y": 45.0, "z": 0.0},
                    },
                    "width": 640,
                    "height": 480,
                    "fov": 90.0,
                },
            ],
        }

        # Send request (IMPORTANT: await)
        print(f"Rendering {len(meta['tasks'])} frames from {scene_id}...")
        response_meta, images = await client.render(meta=meta)

        # Process response
        print("Received response:")
        print(f"  Scene ID:     {response_meta.get('scene_id')}")
        print(f"  Image count:  {len(images)}")
        print(f"  Meta:         {response_meta}")

        # Save images
        for i, img in enumerate(images):
            output_path = f"output_{i}.png"
            img.save(output_path)
            print(f"  Saved: {output_path} ({img.size})")

    finally:
        # IMPORTANT: close the underlying httpx client
        await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
