import sys
import subprocess
from pathlib import Path
from typing import List, Sequence
from datasets import load_dataset
from view_suite.scannet.download_scannet_batch import download_scannet_batch
import fire

def collect_scannet_scene_ids_1(
) -> List[str]:
    ds = load_dataset("lidingm/ViewSpatial-Bench")
    scene_ids=[]
    for i in range(len(ds["test"])):
        image_path = ds["test"][i]["image_path"][0]
        data_type=image_path.split("/")[1]
        scene_id=image_path.split("/")[2]
        if data_type=="scannetv2_val":
            scene_ids.append(scene_id)
    return list(set(scene_ids))

def collect_scannet_scene_ids_2(
) -> List[str]:
    ds = load_dataset("nyu-visionx/VSI-Bench")
    scene_ids=[]
    for i in range(len(ds["test"])):
        item = ds["test"][i]
        if item["dataset"]=="scannet":
            scene_id=item.get("scene_name", "")
            if scene_id.startswith("scene"):
                scene_ids.append(scene_id)
    scene_ids=list(set(scene_ids))
    print(f"Collected {len(scene_ids)} unique ScanNet scene IDs from VSI-Bench dataset.")
    return list(set(scene_ids))

def collect_scannet_scene_ids():
    scene_ids=collect_scannet_scene_ids_1()+collect_scannet_scene_ids_2()
    scene_ids=list(set(scene_ids))
    print(f"Collected {len(scene_ids)} unique ScanNet scene IDs in total.")
    return scene_ids
    
    
def download_scannet_data(
    scannet_download_script_path: str,
    out_dir: str="data/viewagent_scannet",
    timeout_s: int = 900,
):
    download_scannet_batch(
        scannet_download_script_path=scannet_download_script_path,
        out_dir=out_dir,
        scene_ids=collect_scannet_scene_ids(),
        file_types=["_vh_clean.ply"],
        timeout_s=timeout_s,
        verbose=True,
    )


if __name__ == "__main__":
    fire.Fire(download_scannet_data)