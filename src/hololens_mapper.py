import blenderproc as bproc
import argparse
import numpy as np
from pathlib import Path
import os
import shutil
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt
from PIL import Image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="00700")

    args = parser.parse_args()

    home_dir = Path(os.environ.get("CLUSTER_HOME", "/local/home/hanlonm"))
    environment = args.environment
    environment_dataset_path = home_dir / "mt-matthew/data"
    hloc_datasets_path = home_dir / "Hierarchical-Localization/datasets"
    image_output_dir = hloc_datasets_path / environment / "localization"

    output_directory = hloc_datasets_path / environment

    pose_file_name = Path(output_directory) / "image_poses.txt"
    pose_file = open(pose_file_name, "w")

    # [dev_LF, dev_LL, dev_RF, dev_RR]
    hololens_transforms = np.load("/local/home/hanlonm/mt-matthew/hololens_transforms.npy")
    cam_rotation = np.array([[0.0, -0.0, -1.0], [-1.0, 0.0, -0.0],
                             [0.0, 1.0, 0.0]])

    image_names = []
    with open("trajectory.txt", "r") as f:
        frame = 0
        for line in f.readlines():
            line = [float(x) for x in line.split()]
            T_world_dev = pt.transform_from(pr.matrix_from_quaternion(pr.quaternion_wxyz_from_xyzw(line[3:])), line[:3])
            for i, cam in enumerate(["LF", "LL", "RF", "RR"]):
                T_dev_cam = pt.transform_from_pq(hololens_transforms[i])

                T_world_base = T_world_dev @ T_dev_cam

                cam_pose = pt.pq_from_transform(T_world_base)

                image_file_name = cam + "_" + str(frame).zfill(4) + ".jpeg"

                pose_file.write(
                    "mapping/"
                    + image_file_name
                    + " "
                    + str(cam_pose[0])
                    + " "
                    + str(cam_pose[1])
                    + " "
                    + str(cam_pose[2])
                    + " "
                    + str(cam_pose[4])
                    + " "
                    + str(cam_pose[5])
                    + " "
                    + str(cam_pose[6])
                    + " "
                    + str(cam_pose[3])
                    + "\n"
                )
                frame += 1
            rot = rot @ cam_rotation