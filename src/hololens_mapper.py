import blenderproc as bproc
import argparse
import numpy as np
from pathlib import Path
import os
import shutil
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt
from PIL import Image
import h5py


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="00700")

    args = parser.parse_args()

    home_dir = Path(os.environ.get("CLUSTER_HOME", "/local/home/hanlonm"))
    environment = args.environment
    environment_dataset_path = home_dir / "mt-matthew/data"
    hloc_datasets_path = home_dir / "Hierarchical-Localization/datasets"
    output_directory: Path = hloc_datasets_path / environment
    output_directory.mkdir(exist_ok=True)
    # output_directory = Path("output")

    image_output_dir: Path = hloc_datasets_path / environment / "mapping"
    image_output_dir.mkdir(exist_ok=True)
    # image_output_dir = Path("output")

    pose_file_name = Path(output_directory) / "image_poses.txt"
    pose_file = open(pose_file_name, "w")

    # [dev_LF, dev_LL, dev_RF, dev_RR]
    hololens_transforms = np.load(
        str(home_dir) + "/mt-matthew/data/hololens_transforms.npy")
    cam_rotation = np.array([[0.0, -0.0, -1.0], [-1.0, 0.0, -0.0],
                             [0.0, 1.0, 0.0]])
    bproc.init()
    scene = bproc.loader.load_blend(
        str(environment_dataset_path) + f"/{environment}/{environment}.blend")
    bproc.lighting.light_surface(scene, 1.0)
    # K = [366.8789978027344, 0.0, 240.0, 0.0, 365.71429443359375, 320.0, 0.0, 0.0, 1.0]
    camera_matrix = np.array([[366.8789978027344, 0, 240],
                              [0.0, 365.71429443359375, 320], [0, 0.0, 1.0]])
    bproc.camera.set_intrinsics_from_K_matrix(K=camera_matrix,
                                              image_width=480,
                                              image_height=640)
    bproc.renderer.set_max_amount_of_samples(5)
    bproc.renderer.set_output_format("JPEG")
    bproc.renderer.set_denoiser("OPTIX")
    bproc.renderer.enable_depth_output(activate_antialiasing=True)

    depth_h5 = h5py.File(f"output/{environment}_depth.h5", "w")


    image_names = []
    with open(
            str(environment_dataset_path) +
            f"/{environment}/mapping_trajectory.txt", "r") as f:
        frame = 0
        for line in f.readlines():
            line = [float(x) for x in line.split()]
            T_world_dev = pt.transform_from(
                pr.matrix_from_quaternion(
                    pr.quaternion_wxyz_from_xyzw(line[3:])), line[:3])
            for i, cam in enumerate(["LF", "LL", "RF", "RR"]):
                T_dev_cam = pt.transform_from_pq(hololens_transforms[i])

                T_world_base = T_world_dev @ T_dev_cam

                cam_pose = pt.pq_from_transform(T_world_base)

                image_file_name = cam + "_" + str(frame).zfill(4) + ".jpeg"
                image_names.append(image_file_name)

                pose_file.write("mapping/" + image_file_name + " " +
                                str(cam_pose[0]) + " " + str(cam_pose[1]) +
                                " " + str(cam_pose[2]) + " " +
                                str(cam_pose[4]) + " " + str(cam_pose[5]) +
                                " " + str(cam_pose[6]) + " " +
                                str(cam_pose[3]) + "\n")

                p_xyz = cam_pose[:3]
                q_wxyz = cam_pose[3:]
                rot = pr.matrix_from_quaternion(q_wxyz)
                rot = rot @ cam_rotation
                matrix_world = bproc.math.build_transformation_mat(p_xyz, rot)
                bproc.camera.add_camera_pose(matrix_world)
            frame += 1

    data = bproc.renderer.render()
    #bproc.writer.write_hdf5("output", data)

    # for i, image_array in enumerate(data["colors"]):
    #     # Convert the NumPy array to PIL Image
    #     pil_image = Image.fromarray(image_array)

    #     # Save the image to disk using PIL
    #     pil_image.save(image_output_dir / image_names[i])
    for i, depth_array in enumerate(data["depth"]):
        depth_h5.create_dataset(image_names[i], data=depth_array)
    depth_h5.close()


if __name__ == "__main__":
    main()