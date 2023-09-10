import blenderproc as bproc
import argparse
import numpy as np
from pathlib import Path
import os
import shutil
import pytransform3d.rotations as pr
from PIL import Image
import h5py
import uuid


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="test1")

    args = parser.parse_args()

    home_dir = Path(os.environ.get("BASE_DIR", "/local/home/hanlonm"))
    run_id = args.run_name
    hloc_datasets_path = home_dir / "Hierarchical-Localization/datasets"
    environment_dataset_path = home_dir / "active-viewpoint-selection/data"

    bproc.init()

    camera_matrix = np.array([[609.5238037109375, 0, 640],
                              [0.0, 610.1694946289062, 360], [0, 0.0, 1.0]])
    bproc.camera.set_intrinsics_from_K_matrix(K=camera_matrix,
                                              image_width=1280,
                                              image_height=720)
    cam_rotation = np.array([[0.0, -0.0, -1.0], [-1.0, 0.0, -0.0],
                             [0.0, 1.0, 0.0]])
    bproc.renderer.set_max_amount_of_samples(5)
    bproc.renderer.set_output_format("JPEG")
    bproc.renderer.set_denoiser("OPTIX")

    hf = h5py.File(
        str(environment_dataset_path) + f"/training_data/{run_id}.h5")
    print(hf.keys())
    for environment_key in hf.keys():
        print(environment_key)
        scene = bproc.loader.load_blend(
            str(environment_dataset_path) +
            f"/{environment_key}/{environment_key}.blend")
        bproc.lighting.light_surface(scene, 1.0)
        pose_data = hf[environment_key]["pose_data"][:]
        image_output_dir = hloc_datasets_path / environment_key / "training"
        if image_output_dir.exists():
            shutil.rmtree(image_output_dir)
        image_output_dir.mkdir(exist_ok=True)
        bproc.utility.reset_keyframes()
        file_names = []

        for i, pose in enumerate(pose_data):
            p_xyz = pose[:3]
            q_wxyz = pose[3:]
            rot = pr.matrix_from_quaternion(q_wxyz)
            rot = rot @ cam_rotation
            matrix_world = bproc.math.build_transformation_mat(p_xyz, rot)
            bproc.camera.add_camera_pose(matrix_world)
            file_name = run_id + "_" + (str(i).zfill(4) + ".jpeg")
            file_names.append(file_name)
        temp_dir = uuid.uuid4().hex
        temp_dir = temp_dir[:8]
        data = bproc.renderer.render(output_dir=temp_dir)
        for i, image_array in enumerate(data["colors"]):
            # Convert the NumPy array to PIL Image
            pil_image = Image.fromarray(image_array)

            # Save the image to disk using PIL
            pil_image.save(image_output_dir / file_names[i])

        bproc.object.delete_multiple(scene)

    print()


if __name__ == "__main__":
    main()
