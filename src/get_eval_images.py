import blenderproc as bproc
import argparse
import numpy as np
from pathlib import Path
import os
import shutil
import pytransform3d.rotations as pr
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="00700")
    parser.add_argument("--run_name", type=str, default="clf5_500")

    args = parser.parse_args()

    home_dir = Path(os.environ.get("CLUSTER_HOME", "/local/home/hanlonm"))
    environment = args.environment
    run_id = args.run_name
    run_name = f"{environment}_{run_id}"
    eval_dir = home_dir / "mt-matthew/eval_results"
    run_dir = eval_dir / run_name
    hloc_datasets_path = home_dir / "Hierarchical-Localization/datasets"
    environment_dataset_path = home_dir / "mt-matthew/data"
    trajectory_dirs = sorted(os.listdir(run_dir))
    image_output_dir = hloc_datasets_path / environment / "localization"
    if image_output_dir.exists():
        shutil.rmtree(image_output_dir)
    image_output_dir.mkdir(exist_ok=True)

    bproc.init()
    scene = bproc.loader.load_blend(
        str(environment_dataset_path) + f"/{environment}/{environment}.blend")
    bproc.lighting.light_surface(scene, 1.0)
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

    for trajectory_dir in trajectory_dirs:
        for path in sorted(os.listdir(run_dir / trajectory_dir)):
            path_array: np.ndarray = np.load(run_dir / trajectory_dir / path)
            file_names = []
            bproc.utility.reset_keyframes()
            for i, pose in enumerate(path_array):
                p_xyz = pose[:3]
                q_wxyz = pose[3:]
                rot = pr.matrix_from_quaternion(q_wxyz)
                rot = rot @ cam_rotation
                matrix_world = bproc.math.build_transformation_mat(p_xyz, rot)
                bproc.camera.add_camera_pose(matrix_world)
                file_name = (trajectory_dir + "_" + path[:-4] + "_wp_" +
                             str(i).zfill(3) + ".jpeg")
                file_names.append(file_name)
            data = bproc.renderer.render(None)
            print(data.keys())
            print(len(data["colors"]))
            print(len(file_names))
            for i, image_array in enumerate(data["colors"]):
                # Convert the NumPy array to PIL Image
                pil_image = Image.fromarray(image_array)

                # Save the image to disk using PIL
                pil_image.save(image_output_dir / file_names[i])


if __name__ == "__main__":
    main()
