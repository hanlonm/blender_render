import blenderproc as bproc
import argparse
import numpy as np
from pathlib import Path
import os
import shutil
import pytransform3d.rotations as pr
from PIL import Image
from tqdm import tqdm
import uuid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="00195")
    parser.add_argument("--run_name", type=str, default="test")

    args = parser.parse_args()

    home_dir = Path(os.environ.get("CLUSTER_HOME", "/local/home/hanlonm"))
    environment = args.environment
    run_id = args.run_name
    run_name = f"{environment}_{run_id}"
    # run_dir = eval_dir / run_name
    hloc_datasets_path = home_dir / "Hierarchical-Localization/datasets"
    environment_dataset_path = home_dir / "mt-matthew/data"
    eval_dir = environment_dataset_path / environment / "best_loc"
    # trajectory_dirs = sorted(os.listdir(run_dir))
    image_output_dir = hloc_datasets_path / environment / "best_loc"
    # image_output_dir = Path("output")
    if image_output_dir.exists():
        shutil.rmtree(image_output_dir)
    image_output_dir.mkdir(exist_ok=True)

    path_file = np.load(eval_dir / "paths.npz")

    path_keys = list(path_file.keys())
    path_keys = [str(key) for key in path_keys]

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
    print()

    image_names = []
    for key in tqdm(path_keys):
        path = path_file[key]
        for i, waypoint in enumerate(path):
            for j, pose in enumerate(waypoint):
                p_xyz = pose[0][:3]
                q_wxyz = pose[0][3:]
                rot = pr.matrix_from_quaternion(q_wxyz)
                rot = rot @ cam_rotation
                matrix_world = bproc.math.build_transformation_mat(p_xyz, rot)
                bproc.camera.add_camera_pose(matrix_world)

                image_name = key + "_wp_" + str(i).zfill(3) + "_v_" + str(
                    j).zfill(3) + ".jpeg"
                image_names.append(image_name)

    temp_dir = uuid.uuid4().hex
    temp_dir = temp_dir[:8]
    data = bproc.renderer.render(output_dir=temp_dir)

    for i, image_array in enumerate(data["colors"]):
        # Convert the NumPy array to PIL Image
        pil_image = Image.fromarray(image_array)

        # Save the image to disk using PIL
        pil_image.save(image_output_dir / image_names[i])


if __name__ == "__main__":
    main()
