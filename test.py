import blenderproc as bproc
import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--camera', default="trajectory.txt",help="Path to the camera file, should be examples/resources/camera_positions")
parser.add_argument('--output_dir', default="output", help="Path to where the final files, will be saved, could be examples/basics/basic/output")
args = parser.parse_args()

bproc.init()

import pytransform3d.rotations as pr
# load the objects into the scene
#objs = bproc.loader.load_obj(args.scene)
objs = bproc.loader.load_blend("00067.blend")


# define a light and set its location and energy level
bproc.lighting.light_surface(objs, 0.8)
# define the camera resolution
#bproc.camera.set_resolution(1280, 720)
K = np.array([[609.5238037109375, 0, 640], [0.0, 610.1694946289062, 360], [0, 0.0, 1.0]])

bproc.camera.set_intrinsics_from_K_matrix(K=K, image_width=1280,image_height=720)
cam_rotation = np.array([[0.0, -0.0, -1.0], [-1.0, 0.0, -0.0], [0.0, 1.0, 0.0]])


# width=1280,
#                     height=720,
#                     fx=609.5238037109375,
#                     fy=610.1694946289062,
#                     cx=640,
#                     cy=360

# read the camera positions file and convert into homogeneous camera-world transformation
with open(args.camera, "r") as f:
    for line in f.readlines():
        line = [float(x) for x in line.split()] 
        position, quat = line[:3], line[3:]
        reorder = [3,0,1,2]
        quat = [quat[i] for i in reorder]
        rot = pr.matrix_from_quaternion(quat)
        print(rot)
        rot = rot @ cam_rotation
        matrix_world = bproc.math.build_transformation_mat(position, rot)
        bproc.camera.add_camera_pose(matrix_world)
        
        
        

# activate normal and depth rendering
# bproc.renderer.enable_normals_output()
# bproc.renderer.enable_depth_output(activate_antialiasing=False)

# render the whole pipeline
bproc.renderer.set_max_amount_of_samples(0)
bproc.renderer.set_output_format("JPEG")
bproc.renderer.set_denoiser("OPTIX")
#bproc.renderer.set_simplify_subdivision_render(5)
data = bproc.renderer.render(args.output_dir)

# write the data to a .hdf5 container
bproc.writer.write_hdf5(args.output_dir, data)