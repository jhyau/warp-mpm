import open3d as o3d
import os
import matplotlib.pyplot as plt
import numpy as np

from time import sleep
from tqdm import tqdm

frames = 200

dir_path = "./sim_results/jelly_fluid_g_neg_neutral"

def rotate_view(vis):
    ctr = vis.get_view_control()
    ctr.rotate(10.0, 0.0)
    return False

image_path = os.path.join(dir_path, 'image')
if not os.path.exists(image_path):
    os.makedirs(image_path)

print(f"dir_path: {dir_path}")
print("creating visualization window")
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.set_full_screen(True)

print("creating first point cloud")
pcd = o3d.io.read_point_cloud(f'{dir_path}/sim_0000000000.ply')
vis.add_geometry(pcd)

# The field of view (FoV) can be set to a degree in the range [5,90]. Note that change_field_of_view adds the specified FoV to the current FoV. By default, the visualizer has an FoV of 60 degrees
fov_step = 30.0
ctr = vis.get_view_control()
#print("Field of view (before changing) %.2f" % ctr.get_field_of_view())
#ctr.change_field_of_view(step=fov_step)
#print("Field of view (after changing) %.2f" % ctr.get_field_of_view())
ctr.rotate(0.0, -300.0)
image = vis.capture_screen_float_buffer(False)
vis.poll_events()
vis.update_renderer()
plt.imsave(os.path.join(image_path, '{:05d}.png'.format(0)),
                       np.asarray(image),
                       dpi=1)

print("drawing the rest of the point clouds")
for i in tqdm(range(1, frames)):
    pcd.points = o3d.io.read_point_cloud(f'{dir_path}/sim_{i:010d}.ply').points
    vis.update_geometry(pcd)
    image = vis.capture_screen_float_buffer(False)
    #ctr.rotate(0.0, 10.0)
    #ctr = vis.get_view_control()
    #print("Field of view (before changing) %.2f" % ctr.get_field_of_view())
    #ctr.change_field_of_view(step=fov_step)
    #print("Field of view (after changing) %.2f" % ctr.get_field_of_view())
    #ctr.rotate(10.0, 0.0) 
    vis.poll_events()
    vis.update_renderer()
    plt.imsave(os.path.join(image_path, '{:05d}.png'.format(i)),
                       np.asarray(image),
                       dpi=1)
