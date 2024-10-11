import open3d as o3d
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

from time import sleep
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("dir_path", type=str, default="./sim_results/jelly_fluid_g_10_no_shift")
parser.add_argument('--frames', type=int, default=100)
args = parser.parse_args()

frames = args.frames

#dir_path = "./sim_results/jelly_mat_only_g_neg_neutral"
#dir_path = "./sim_results/fluid_g_10_no_x_shift"
dir_path = args.dir_path

def rotate_view(vis):
    ctr = vis.get_view_control()
    ctr.rotate(10.0, 0.0)
    return False

image_path = os.path.join(dir_path, 'image')
if not os.path.exists(image_path):
    os.makedirs(image_path)

depth_path = os.path.join(dir_path, 'depth')
if not os.path.exists(depth_path):
    os.makedirs(depth_path)

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
vis.poll_events()
vis.update_renderer()
depth = vis.capture_depth_float_buffer(False)
vis.capture_screen_image(os.path.join(image_path, 'img_{:05d}.png'.format(0)))
image = vis.capture_screen_float_buffer(False)
plt.imsave(os.path.join(image_path, '{:05d}.png'.format(0)),
                       np.asarray(image),
                       dpi=1)
plt.imsave(os.path.join(depth_path, '{:05d}.png'.format(0)),
                       np.asarray(depth),
                       dpi=1)


print("drawing the rest of the point clouds")
for i in tqdm(range(1, frames)):
    pcd.points = o3d.io.read_point_cloud(f'{dir_path}/sim_{i:010d}.ply').points
    vis.update_geometry(pcd)
    #ctr.rotate(0.0, 10.0)
    #ctr = vis.get_view_control()
    #print("Field of view (before changing) %.2f" % ctr.get_field_of_view())
    #ctr.change_field_of_view(step=fov_step)
    #print("Field of view (after changing) %.2f" % ctr.get_field_of_view())
    #ctr.rotate(10.0, 0.0) 
    vis.poll_events()
    vis.update_renderer()
    depth = vis.capture_depth_float_buffer(False)
    vis.capture_screen_image(os.path.join(image_path, 'img_{:05d}.png'.format(i)))
    image = vis.capture_screen_float_buffer(False)
    plt.imsave(os.path.join(image_path, '{:05d}.png'.format(i)),
                       np.asarray(image),
                       dpi=1)
    plt.imsave(os.path.join(depth_path, '{:05d}.png'.format(i)),
                       np.asarray(depth),
                       dpi=1)
# close the window
vis.destroy_window()

# Create animation video from the saved image frames
import matplotlib.animation as animation

fig, ax = plt.subplots()
# ims is a list of lists, each row is a list of artists to draw in the
# current frame; here we are just animating one artist, the image, in
# each frame
ims = []
for i in range(frames):
    #x += np.pi / 15
    #y += np.pi / 30
    #im = ax.imshow(f(x, y), animated=True)
    #if i == 0:
    #    ax.imshow(f(x, y))  # show an initial one first
    img_array = plt.imread(os.path.join(depth_path, '{:05d}.png'.format(i)), "png")
    im = ax.imshow(img_array, animated=True)
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=1000)

# To save the animation, use e.g.
# Need ffmpeg to write out to a movie
# conda install -c conda-forge ffmpeg
ani.save(os.path.join(depth_path, "depth.mp4"))
#
# or
#
# writer = animation.FFMpegWriter(
#     fps=15, metadata=dict(artist='Me'), bitrate=1800)
# ani.save("movie.mp4", writer=writer)

plt.show()
