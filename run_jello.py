import argparse
import warp as wp
from mpm_solver_warp import MPM_Simulator_WARP
from engine_utils import *
import torch
wp.init()
wp.config.verify_cuda = True


dvc = "cuda:0"

parser = argparse.ArgumentParser()
parser.add_argument("out_path", type=str, default="./sim_results/jelly_temp")
parser.add_argument('--time_steps', type=int, default=100)
args = parser.parse_args()

print("args: ", args)

mpm_solver = MPM_Simulator_WARP(10) # initialize with whatever number is fine. it will be reintialized


# You can either load sampling data from an external h5 file, containing initial position (n,3) and particle_volume (n,)
mpm_solver.load_from_sampling("sand_column.h5", n_grid = 150, device=dvc) 

# Or load from torch tensor (also position and volume)
# Here we borrow the data from h5, but you can use your own
volume_tensor = torch.ones(mpm_solver.n_particles) * 2.5e-8
position_tensor = mpm_solver.export_particle_x_to_torch()

# shift up along z-axis
# np vector of x # shape now is (n_particles, dim)
print("position tensor size: ", position_tensor.size())
print("current location: ", position_tensor[0, :])
# permute z and y so cylinder is on its side instead
position_tensor = position_tensor.index_select(1, torch.tensor([0,2,1], device=dvc))
position_tensor[:,2] = position_tensor[:,2] + 10.0
print("new location: ", position_tensor[0, :])
#mpm_solver.import_particle_x_from_torch(position)

mpm_solver.load_initial_data_from_torch(position_tensor, volume_tensor)

# Note: You must provide 'density=..' to set particle_mass = density * particle_volume

# E is Young's modulus
# nu is Poisson's ratio
material_params = {
    #'bulk_modulus': 200.0,
    #"yield_stress": 100.0,
    #"plastic_viscosity": 100.0,
    "material": "jelly",
    #"E": 1e5,
    "E": 1.5e3,
    #"nu": 0.3,
    "nu": 0.4,
    'friction_angle': 35,
    'g': [0.0, 0.0, -4.0],
    "density": 400.0
}
mpm_solver.set_parameters_dict(material_params)

# add bounding box for fluid
box_length = 0.4
mpm_solver.add_surface_collider((0.0, 0.0, 0.13), (0.0,0.0,1.0), 'cut', 0.0)
mpm_solver.add_surface_collider((0.5-box_length/2., 0.0, 0.0), (1.0,0.0,0.0), 'cut', 0.0)
mpm_solver.add_surface_collider((0.5+box_length/2., 0.0, 0.0), (-1.0,0.0,0.0), 'cut', 0.0)
mpm_solver.add_surface_collider((0.0, 0.5+box_length/2., 0.0), (0.0,-1.0,0.0), 'cut', 0.0)
mpm_solver.add_surface_collider((0.0, 0.5-box_length/2., 0.0), (0.0,1.0,0.0), 'cut', 0.0)

#directory_to_save = './sim_results/jelly_mat_only_small_g_height_z'
directory_to_save = args.out_path

save_data_at_frame(mpm_solver, directory_to_save, 0, save_to_ply=True, save_to_h5=True)

num_frames = args.time_steps
#for k in range(1,50):
for k in range(1, int(num_frames)):
    mpm_solver.p2g2p(k, 0.002, device=dvc)
    save_data_at_frame(mpm_solver, directory_to_save, k, save_to_ply=True, save_to_h5=True)

# You can either load sampling data from an external h5 file, containing initial position (n,3) and particle_volume (n,)
#mpm_solver = MPM_Simulator_WARP(10)
#mpm_solver.load_from_sampling(f"{directory_to_save}/sim_0000000099.h5", n_grid = 150, device=dvc)

# Try reducing gravity
material_params2 = {
    #'bulk_modulus': 2000.0,
    "material": "jelly",
    "E": 1e5,
    "nu": 0.3,
    #'friction_angle': 35,
    'g': [0.0, 0.0, 0.0],
    "density": 1000.0
}
#mpm_solver.set_parameters_dict(material_params2)

# add bounding box for fluid
#box_length = 0.4
#mpm_solver.add_surface_collider((0.0, 0.0, 0.13), (0.0,0.0,1.0), 'cut', 0.0)
#mpm_solver.add_surface_collider((0.5-box_length/2., 0.0, 0.0), (1.0,0.0,0.0), 'cut', 0.0)
#mpm_solver.add_surface_collider((0.5+box_length/2., 0.0, 0.0), (-1.0,0.0,0.0), 'cut', 0.0)
#mpm_solver.add_surface_collider((0.0, 0.5+box_length/2., 0.0), (0.0,-1.0,0.0), 'cut', 0.0)
#mpm_solver.add_surface_collider((0.0, 0.5-box_length/2., 0.0), (0.0,1.0,0.0), 'cut', 0.0)

#save_data_at_frame(mpm_solver, directory_to_save, 100, save_to_ply=True, save_to_h5=True)

#for i in range(int(num_frames/2), num_frames):
#    mpm_solver.p2g2p(i, 0.002, device=dvc)
#    save_data_at_frame(mpm_solver, directory_to_save, i, save_to_ply=True, save_to_h5=True)

# extract the position, make some changes, load it back
#position = mpm_solver.export_particle_x_to_torch()
# e.g. we shift the x position
#position[:,0] = position[:,0] + 0.1
#mpm_solver.import_particle_x_from_torch(position)
# keep running sim
#for k in range(50,100):
 
#    mpm_solver.p2g2p(k, 0.002, device=dvc)
#    save_data_at_frame(mpm_solver, directory_to_save, k, save_to_ply=True, save_to_h5=True)
