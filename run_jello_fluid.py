
import warp as wp
from mpm_solver_warp import MPM_Simulator_WARP
from engine_utils import *
import torch
wp.init()
wp.config.verify_cuda = True


dvc = "cuda:0"

mpm_solver = MPM_Simulator_WARP(10) # initialize with whatever number is fine. it will be reintialized


# You can either load sampling data from an external h5 file, containing initial position (n,3) and particle_volume (n,)
mpm_solver.load_from_sampling("sand_column.h5", n_grid = 150, device=dvc) 

# Or load from torch tensor (also position and volume)
# Here we borrow the data from h5, but you can use your own
volume_tensor = torch.ones(mpm_solver.n_particles) * 2.5e-8
position_tensor = mpm_solver.export_particle_x_to_torch()

mpm_solver.load_initial_data_from_torch(position_tensor, volume_tensor)

# Note: You must provide 'density=..' to set particle_mass = density * particle_volume

material_params = {
    'bulk_modulus': 2000.0,
    "material": "jelly",
    'friction_angle': 35,
    'g': [0.0, 0.0, -10.0],
    "density": 1000.0
}
mpm_solver.set_parameters_dict(material_params)

# add bounding box for fluid
box_length = 0.4
mpm_solver.add_surface_collider((0.0, 0.0, 0.13), (0.0,0.0,1.0), 'cut', 0.0)
mpm_solver.add_surface_collider((0.5-box_length/2., 0.0, 0.0), (1.0,0.0,0.0), 'cut', 0.0)
mpm_solver.add_surface_collider((0.5+box_length/2., 0.0, 0.0), (-1.0,0.0,0.0), 'cut', 0.0)
mpm_solver.add_surface_collider((0.0, 0.5+box_length/2., 0.0), (0.0,-1.0,0.0), 'cut', 0.0)
mpm_solver.add_surface_collider((0.0, 0.5-box_length/2., 0.0), (0.0,1.0,0.0), 'cut', 0.0)

directory_to_save = './sim_results/jelly_fluid_g_neg_neutral'

save_data_at_frame(mpm_solver, directory_to_save, 0, save_to_ply=True, save_to_h5=True)

#for k in range(1,50):
for k in range(1, 100):
    mpm_solver.p2g2p(k, 0.002, device=dvc)
    save_data_at_frame(mpm_solver, directory_to_save, k, save_to_ply=True, save_to_h5=True)

# You can either load sampling data from an external h5 file, containing initial position (n,3) and particle_volume (n,)
mpm_solver = MPM_Simulator_WARP(10)
mpm_solver.load_from_sampling(f"{directory_to_save}/sim_0000000099.h5", n_grid = 150, device=dvc)

# Try reducing gravity
material_params2 = {
    'bulk_modulus': 2000.0,
    "material": "jelly",
    'friction_angle': 35,
    'g': [0.0, 0.0, 0.0],
    "density": 1000.0
}
mpm_solver.set_parameters_dict(material_params2)

# add bounding box for fluid
box_length = 0.4
mpm_solver.add_surface_collider((0.0, 0.0, 0.13), (0.0,0.0,1.0), 'cut', 0.0)
mpm_solver.add_surface_collider((0.5-box_length/2., 0.0, 0.0), (1.0,0.0,0.0), 'cut', 0.0)
mpm_solver.add_surface_collider((0.5+box_length/2., 0.0, 0.0), (-1.0,0.0,0.0), 'cut', 0.0)
mpm_solver.add_surface_collider((0.0, 0.5+box_length/2., 0.0), (0.0,-1.0,0.0), 'cut', 0.0)
mpm_solver.add_surface_collider((0.0, 0.5-box_length/2., 0.0), (0.0,1.0,0.0), 'cut', 0.0)

save_data_at_frame(mpm_solver, directory_to_save, 100, save_to_ply=True, save_to_h5=True)

for i in range(1, 100):
    mpm_solver.p2g2p(100+i, 0.002, device=dvc)
    save_data_at_frame(mpm_solver, directory_to_save, i+100, save_to_ply=True, save_to_h5=True)

# extract the position, make some changes, load it back
#position = mpm_solver.export_particle_x_to_torch()
# e.g. we shift the x position
#position[:,0] = position[:,0] + 0.1
#mpm_solver.import_particle_x_from_torch(position)
# keep running sim
#for k in range(50,100):
 
#    mpm_solver.p2g2p(k, 0.002, device=dvc)
#    save_data_at_frame(mpm_solver, directory_to_save, k, save_to_ply=True, save_to_h5=True)
