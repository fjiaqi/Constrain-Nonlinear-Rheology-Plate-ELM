import numpy as np
from scipy.interpolate import LinearNDInterpolator
import pandas as pd
import h5py
import os
from pathlib import Path
import logging
import sys
sys.path.append('./model/')
sys.path.append('./utils/')
from references import *
from domain import *
from structures import *
from materials import *
from diag_utils import *

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s  (%(relativeCreated)d ms spent)\n%(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
logger.debug(f'Started extracting key results...')

def load_tmesh(output_dir, yres):
    with h5py.File(f'{output_dir}/data/tracer_swarm_0.h5', 'r') as h5:
        mesh_loc = h5['data'][...]
        arg_mesh = np.lexsort((mesh_loc[:, 0], mesh_loc[:, 1], mesh_loc[:, 2]))
        mesh_loc = mesh_loc[arg_mesh]

    return l_km(mesh_loc).reshape((129, yres+1, 1025, 3)), arg_mesh

def load_tvel(output_dir, step, yres, arg_mesh, flag='l'):
    with h5py.File(f'{output_dir}/data/tracer_velocity_{flag}{step}.h5', 'r') as h5:
        mesh_vel = h5['data'][...][arg_mesh]

    return v_mm_yr(mesh_vel).reshape((129, yres+1, 1025, 3))

def load_tvisc(output_dir, step, yres, arg_mesh, flag='l'):
    with h5py.File(f'{output_dir}/data/tracer_viscosity_{flag}{step}.h5', 'r') as h5:
        mesh_visc = h5['data'][...][arg_mesh]

    return logeta_Pa_s(mesh_visc).reshape((129, yres+1, 1025, 1))


# Directory to process, number os steps, and y-resolution
output_dir = os.path.normpath(sys.argv[1])
nsteps = 5
yres = 256

# Mesh and long-term
flag = 'l'
step = 10
mesh_loc, arg_mesh = load_tmesh(output_dir, yres)
mesh_vel_l = load_tvel(output_dir, step, yres, arg_mesh, flag)

# Pre-seismic
flag = 'i'
step = 10
mesh_vel_i = load_tvel(output_dir, step, yres, arg_mesh, flag)
mesh_visc_i = load_tvisc(output_dir, step, yres, arg_mesh, flag)

# Co-seismic, uncomment the next line for restarted cases
# output_dir = f'{output_dir}_f0.3'
# output_dir = f'{output_dir}_dt0.1'
flag = 'c'
step = 0
mesh_vel_c = load_tvel(output_dir, step, yres, arg_mesh, flag)
mesh_visc_c = load_tvisc(output_dir, step, yres, arg_mesh, flag)

# Post-seismic, uncomment the next line for restarted cases
# output_dir = f'{output_dir}_r0.0125'
flag = 'p'
step = 0
mesh_vel_p = load_tvel(output_dir, step, yres, arg_mesh, flag)
mesh_visc_p = load_tvisc(output_dir, step, yres, arg_mesh, flag)
for i in range(1, nsteps):
    mesh_vel_p += load_tvel(output_dir, i, yres, arg_mesh, flag)
mesh_vel_p /= nsteps

logger.debug(f'Data loaded')

res_dir = f'./res/{os.path.basename(output_dir)}'
Path(res_dir).mkdir(parents=True, exist_ok=True)



# Surface velocity (Fig. 2d)
surf_x = mesh_loc[-1, 0, :, 0]
surf_vx_l = np.mean(mesh_vel_l[-1, :, :, 0], axis=0)
surf_vx_i = np.mean(mesh_vel_i[-1, :, :, 0], axis=0)
surf_ux_c = mesh_vel_c[-1, 0, :, 0]/1e3



# Viscosity profile (Fig. 3a)
i_all = np.where(np.abs(mesh_loc[-1, 0, :, 0] - -2250) <= 250)[0]
j = 0
visc_i_a = np.mean(mesh_visc_i[:, j, i_all, 0], axis=-1)
visc_c_a = np.mean(mesh_visc_c[:, j, i_all, 0], axis=-1)
visc_p_a = np.mean(mesh_visc_p[:, j, i_all, 0], axis=-1)
visc_z = mesh_loc[:, 0, 0, 2]

# Viscosity profile (Fig. 2c, 3b)
i_all = np.where(np.abs(mesh_loc[-1, 0, :, 0] - -250) <= 250)[0]
j = 0

visc_i_b = np.mean(mesh_visc_i[:, j, i_all, 0], axis=-1)
visc_c_b = np.mean(mesh_visc_c[:, j, i_all, 0], axis=-1)
visc_p_b = np.mean(mesh_visc_p[:, j, i_all, 0], axis=-1)

# Viscosity profile (Fig. 3d)
i_all = np.where(np.abs(mesh_loc[-1, 0, :, 0] - -250) <= 250)[0]
j = 64

visc_i_c = np.mean(mesh_visc_i[:, j, i_all, 0], axis=-1)
visc_c_c = np.mean(mesh_visc_c[:, j, i_all, 0], axis=-1)
visc_p_c = np.mean(mesh_visc_p[:, j, i_all, 0], axis=-1)



# Surface velocity change (Fig. 4)
dvel_p_func = LinearNDInterpolator((mesh_loc[-1, :, :, 0].flatten(), mesh_loc[-1, :, :, 1].flatten()), (mesh_vel_p-mesh_vel_i)[-1, :, :, :].reshape(-1, 3))

intv = 50
pts = np.meshgrid(np.arange(-250, 501, intv), np.arange(250, 401, intv))
dvel_near_x = pts[0].flatten()
dvel_near_y = pts[1].flatten()
dvel_near = dvel_p_func(dvel_near_x, dvel_near_y)


pts = np.meshgrid(np.arange(-250, 501, intv), np.arange(400+intv, 1001, intv))
dvel_far_x = pts[0].flatten()
dvel_far_y = pts[1].flatten()
dvel_far = dvel_p_func(dvel_far_x, dvel_far_y)



# Surface velocity change profile (Fig. 5)
x_all = np.arange(100, 201, 20)

n = 101
dvel_line_y = np.linspace(0, 1000, n)
dvel_line = np.zeros(n)

for i in range(len(x_all)):
    ptx = np.repeat(x_all[i], n)
    dvel_line += dvel_p_func(ptx, dvel_line_y)[:, 0]

dvel_line /= len(x_all)



# Save results
np.savez(f'{res_dir}/key_res.npz', 
         surf_x=surf_x, surf_vx_l=surf_vx_l, surf_vx_i=surf_vx_i, surf_ux_c=surf_ux_c,
         visc_z=visc_z, visc_i_a=visc_i_a, visc_c_a=visc_c_a, visc_p_a=visc_p_a,
         visc_i_b=visc_i_b, visc_c_b=visc_c_b, visc_p_b=visc_p_b,
         visc_i_c=visc_i_c, visc_c_c=visc_c_c, visc_p_c=visc_p_c,
         dvel_near=dvel_near, dvel_near_x=dvel_near_x, dvel_near_y=dvel_near_y,
         dvel_far=dvel_far, dvel_far_x=dvel_far_x, dvel_far_y=dvel_far_y,
         dvel_line=dvel_line, dvel_line_y=dvel_line_y)

logger.debug(f'Results saved to {res_dir}/key_res.npz')