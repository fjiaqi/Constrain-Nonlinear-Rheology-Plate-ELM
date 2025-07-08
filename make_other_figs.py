# %%
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from scipy.stats import trim_mean
from scipy.interpolate import LinearNDInterpolator
import pandas as pd
import h5py
import os
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.append('./model/')
sys.path.append('./utils/')
from references import *
from domain import *
from structures import *
from diag_utils import *

params = {
    'image.interpolation': 'nearest',
    'image.cmap': 'RdYlBu',
    'savefig.dpi': 300,  # to adjust notebook inline plot size
    'axes.labelsize': 16, # fontsize for x and y labels (was 10)
    'axes.titlesize': 16,
    'font.size': 16,
    'legend.fontsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'text.usetex': False
}
mpl.rcParams.update(params)

df = pd.read_csv("./data/Maule_rates_processed.csv")
df_st = df[df['Y'].ge(350) & df['Y'].le(1000) & df['X'].le(350)]

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

def load_ttemp(output_dir, yres, arg_mesh):
    with h5py.File(f'{output_dir}/data/tracer_temperature_0.h5', 'r') as h5:
        mesh_temp = h5['data'][...][arg_mesh]

    return T_C(mesh_temp).reshape((129, yres+1, 1025, 1))

def load_tmod(output_dir, yres, arg_mesh):
    with h5py.File(f'{output_dir}/data/tracer_modulus_0.h5', 'r') as h5:
        mesh_mod = h5['data'][...][arg_mesh]

    return tau_MPa(mesh_mod).reshape((129, yres+1, 1025, 1))
    
def load_sz_args(output_dir, step, flag):
    with h5py.File(f'{output_dir}/data/tracer_swarm_0.h5', 'r') as h5:
        mesh_loc = h5['data'][...]

    with h5py.File(f'{output_dir}/data/tracer_material_0.h5', 'r') as h5:
        mesh_mat = h5['data'][...]
        arg_sz = np.where(mesh_mat == 3)[0]

    with h5py.File(f'{output_dir}/data/tracer_velocity_{flag}{step}.h5', 'r') as h5:
        mesh_vel = h5['data'][...]

    with h5py.File(f'{output_dir}/data/tracer_strainrate_{flag}{step}.h5', 'r') as h5:
        mesh_strainrate = h5['data'][...]

    with h5py.File(f'{output_dir}/data/tracer_stress_{flag}{step}.h5', 'r') as h5:
        mesh_stress = h5['data'][...]

    return l_km(mesh_loc[arg_sz]), v_mm_yr(mesh_vel[arg_sz]), logeps_s(mesh_strainrate[arg_sz]), tau_MPa(mesh_stress[arg_sz])
    
# %%
# Fig. 2b
sz_fric_zbins = -np.arange(0, 101, 2.5)[::-1]
sz_fric_ybins = np.linspace(0, l_km(domain_length), yres+1)

sz_yfactor = np.exp(-0.5*(sz_fric_ybins[:-1]/200)**2)
sz_fric_edge = 0.55*np.ones(sz_fric_zbins[1:].shape)
sz_fric_center = np.zeros(sz_fric_zbins[1:].shape)

npz = np.load('/home/x-jqfang/scratch/model_2504/2d_A1_V10.6_Gw80_Ge40/vars/vars_c0.npz')
npz_loc = l_km(npz['loc'])
npz_vars = npz['vars']
npz_mat = npz_vars[:, 0]
npz_fric = npz_vars[:, 8]

npz_arg_sz = np.where(npz_mat == 3)[0]
for j in range(len(sz_fric_zbins)-1):
    npz_arg_bin = np.where((npz_loc[npz_arg_sz, -1] >= sz_fric_zbins[j]) & (npz_loc[npz_arg_sz, -1] <= sz_fric_zbins[j+1]))[0]
    sz_fric_center[j] = np.mean(npz_fric[npz_arg_sz][npz_arg_bin])
sz_fric = sz_yfactor[:, None]*sz_fric_center + (1-sz_yfactor[:, None])*sz_fric_edge

fig = plt.figure(figsize=(8, 2.5))

ax = fig.add_subplot(111)
sz_fric_grid = np.meshgrid(sz_fric_ybins[:-1], sz_fric_zbins[1:])

colors = [(1, 0, 0), (1, 1, 1), (0, 0, 1)]
values = [-0.35, 0, 0.05]
norm_vals = [(v - values[0]) / (values[-1] - values[0]) for v in values]
custom_cmap = mpl.colors.LinearSegmentedColormap.from_list("custom_RdBu", list(zip(norm_vals, colors)))

cax = ax.contourf(sz_fric_grid[0], sz_fric_grid[1], sz_fric.T-0.55, levels=101, cmap=custom_cmap)
ax.set_xlabel('$y$ (km)')
ax.set_ylabel('$z$ (km)')
ax.set_xlim(0, 2000)
ax.set_ylim(-80, 0)
ax.set_xticks([0, 500, 1000, 1500, 2000])
ax.minorticks_on()

cbar_ax = fig.add_axes([0.65, 0.75, 0.2, 0.03])
cbar = fig.colorbar(cax, cax=cbar_ax, orientation='horizontal')
cbar.ax.text(1.1, 0.5, r'$\Delta \mu$', transform=cbar.ax.transAxes,
             va='center', ha='left')
cbar.set_ticks([-0.3, 0])

plt.tight_layout()
plt.savefig('./fig2b.png')



# %%
# Fig. S1
output_dir = '/home/x-jqfang/scratch/model_2504/3d_A1_V10.6_Gw80_Ge40'
yres = 256
mesh_loc, arg_mesh = load_tmesh(output_dir, yres)

mesh_temp = load_ttemp(output_dir, yres, arg_mesh)

fig, ax = plt.subplots(figsize=(12, 3.8))

core_left, core_right = -200, 700
core_bottom, core_top = -300, 0
core_buffer = 10
arg_x = np.where((mesh_loc[0, 0, :, 0] >= core_left-core_buffer) &
                 (mesh_loc[0, 0, :, 0] <= core_right+core_buffer))[0]
arg_z = np.where((mesh_loc[:, 0, 0, -1] >= core_bottom-core_buffer) &
                 (mesh_loc[:, 0, 0, -1] <= core_top+core_buffer))[0]

mappable = ax.contourf(mesh_loc[-1, 0, arg_x, 0], mesh_loc[arg_z, 0, 0, 2], mesh_temp[:, 0, arg_x, 0][arg_z], 
                cmap='coolwarm', levels=np.linspace(0, 1640, 101))
cbar = plt.colorbar(mappable, ax=ax, label=r'$T$ ($^\circ$C)')
cbar.set_ticks(np.arange(0, 1601, 400))
cs = ax.contour(mesh_loc[-1, 0, arg_x, 0], mesh_loc[arg_z, 0, 0, 2], mesh_temp[:, 0, arg_x, 0][arg_z], 
                colors='k', levels=np.arange(100, 1301, 200))
ax.clabel(cs, inline=True, fontsize=10, fmt='%d', inline_spacing=-2)

ax.set_aspect('equal')
ax.set_xlabel('$x$ (km)')
ax.set_ylabel('$z$ (km)')
ax.set_xlim(core_left, core_right)
ax.set_ylim(core_bottom, core_top)
ax.minorticks_on()

plt.tight_layout()
plt.savefig('./figS1.png')

# %%
# Fig. S2
mesh_mod = load_tmod(output_dir, yres, arg_mesh)/1e3

fig, ax = plt.subplots(figsize=(11, 4))

mappable = ax.contourf(mesh_loc[-1, 0, :, 0], mesh_loc[:, 0, 0, 2], mesh_mod[:, 0, :, 0], 
                cmap='RdYlBu', levels=101)
cbar = plt.colorbar(mappable, ax=ax, label=r'$G$ (GPa)')
cbar.set_ticks(np.arange(20, 121, 20))

ax.set_aspect('equal')
ax.set_xlabel('$x$ (km)')
ax.set_ylabel('$z$ (km)')
ax.set_xlim(-1000, 1000)
ax.minorticks_on()

plt.tight_layout()
plt.savefig('./figS2.png')

# %%
# Fig. S3
flag = 'i'
step = 10
sz_loc, sz_vel_i, sz_sr_i, sz_st_i = load_sz_args(output_dir, step, flag)

sz_zbins = -np.arange(0, 101, 5)[::-1]
sz_ybins = np.linspace(0, np.max(sz_loc[:, 1]), 81)
sz_dv_i = np.zeros((len(sz_ybins)-1, len(sz_zbins)-1, 3))
sz_meansr_i = np.zeros((len(sz_ybins)-1, len(sz_zbins)-1))
sz_meanst_i = np.zeros((len(sz_ybins)-1, len(sz_zbins)-1))

for i in tqdm(range(len(sz_ybins)-1)):
    for j in range(len(sz_zbins)-1):
        arg_bin = np.where((sz_loc[:, 1] >= sz_ybins[i]) & (sz_loc[:, 1] <= sz_ybins[i+1]) & 
                          (sz_loc[:, -1] >= sz_zbins[j]) & (sz_loc[:, -1] <= sz_zbins[j+1]))[0]
        sz_dv_i[i, j] = np.percentile(sz_vel_i[arg_bin], 95, axis=0) - np.percentile(sz_vel_i[arg_bin], 5, axis=0)
        sz_meansr_i[i, j] = np.log10(np.mean(10**sz_sr_i[arg_bin]))
        sz_meanst_i[i, j] = np.mean(sz_st_i[arg_bin, 0])

flag = 'c'
step = 0
sz_loc, sz_vel_c, sz_sr_c, sz_st_c = load_sz_args(output_dir, step, flag)

sz_dv_c = np.zeros((len(sz_ybins)-1, len(sz_zbins)-1, 3))
sz_meansr_c = np.zeros((len(sz_ybins)-1, len(sz_zbins)-1))
sz_meanst_c = np.zeros((len(sz_ybins)-1, len(sz_zbins)-1))

for i in tqdm(range(len(sz_ybins)-1)):
    for j in range(len(sz_zbins)-1):
        arg_bin = np.where((sz_loc[:, 1] >= sz_ybins[i]) & (sz_loc[:, 1] <= sz_ybins[i+1]) & 
                          (sz_loc[:, -1] >= sz_zbins[j]) & (sz_loc[:, -1] <= sz_zbins[j+1]))[0]
        sz_dv_c[i, j] = np.percentile(sz_vel_c[arg_bin], 95, axis=0) - np.percentile(sz_vel_c[arg_bin], 5, axis=0)
        sz_meansr_c[i, j] = np.log10(np.mean(10**sz_sr_c[arg_bin]))
        sz_meanst_c[i, j] = np.mean(sz_st_c[arg_bin, 0])

fig = plt.figure(figsize=(8, 2.5))

ax = fig.add_subplot(111)
sz_grid = np.meshgrid(sz_ybins[:-1], sz_zbins[1:])
cax = ax.contourf(sz_grid[0], sz_grid[1], (sz_meanst_c-sz_meanst_i).T, 
                  levels=np.linspace(-8, 8, 101), cmap='RdBu', vmin=-8, vmax=8)
ax.set_xlabel('$y$ (km)')
ax.set_ylabel('$z$ (km)')
ax.set_xlim(0, 2000)
ax.set_ylim(-80, 0)
ax.set_xticks([0, 500, 1000, 1500, 2000])
ax.minorticks_on()

cbar_ax = fig.add_axes([0.58, 0.75, 0.2, 0.03])
cbar = fig.colorbar(cax, cax=cbar_ax, orientation='horizontal')
cbar.ax.text(1.1, 0.5, r'$\Delta \tau$ (MPa)', transform=cbar.ax.transAxes,
             va='center', ha='left')
cbar.set_ticks([-5, 0, 5])

plt.tight_layout()
plt.savefig('./figS3.png')

# %%
# Fig. S8
flag = 'p'
step = 0
sz_loc, sz_vel_p, sz_sr_p, sz_st_p = load_sz_args(output_dir, step, flag)

sz_dv_p = np.zeros((len(sz_ybins)-1, len(sz_zbins)-1, 3))
sz_meansr_p = np.zeros((len(sz_ybins)-1, len(sz_zbins)-1))
sz_meanst_p = np.zeros((len(sz_ybins)-1, len(sz_zbins)-1))

for i in tqdm(range(len(sz_ybins)-1)):
    for j in range(len(sz_zbins)-1):
        arg_bin = np.where((sz_loc[:, 1] >= sz_ybins[i]) & (sz_loc[:, 1] <= sz_ybins[i+1]) & 
                          (sz_loc[:, -1] >= sz_zbins[j]) & (sz_loc[:, -1] <= sz_zbins[j+1]))[0]
        sz_dv_p[i, j] = np.percentile(sz_vel_p[arg_bin], 95, axis=0) - np.percentile(sz_vel_p[arg_bin], 5, axis=0)
        sz_meansr_p[i, j] = np.log10(np.mean(10**sz_sr_p[arg_bin]))
        sz_meanst_p[i, j] = np.mean(sz_st_p[arg_bin, 0])

output_dir_w = '/home/x-jqfang/scratch/model_2504/3d_A1_V10.6_Gw80_Ge40_r0.0125'
flag = 'p'
step = 0
sz_loc, sz_vel_p_w, sz_sr_p_w, sz_st_p_w = load_sz_args(output_dir_w, step, flag)

sz_dv_p_w = np.zeros((len(sz_ybins)-1, len(sz_zbins)-1, 3))
sz_meansr_p_w = np.zeros((len(sz_ybins)-1, len(sz_zbins)-1))
sz_meanst_p_w = np.zeros((len(sz_ybins)-1, len(sz_zbins)-1))

for i in tqdm(range(len(sz_ybins)-1)):
    for j in range(len(sz_zbins)-1):
        arg_bin = np.where((sz_loc[:, 1] >= sz_ybins[i]) & (sz_loc[:, 1] <= sz_ybins[i+1]) & 
                          (sz_loc[:, -1] >= sz_zbins[j]) & (sz_loc[:, -1] <= sz_zbins[j+1]))[0]
        sz_dv_p_w[i, j] = np.percentile(sz_vel_p_w[arg_bin], 95, axis=0) - np.percentile(sz_vel_p_w[arg_bin], 5, axis=0)
        sz_meansr_p_w[i, j] = np.log10(np.mean(10**sz_sr_p_w[arg_bin]))
        sz_meanst_p_w[i, j] = np.mean(sz_st_p_w[arg_bin, 0])

fig = plt.figure(figsize=(15, 6))

gs = mpl.gridspec.GridSpec(1, 3, width_ratios=[1, 2, 2])
axes = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])]

ax = axes[0]
k = 0
ax.plot(np.linalg.norm(sz_dv_p_w, axis=2)[k], sz_zbins[1:], lw=3, c='r', label=0.0125)
ax.plot(np.linalg.norm(sz_dv_p, axis=2)[k], sz_zbins[1:], lw=3, c='b', label=0.4)

ax.set_xlabel('$V$ (mm/yr)')
ax.set_ylabel('$z$ (km)')
ax.set_xlim(0, 700)
ax.set_ylim(-95, 0)
ax.minorticks_on()
ax.legend(title=r'$\dot{\mu}$ (1/year)', loc='lower right', fontsize=12,
          title_fontsize=12)
ax.set_title('(a)', loc='left')

ax = axes[1]
mappable = ax.contourf(sz_ybins[:-1], sz_zbins[1:], np.linalg.norm(sz_dv_p, axis=2).T, 
                       cmap='RdYlBu_r', levels=np.linspace(0, 700, 201))
ax.set_yticks([])
ax.set_xlabel('$y$ (km)')
ax.minorticks_on()
ax.text(1900, -92, r'$\dot{\mu}$ = 0.4/year', fontsize=12,
            ha='right', va='bottom', c='k',
            bbox=dict(facecolor='w', edgecolor='none', alpha=0.8, boxstyle='round'))
ax.set_title('(b)', loc='left')

ax = axes[2]
label_list = ['$V$ (mm/yr)', '$\\dot{\\varepsilon}$ (1/s)', '$\\tau$ (MPa)']
cmap_list = ['RdYlBu_r', 'RdYlBu_r', 'RdYlBu']

mappable = ax.contourf(sz_ybins[:-1], sz_zbins[1:], np.linalg.norm(sz_dv_p_w, axis=2).T, 
                       cmap='RdYlBu_r', levels=np.linspace(0, 700, 201))
ax.set_yticks([])
ax.set_xlabel('$y$ (km)')
ax.minorticks_on()
ax.text(1900, -92, r'$\dot{\mu}$ = 0.0125/year', fontsize=12,
            ha='right', va='bottom', c='k',
            bbox=dict(facecolor='w', edgecolor='none', alpha=0.8, boxstyle='round'))
ax.set_title('(c)', loc='left')

plt.tight_layout()
plt.savefig('./figS8.png')

# %%
# Fig. S6
fig, ax = plt.subplots(figsize=(6, 6))

ax.plot([2, 1, 0.5], [7.8, 10.6, 13.4], c='C0', lw=3, marker='o', label='Plate motion')
ax.plot([1.55, 1, 0.63], [7.8, 10.6, 13.4], c='C3', lw=3, ls='--', marker='D', label='ELM')

ax.set_xscale('log')
ax.legend()
ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:.1f}'))
ax.set_xticks([0.5, 1, 2])
ax.minorticks_on()
ax.set_xlabel(r'$\omega$')
ax.set_ylabel(r'$V$ (cm$^3$/mol)')

plt.tight_layout()
plt.savefig('./figS6.png')

# %%
# Fig. S7
flag = 'i'
step = 10
mesh_vel_i = load_tvel(output_dir, step, yres, arg_mesh, flag)

flag = 'p'
step = 0
mesh_vel_p = load_tvel(output_dir, step, yres, arg_mesh, flag)
for i in range(1, 5):
    mesh_vel_p += load_tvel(output_dir, step+i, yres, arg_mesh, flag)
mesh_vel_p /= 5

fig, ax = plt.subplots(figsize=(11, 3))

j_all = np.arange(64, 90)
mappable = ax.contourf(mesh_loc[-1, 0, :, 0], mesh_loc[:, 0, 0, 2], np.mean((mesh_vel_p-mesh_vel_i)[:, j_all, :, 0], axis=0), 
                cmap='RdBu', levels=np.linspace(-5, 5, 101), extend='both')
cbar = plt.colorbar(mappable, ax=ax, label=r'$\Delta v_x$ (mm/yr)')
cbar.set_ticks(np.arange(-5, 5.1, 2.5))

ax.plot(slab_top_coords[:, 0]*1e3, slab_top_coords[:, 1]*1e3, color='k', ls='--')

ax.set_aspect('equal')
ax.set_xlabel('$x$ (km)')
ax.set_ylabel('$z$ (km)')
ax.set_xlim(-300, 600)
ax.set_ylim(-300, 0)
plt.tight_layout()
plt.savefig('./figS7.png')

# %%
