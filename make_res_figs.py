# %%
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import os
from pathlib import Path
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
    'savefig.dpi': 300,
    'axes.labelsize': 16,
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

def load_res_npz(output_dir, key):
    with np.load(f'{output_dir}/key_res.npz') as npz:
        return npz[key]

# %%
# Fig. 2c
dir_list = ['3d_A2_V7.8_Gw80_Ge40',
            '3d_A1_V10.6_Gw80_Ge40',
            '3d_A0.5_V13.4_Gw80_Ge40',
            '3d_A0.25_V16.1_Gw80_Ge40']
w_list = [2, 1, 0.5, 0.25]
colors = plt.cm.RdYlBu(np.linspace(0, 1, len(w_list))[::-1])

fig = plt.figure(figsize=(5, 8))

ax = fig.add_subplot(111)

for i in range(len(dir_list)):
    visc_i_b = load_res_npz(f'./res/{dir_list[i]}', 'visc_i_b')
    visc_p_b = load_res_npz(f'./res/{dir_list[i]}', 'visc_p_b')
    visc_z = load_res_npz(f'./res/{dir_list[i]}', 'visc_z')
    ax.plot(np.power(10, visc_i_b), visc_z, c=colors[i], lw=3, label=f'${w_list[i]}$')

ax.set_xlim(1e17, 1e24)
ax.set_ylim(-800, 0)
ax.set_xscale('log')
ax.set_ylabel(r'$z$ (km)')
ax.set_xlabel(r'$\eta$ (Pa$\cdot$s)')
ax.minorticks_on()
ax.legend(title=r'$\omega$', loc='lower left')

plt.tight_layout()
plt.savefig('./fig2c.png')

# %%
# Fig. 2d
i = 1

surf_x = load_res_npz(f'./res/{dir_list[i]}', 'surf_x')
surf_vx_l = load_res_npz(f'./res/{dir_list[i]}', 'surf_vx_l')
surf_vx_i = load_res_npz(f'./res/{dir_list[i]}', 'surf_vx_i')
surf_ux_c = load_res_npz(f'./res/{dir_list[i]}', 'surf_ux_c')

fig = plt.figure(figsize=(12, 3))

ax = fig.add_subplot(111)

ax.plot(surf_x, surf_vx_l, 
        'k--', lw=3, label='Long-term')
ax.plot(surf_x, surf_vx_i, 
        'b', lw=3, label='Pre-seismic')
ax.plot([], [], 'r', lw=3, label='Co-seismic')

ax.set_xlim(surf_x.min(), surf_x.max())
ax.set_xlabel(r'$x$ (km)')
ax.set_ylabel(r'$v_x$ (mm/yr)', c='b')
ax.minorticks_on()
ax.legend(loc='lower left')

axt = ax.twinx()
axt.plot(surf_x, surf_ux_c, 'r', lw=3)
axt.set_ylabel(r'$u_x$ (m)', c='r')
axt.minorticks_on()

plt.tight_layout()
plt.savefig('./fig2d.png')

# %%
# Fig. 3
i = 1

fig, axes = plt.subplots(1, 3, figsize=(15, 8))

ax = axes[0]
visc_i_a = load_res_npz(f'./res/{dir_list[i]}', 'visc_i_a')
visc_c_a = load_res_npz(f'./res/{dir_list[i]}', 'visc_c_a')
visc_p_a = load_res_npz(f'./res/{dir_list[i]}', 'visc_p_a')
visc_z = load_res_npz(f'./res/{dir_list[i]}', 'visc_z')
ax.plot(np.power(10, visc_i_a), visc_z, c='b', ls='-', lw=3, label=f'Pre-seismic')
ax.plot(np.power(10, visc_c_a), visc_z, c='r', ls=':', lw=3, label=f'Co-seismic')
ax.plot(np.power(10, visc_p_a), visc_z, c='m', ls='--', lw=3, label=f'Post-seismic')

ax.set_xlim(1e17, 1e24)
ax.set_ylim(-800, 0)
ax.set_xscale('log')
ax.set_ylabel(r'$z$ (km)')
ax.set_xlabel(r'$\eta$ (Pa$\cdot$s)')
ax.minorticks_on()
ax.set_title(r'(a) $x$ = [-2500, -2000] km, $y$ = 0 km', loc='left')
ax.legend(loc='lower left')

ax = axes[1]
visc_i_b = load_res_npz(f'./res/{dir_list[i]}', 'visc_i_b')
visc_c_b = load_res_npz(f'./res/{dir_list[i]}', 'visc_c_b')
visc_p_b = load_res_npz(f'./res/{dir_list[i]}', 'visc_p_b')
visc_z = load_res_npz(f'./res/{dir_list[i]}', 'visc_z')
ax.plot(np.power(10, visc_i_b), visc_z, c='b', ls='-', lw=3, label=f'Pre-seismic')
ax.plot(np.power(10, visc_c_b), visc_z, c='r', ls=':', lw=3, label=f'Co-seismic')
ax.plot(np.power(10, visc_p_b), visc_z, c='m', ls='--', lw=3, label=f'Post-seismic')

ax.set_yticks([])
ax.set_xlim(1e17, 1e24)
ax.set_ylim(-800, 0)
ax.set_xscale('log')
ax.set_xlabel(r'$\eta$ (Pa$\cdot$s)')
ax.minorticks_on()
ax.set_title(r'$x$ = [-500, 0] km, $y$ = 0 km')
ax.set_title('(b)', loc='left')

ax = axes[2]
visc_i_c = load_res_npz(f'./res/{dir_list[i]}', 'visc_i_c')
visc_c_c = load_res_npz(f'./res/{dir_list[i]}', 'visc_c_c')
visc_p_c = load_res_npz(f'./res/{dir_list[i]}', 'visc_p_c')
visc_z = load_res_npz(f'./res/{dir_list[i]}', 'visc_z')
ax.plot(np.power(10, visc_i_c), visc_z, c='b', ls='-', lw=3, label=f'Pre-seismic')
ax.plot(np.power(10, visc_c_c), visc_z, c='r', ls=':', lw=3, label=f'Co-seismic')
ax.plot(np.power(10, visc_p_c), visc_z, c='m', ls='--', lw=3, label=f'Post-seismic')

ax.set_yticks([])
ax.set_xlim(1e17, 1e24)
ax.set_ylim(-800, 0)
ax.set_xscale('log')
ax.set_xlabel(r'$\eta$ (Pa$\cdot$s)')
ax.minorticks_on()
ax.set_title(r'$x$ = [-500, 0] km, $y$ = 500 km')
ax.set_title('(c)', loc='left')

plt.tight_layout()
plt.savefig('./fig3.png')

# %%
# Fig. 4
fig, axes = plt.subplots(2, 2, figsize=(13, 12))

for i in range(len(dir_list)):
    ax = axes.flatten()[i]

    pts = np.meshgrid(np.linspace(-1000, 1000, 201), np.linspace(0, 2000, 201))
    mappable = ax.contourf(pts[0], pts[1], pts[0], levels=[-1000, 100, 1000],
                cmap=mcolors.ListedColormap(['lightblue', 'lightgray']), zorder=-1)
    ax.set_aspect('equal')

    intv = 50
    ptx = load_res_npz(f'./res/{dir_list[i]}', 'dvel_near_x')
    pty = load_res_npz(f'./res/{dir_list[i]}', 'dvel_near_y')
    vel = load_res_npz(f'./res/{dir_list[i]}', 'dvel_near')
    Q = ax.quiver(ptx, pty, vel[:, 0], vel[:, 1], 
            scale=5e2, width=0.005, color='darkgoldenrod', pivot='tail')
    ax.quiverkey(Q, 0.68, 0.96, 50, '50 mm/year', labelpos='E', 
                fontproperties={'size':16}, zorder=10)

    ptx = load_res_npz(f'./res/{dir_list[i]}', 'dvel_far_x')
    pty = load_res_npz(f'./res/{dir_list[i]}', 'dvel_far_y')
    vel = load_res_npz(f'./res/{dir_list[i]}', 'dvel_far')
    Q = ax.quiver(ptx, pty, vel[:, 0], vel[:, 1], 
            scale=1e2, width=0.005, color='purple', pivot='tail')
    ax.quiverkey(Q, 0.68, 0.91, 10, '10 mm/year', labelpos='E', 
                fontproperties={'size':16}, zorder=10)

    ax.axvline(0, c='k', ls='--', lw=3, zorder=0)
    ax.axvline(100, c='gray', ls='-', lw=1, zorder=0)

    ax.plot([40, 150, 150, 40, 40], [300, 300, -300, -300, 300], lw=3, ls='-', c='r', zorder=0)

    ax.set_xlim(-250, 500)
    ax.set_ylim(250, 1000)

    ax.minorticks_on()
    ax.set_title(r'$\omega$ = ' + f'{w_list[i]}')

axes[0, 0].set_ylabel(r'$y$ (km)')
axes[1, 0].set_ylabel(r'$y$ (km)')
axes[1, 0].set_xlabel(r'$x$ (km)')
axes[1, 1].set_xlabel(r'$x$ (km)')
axes[0, 0].set_xticks([])
axes[0, 1].set_xticks([])
axes[0, 1].set_yticks([])
axes[1, 1].set_yticks([])
axes[0, 0].set_title('(a)', loc='left')
axes[0, 1].set_title('(b)', loc='left')
axes[1, 0].set_title('(c)', loc='left')
axes[1, 1].set_title('(d)', loc='left')

plt.tight_layout()
plt.savefig('./fig4.png')

# %%
# Fig. 5
fig = plt.figure(figsize=(18, 12))

gs = fig.add_gridspec(3, 3)

dir_list = ['3d_A2_V7.8_Gw80_Ge40',
            '3d_A1_V10.6_Gw80_Ge40',
            '3d_A0.5_V13.4_Gw80_Ge40',
            '3d_A0.25_V16.1_Gw80_Ge40']
w_list = [2, 1, 0.5, 0.25]
colors = plt.cm.RdYlBu(np.linspace(0, 1, len(w_list))[::-1])

ax = fig.add_subplot(gs[:2, 0])
ax1 = ax

for i in range(len(dir_list)):
    ptvx = load_res_npz(f'./res/{dir_list[i]}', 'dvel_line')
    pty = load_res_npz(f'./res/{dir_list[i]}', 'dvel_line_y')
    ax.plot(ptvx, pty, c=colors[i], lw=3, label=f'${w_list[i]}$')

quantity = (df_st['E_rate1']-df_st['E_rate0'])
quantity_error = np.sqrt(df_st['E_err1']**2+df_st['E_err0']**2)
ax.errorbar(quantity, df_st['Y'], xerr=quantity_error, ls="None", 
            ecolor='k', capsize=5, zorder=5)
ax.scatter(quantity, df_st['Y'], c='purple', edgecolors='k', s=50, zorder=10)

ax.axvline(0, c='k', ls='-', lw=1)

ax.minorticks_on()
ax.set_xlabel(r'$\Delta v_x$ (mm/yr)')
ax.set_ylabel(r'$y$ (km)')
ax.set_xlim(-10, 10)
ax.set_ylim(300, 1000)
ax.legend(title=r'$\omega$', loc='upper left')
ax.set_title('(a)', loc='left')
ax.set_title('Reference')

ax = fig.add_subplot(gs[2:, 0])
ax2 = ax

vx_list = []
vxerr_list = []
etai_list = []
etap_list = []

for i in range(len(dir_list)):
    vx_list.append(load_res_npz(f'./res/{dir_list[i]}', 'dvel_line')[50:71].mean())
    vxerr_list.append(load_res_npz(f'./res/{dir_list[i]}', 'dvel_line')[50:71].std())
    etai_list.append(load_res_npz(f'./res/{dir_list[i]}', 'visc_i_b').min())
    etap_list.append(load_res_npz(f'./res/{dir_list[i]}', 'visc_p_b').min())

ax.plot(w_list, vx_list, c='darkviolet', lw=2, marker='o', zorder=10)
ax.scatter(w_list, vx_list, c='darkviolet', edgecolors='k', zorder=10)
ax.errorbar(w_list, vx_list, yerr=vxerr_list, ls="None", ecolor='k', capsize=5)

ax.axhline(quantity[df_st['Y'].gt(500) & df_st['Y'].lt(700)].mean(), c='darkviolet', ls=':', lw=2)

ax.set_ylim(-0.5, 8)
ax.set_xscale('log')
ax.set_xlabel(r'$\omega$')
ax.set_ylabel(r'$\Delta v_x$ (mm/yr)', c='darkviolet')
ax.minorticks_on()

axt = ax.twinx()

axt.plot(w_list, np.power(10, etai_list), c='orangered', lw=2, ls='-', marker='o')
axt.plot(w_list, np.power(10, etap_list), c='orangered', lw=2, ls='--', marker='o')

axt.set_ylim(1e18, 2e19)
axt.set_yscale('log')
axt.minorticks_on()
ax.set_title('(b)', loc='left')

ax.xaxis.set_minor_locator(mpl.ticker.FixedLocator(w_list))
ax.xaxis.set_major_locator(mpl.ticker.NullLocator())
ax.xaxis.set_minor_formatter(mpl.ticker.ScalarFormatter())
ax.set_zorder(axt.get_zorder()+1)
ax.patch.set_visible(False)

dir_list = ['3d_A2_V7.8_Gw80_Ge40_f0.3',
            '3d_A1_V10.6_Gw80_Ge40_f0.3',
            '3d_A0.5_V13.4_Gw80_Ge40_f0.3',
            '3d_A0.25_V16.1_Gw80_Ge40_f0.3']
w_list = [2, 1, 0.5, 0.25]
colors = plt.cm.RdYlBu(np.linspace(0, 1, len(w_list))[::-1])

ax = fig.add_subplot(gs[:2, 1], sharex=ax1, sharey=ax1)
ax3 = ax

for i in range(len(dir_list)):
    ptvx = load_res_npz(f'./res/{dir_list[i]}', 'dvel_line')
    pty = load_res_npz(f'./res/{dir_list[i]}', 'dvel_line_y')
    ax.plot(ptvx, pty, c=colors[i], lw=3, label=f'${w_list[i]}$')

quantity = (df_st['E_rate1']-df_st['E_rate0'])
quantity_error = np.sqrt(df_st['E_err1']**2+df_st['E_err0']**2)
ax.errorbar(quantity, df_st['Y'], xerr=quantity_error, ls="None", 
            ecolor='k', capsize=5, zorder=5)
ax.scatter(quantity, df_st['Y'], c='purple', edgecolors='k', s=50, zorder=10)

ax.axvline(0, c='k', ls='-', lw=1)

ax.minorticks_on()
ax.set_xlabel(r'$\Delta v_x$ (mm/yr)')
ax.set_title('(c)', loc='left')
ax.set_title(r'Smaller $M_w$')

ax = fig.add_subplot(gs[2:, 1], sharex=ax2, sharey=ax2)
ax4 = ax

vx_list = []
vxerr_list = []
etai_list = []
etap_list = []

for i in range(len(dir_list)):
    vx_list.append(load_res_npz(f'./res/{dir_list[i]}', 'dvel_line')[50:71].mean())
    vxerr_list.append(load_res_npz(f'./res/{dir_list[i]}', 'dvel_line')[50:71].std())
    etai_list.append(load_res_npz(f'./res/{dir_list[i]}', 'visc_i_b').min())
    etap_list.append(load_res_npz(f'./res/{dir_list[i]}', 'visc_p_b').min())

ax.plot(w_list, vx_list, c='darkviolet', lw=2, marker='o', zorder=10)
ax.scatter(w_list, vx_list, c='darkviolet', edgecolors='k', zorder=10)
ax.errorbar(w_list, vx_list, yerr=vxerr_list, ls="None", ecolor='k', capsize=5)

ax.axhline(quantity[df_st['Y'].gt(500) & df_st['Y'].lt(700)].mean(), c='darkviolet', ls=':', lw=2)

ax.set_xscale('log')
ax.set_xlabel(r'$\omega$')
ax.minorticks_on()

axt = ax.twinx()

axt.plot(w_list, np.power(10, etai_list), c='orangered', lw=2, ls='-', marker='o')
axt.plot(w_list, np.power(10, etap_list), c='orangered', lw=2, ls='--', marker='o')

axt.set_ylim(1e18, 2e19)
axt.set_yscale('log')
axt.minorticks_on()
ax.set_title('(d)', loc='left')

ax.xaxis.set_minor_locator(mpl.ticker.FixedLocator(w_list))
ax.xaxis.set_major_locator(mpl.ticker.NullLocator())
ax.xaxis.set_minor_formatter(mpl.ticker.ScalarFormatter())
ax.set_zorder(axt.get_zorder()+1)
ax.patch.set_visible(False)


dir_list = ['3d_A2_V7.8_Gw60_Ge60',
            '3d_A1_V10.6_Gw60_Ge60',
            '3d_A0.5_V13.4_Gw60_Ge60',
            '3d_A0.25_V16.1_Gw60_Ge60']
w_list = [2, 1, 0.5, 0.25]
colors = plt.cm.RdYlBu(np.linspace(0, 1, len(w_list))[::-1])

ax = fig.add_subplot(gs[:2, 2], sharex=ax1, sharey=ax1)
ax5 = ax

for i in range(len(dir_list)):
    ptvx = load_res_npz(f'./res/{dir_list[i]}', 'dvel_line')
    pty = load_res_npz(f'./res/{dir_list[i]}', 'dvel_line_y')
    ax.plot(ptvx, pty, c=colors[i], lw=3, label=f'${w_list[i]}$')

quantity = (df_st['E_rate1']-df_st['E_rate0'])
quantity_error = np.sqrt(df_st['E_err1']**2+df_st['E_err0']**2)
ax.errorbar(quantity, df_st['Y'], xerr=quantity_error, ls="None", 
            ecolor='k', capsize=5, zorder=5)
ax.scatter(quantity, df_st['Y'], c='purple', edgecolors='k', s=50, zorder=10)

ax.axvline(0, c='k', ls='-', lw=1)

ax.minorticks_on()
ax.set_xlabel(r'$\Delta v_x$ (mm/yr)')
ax.set_title('(e)', loc='left')
ax.set_title(r'Constant $G$')

ax = fig.add_subplot(gs[2:, 2], sharex=ax2, sharey=ax2)
ax6 = ax

vx_list = []
vxerr_list = []
etai_list = []
etap_list = []

for i in range(len(dir_list)):
    vx_list.append(load_res_npz(f'./res/{dir_list[i]}', 'dvel_line')[50:71].mean())
    vxerr_list.append(load_res_npz(f'./res/{dir_list[i]}', 'dvel_line')[50:71].std())
    etai_list.append(load_res_npz(f'./res/{dir_list[i]}', 'visc_i_b').min())
    etap_list.append(load_res_npz(f'./res/{dir_list[i]}', 'visc_p_b').min())

ax.plot(w_list, vx_list, c='darkviolet', lw=2, marker='o', zorder=10)
ax.scatter(w_list, vx_list, c='darkviolet', edgecolors='k', zorder=10)
ax.errorbar(w_list, vx_list, yerr=vxerr_list, ls="None", ecolor='k', capsize=5)

ax.axhline(quantity[df_st['Y'].gt(500) & df_st['Y'].lt(700)].mean(), c='darkviolet', ls=':', lw=2)

ax.set_xscale('log')
ax.set_xlabel(r'$\omega$')
ax.minorticks_on()

axt = ax.twinx()

axt.plot(w_list, np.power(10, etai_list), c='orangered', lw=2, ls='-', marker='o')
axt.plot(w_list, np.power(10, etap_list), c='orangered', lw=2, ls='--', marker='o')

axt.set_ylim(1e18, 2e19)
axt.set_yscale('log')
axt.set_ylabel(r'$\eta$ (Pa$\cdot$s)', c='orangered')
axt.minorticks_on()
ax.set_title('(f)', loc='left')

ax.xaxis.set_minor_locator(mpl.ticker.FixedLocator(w_list))
ax.xaxis.set_major_locator(mpl.ticker.NullLocator())
ax.xaxis.set_minor_formatter(mpl.ticker.ScalarFormatter())
ax.set_zorder(axt.get_zorder()+1)
ax.patch.set_visible(False)

plt.tight_layout()
plt.savefig('./fig5.png')





# %%
# Fig. S4
dir_list = ['3d_A2_V7.8_Gw80_Ge40',
            '3d_A1_V10.6_Gw80_Ge40',
            '3d_A0.5_V13.4_Gw80_Ge40',
            '3d_A0.25_V16.1_Gw80_Ge40']
w_list = [2, 1, 0.5, 0.25]

fig, axes = plt.subplots(4, 1, figsize=(12, 10))

for i in range(len(dir_list)):
    surf_x = load_res_npz(f'./res/{dir_list[i]}', 'surf_x')
    surf_vx_l = load_res_npz(f'./res/{dir_list[i]}', 'surf_vx_l')
    surf_vx_i = load_res_npz(f'./res/{dir_list[i]}', 'surf_vx_i')
    surf_ux_c = load_res_npz(f'./res/{dir_list[i]}', 'surf_ux_c')

    ax = axes[i]
    ax.plot(surf_x, surf_vx_l, 
        'k--', lw=3, label='Long-term')
    ax.plot(surf_x, surf_vx_i, 
            'b', lw=3, label='Pre-seismic')
    ax.plot([], [], 'r', lw=3, label='Co-seismic')

    ax.set_xlim(surf_x.min(), surf_x.max())
    ax.set_ylabel(r'$v_x$ (mm/yr)', c='b')
    ax.minorticks_on()
    ax.text(3900, 5, r'$\omega$ = '+f'${w_list[i]}$', color='brown', ha='right')

    axt = ax.twinx()
    axt.plot(surf_x, surf_ux_c, 'r', lw=3)
    axt.set_ylabel(r'$u_x$ (m)', c='r')
    axt.minorticks_on()

    if(i < len(dir_list) - 1):
        ax.set_xticks([])
    else:
        ax.set_xlabel(r'$x$ (km)')

axes[0].legend(loc='lower left')
axes[0].set_title('(a)', loc='left')
axes[1].set_title('(b)', loc='left')
axes[2].set_title('(c)', loc='left')
axes[3].set_title('(d)', loc='left')

plt.tight_layout()
plt.savefig('./figS4.png')

# %%
# Fig. S5
dir_list = ['3d_A1.55_V7.8_Gw80_Ge40',
            '3d_A1_V10.6_Gw80_Ge40',
            '3d_A0.63_V13.4_Gw80_Ge40']
w_list = [1.55, 1, 0.63]
colors = ['C3', 'C1', 'C0']

fig = plt.figure(figsize=(12, 12))

gs = fig.add_gridspec(4, 2)

ax = fig.add_subplot(gs[0, :])

for i in np.arange(len(dir_list))[::-1]:
    surf_x = load_res_npz(f'./res/{dir_list[i]}', 'surf_x')
    surf_vx_l = load_res_npz(f'./res/{dir_list[i]}', 'surf_vx_l')

    ax.plot(surf_x, surf_vx_l, c=colors[i], lw=3, ls='-', label=f'${w_list[i]}$')
        
ax.set_xlabel(r'$x$ (km)')
ax.set_xlim(surf_x.min(), surf_x.max())
ax.set_ylabel(r'$v_x$ (mm/yr)')
ax.minorticks_on()
ax.legend(title=r'$\omega$', loc='upper right')
ax.set_title('(a)', loc='left')

ax = fig.add_subplot(gs[1:, 0])

for i in np.arange(len(dir_list))[::-1]:
    visc_i_b = load_res_npz(f'./res/{dir_list[i]}', 'visc_i_b')
    visc_z = load_res_npz(f'./res/{dir_list[i]}', 'visc_z')
    ax.plot(np.power(10, visc_i_b), visc_z, c=colors[i], lw=3, ls='-', label=f'${w_list[i]}$')

ax.set_xlim(1e17, 1e24)
ax.set_ylim(-800, 0)
ax.set_xscale('log')
ax.set_ylabel(r'$z$ (km)')
ax.set_xlabel(r'$\eta$ (Pa$\cdot$s)')
ax.minorticks_on()
ax.set_title(r'(b)', loc='left')

ax = fig.add_subplot(gs[1:, 1])

for i in np.arange(len(dir_list))[::-1]:
    ptvx = load_res_npz(f'./res/{dir_list[i]}', 'dvel_line')
    pty = load_res_npz(f'./res/{dir_list[i]}', 'dvel_line_y')
    ax.plot(ptvx, pty, c=colors[i], lw=3, ls='-', label=f'${w_list[i]}$')

quantity = (df_st['E_rate1']-df_st['E_rate0'])
quantity_error = np.sqrt(df_st['E_err1']**2+df_st['E_err0']**2)
ax.errorbar(quantity, df_st['Y'], xerr=quantity_error, ls="None", 
            ecolor='k', capsize=5, zorder=5)
ax.scatter(quantity, df_st['Y'], c='purple', edgecolors='k', s=50, zorder=10)

ax.axvline(0, c='k', ls='-', lw=1)

ax.minorticks_on()
ax.set_xlabel(r'$\Delta v_x$ (mm/yr)')
ax.set_ylabel(r'$y$ (km)')
ax.set_xlim(-10, 10)
ax.set_ylim(300, 1000)
ax.set_title(r'(c)', loc='left')

plt.tight_layout()
plt.show()

# %%
# Fig. S9
dir_list = ['3d_A1_V10.6_Gw80_Ge40',
            '3d_A1_V10.6_Gw80_Ge40_r0.2',
            '3d_A1_V10.6_Gw80_Ge40_r0.1',
            '3d_A1_V10.6_Gw80_Ge40_r0.05',
            '3d_A1_V10.6_Gw80_Ge40_r0.025',
            '3d_A1_V10.6_Gw80_Ge40_r0.0125']
r_list = [0.4, 0.2, 0.1, 0.05, 0.025, 0.0125]
colors = plt.cm.RdYlBu(np.linspace(0, 1, len(r_list))[::-1])

fig = plt.figure(figsize=(6, 9))

ax = fig.add_subplot(111)

for i in np.arange(len(dir_list))[::-1]:
    ptvx = load_res_npz(f'./res/{dir_list[i]}', 'dvel_line')
    pty = load_res_npz(f'./res/{dir_list[i]}', 'dvel_line_y')
    ax.plot(ptvx, pty, c=colors[i], lw=3, ls='--', label=f'${r_list[i]}$')

quantity = (df_st['E_rate1']-df_st['E_rate0'])
quantity_error = np.sqrt(df_st['E_err1']**2+df_st['E_err0']**2)
ax.errorbar(quantity, df_st['Y'], xerr=quantity_error, ls="None", 
            ecolor='k', capsize=5, zorder=5)
ax.scatter(quantity, df_st['Y'], c='purple', edgecolors='k', s=50, zorder=10)

ax.axvline(0, c='k', ls='-', lw=1)

ax.minorticks_on()
# ax.grid()
ax.set_xlabel(r'$\Delta v_x$ (mm/yr)')
ax.set_ylabel(r'$y$ (km)')
ax.set_xlim(-10, 10)
ax.set_ylim(300, 1000)
ax.legend(title=r'$\dot{\mu}$ (1/year)', loc='upper left')

plt.tight_layout()
plt.savefig('./figS9.png')

# %%
# Fig. S10
dir_list = ['3d_A1_V10.6_Gw80_Ge40',
            '3d_A1_V10.6_Gw80_Ge40_dt0.5',
            '3d_A1_V10.6_Gw80_Ge40_dt0.2',
            '3d_A1_V10.6_Gw80_Ge40_dt0.1']
dt_list = [1, 0.5, 0.2, 0.1]
colors = plt.cm.RdYlBu(np.linspace(0, 1, len(dt_list))[::-1])

fig = plt.figure(figsize=(6, 9))

ax = fig.add_subplot(111)

for i in range(len(dir_list)):
    ptvx = load_res_npz(f'./res/{dir_list[i]}', 'dvel_line')
    pty = load_res_npz(f'./res/{dir_list[i]}', 'dvel_line_y')
    ax.plot(ptvx, pty, c=colors[i], lw=3, ls='--', label=f'${dt_list[i]}$')

quantity = (df_st['E_rate1']-df_st['E_rate0'])
quantity_error = np.sqrt(df_st['E_err1']**2+df_st['E_err0']**2)
ax.errorbar(quantity, df_st['Y'], xerr=quantity_error, ls="None", 
            ecolor='k', capsize=5, zorder=5)
ax.scatter(quantity, df_st['Y'], c='purple', edgecolors='k', s=50, zorder=10)

ax.axvline(0, c='k', ls='-', lw=1)

ax.minorticks_on()
# ax.grid()
ax.set_xlabel(r'$\Delta v_x$ (mm/yr)')
ax.set_ylabel(r'$y$ (km)')
ax.set_xlim(-10, 10)
ax.set_ylim(300, 1000)
ax.legend(title=r'$\Delta$t (year)', loc='upper left')

plt.tight_layout()
plt.savefig('./figS10.png')

# %%
