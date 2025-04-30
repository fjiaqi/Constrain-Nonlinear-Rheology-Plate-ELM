import logging
import underworld as uw
import underworld.function as fn
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpi4py import MPI
from pathlib import Path
import csv
import sys
sys.path.append('./model/')
sys.path.append('./utils/')
from references import *
from domain import *
from structures import *
from mesh_utils import *
from mpi_utils import *
from diag_utils import *

plt.rcParams['font.size'] = 16

if(len(sys.argv) > 1):
    A = sys.argv[1]
    V = sys.argv[2]
    Gw = sys.argv[3]
    Ge = sys.argv[4]
else:
    A = 1.
    V = 10.6
    Gw = 80
    Ge = 40

param_Adisl = 1.4485e4*float(A)
param_Vdisl = 1e-6*float(V)
param_ymax = 200e6
param_Gw = 1e9*float(Gw)
param_Ge = 1e9*float(Ge)
param_Gsz = 20e9
param_wpxmax = 100e3
param_wpzmax = 80e3
param_epxmax = 400e3
param_epzmax = 80e3
param_szzmax = 120e3
param_l1 = 0.2
param_l2 = 0.8
param_fst = 0.5
param_fstic = param_fst+0.05
param_fdynmid = 0.2
param_fdynup = 0.6
param_fdynmidr = 0.4
param_fdynupr = 0.05

output_dir = Path(f'/home/x-jqfang/scratch/model_2504/3d_A{A}_V{V}_Gw{Gw}_Ge{Ge}/')
data_dir = output_dir / 'data'
figure_dir = output_dir / 'figures'
var_dir = output_dir / 'vars'
if(uw.mpi.rank == 0):
    output_dir.mkdir(exist_ok=True, parents=True)
    data_dir.mkdir(exist_ok=True, parents=True)
    figure_dir.mkdir(exist_ok=True, parents=True)
    var_dir.mkdir(exist_ok=True, parents=True)

    logger = logging.getLogger("progress_logger")
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(output_dir / f"progress.log", mode='w')
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s  (%(relativeCreated)d ms spent)\n%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.debug(f'Started using {uw.mpi.size} threads')


xres = 1024
zres = 128
yres = 256
domain_length = 2000e3 / reference_length
inner_method = 'mg'
restart_step = -100
basicinput_dir = Path('/home/x-jqfang/scratch/new_2503/basics/data/')
input_dir = Path(f'/home/x-jqfang/scratch/model_2504/3d_A{A}_V{V}_Gw{Gw}_Ge{Ge}_bkp/data/')
restart_flag = 'i'

dt_e_ratio_long = 0.1
dt_e_short = 1. * s_per_year / reference_time
dt_e_event = 1. * s_per_year / reference_time
initial_run = False
start_viscous = True
solve_viscous = False
nsteps_long = 11
nsteps_short = 0
nsteps_ic = 11
impose_event = True
nsteps_post = 6

vp_initial_guess = True
v_file = f'/home/x-jqfang/scratch/model_2504/2d_A{A}_V{V}_Gw{Gw}_Ge{Ge}/data/3d_velocity_n0.h5'
p_file = f'/home/x-jqfang/scratch/model_2504/2d_A{A}_V{V}_Gw{Gw}_Ge{Ge}/data/3d_pressure_n0.h5'

class Material:
    def __init__(self, index, label, 
                 density, diffusivity, alpha,
                 max_viscosity, min_viscosity, 
                 diff_E, diff_V, diff_A,
                 disl_n, disl_E, disl_V, disl_A,
                 shear_modulus_high, shear_modulus_low,
                 cohesion, max_yield_stress):
        self.index = index
        self.label = label
        self.density = density / reference_density
        self.diffusivity = diffusivity / reference_diffusivity
        self.alpha = alpha
        self.max_viscosity = max_viscosity / reference_viscosity
        self.min_viscosity = min_viscosity / reference_viscosity
        self.diff_E = diff_E
        self.diff_V = diff_V
        self.diff_A = diff_A / reference_viscosity
        self.disl_n = disl_n
        self.disl_E = disl_E
        self.disl_V = disl_V
        self.disl_A = disl_A / (reference_viscosity/reference_time**(1.-1./disl_n))
        self.shear_modulus_high = shear_modulus_high / reference_stress
        self.shear_modulus_low = shear_modulus_low / reference_stress
        self.cohesion = cohesion / reference_stress
        self.max_yield_stress = max_yield_stress / reference_stress


air_dict        = { 'density': 1000., 'diffusivity': 1e-5, 'alpha': 0.,
                    'max_viscosity': 1e18, 'min_viscosity': 1e18,
                    'diff_E': 0., 'diff_V': 0., 'diff_A': 2e18,
                    'disl_n': 1., 'disl_E': 0., 'disl_V': 0., 'disl_A': 2e18,
                    'shear_modulus_high': 750e9, 'shear_modulus_low': 750e9,
                    'cohesion': 500e6, 'max_yield_stress': 500e6 }

mantle_dict     = { 'density': 3300., 'diffusivity': 1e-6, 'alpha': 1.,
                    'max_viscosity': 1e24, 'min_viscosity': 1e10,
                    'diff_E': 300e3, 'diff_V': 6e-6, 'diff_A': 7.1671e10,
                    'disl_n': 3.5, 'disl_E': 540e3, 'disl_V': param_Vdisl, 'disl_A': param_Adisl,
                    'shear_modulus_high': 60e9, 'shear_modulus_low': 60e9,
                    'cohesion': 100e6, 'max_yield_stress': param_ymax }

shear_zone_dict = { 'density': 3300., 'diffusivity': 1e-6, 'alpha': 1.,
                    'max_viscosity': 1e24, 'min_viscosity': 1e10,
                    'diff_E': 0e3, 'diff_V': 0e-6, 'diff_A': 1e24,
                    'disl_n': 2.3, 'disl_E': 154e3, 'disl_V': 10e-6, 'disl_A': 1.6538e7,
                    'shear_modulus_high': param_Gsz, 'shear_modulus_low': param_Gsz,
                    'cohesion': 10e6, 'max_yield_stress': param_ymax }

crust_dict      = { 'density': 3300., 'diffusivity': 1e-6, 'alpha': 1.,
                    'max_viscosity': 1e24, 'min_viscosity': 1e10,
                    'diff_E': 300e3, 'diff_V': 6e-6, 'diff_A': 7.1671e10,
                    'disl_n': 3.5, 'disl_E': 540e3, 'disl_V': param_Vdisl, 'disl_A': param_Adisl,
                    'shear_modulus_high': param_Gsz, 'shear_modulus_low': param_Gsz,
                    'cohesion': 100e6, 'max_yield_stress': param_ymax }

west_plate_dict = { 'density': 3300., 'diffusivity': 1e-6, 'alpha': 1.,
                    'max_viscosity': 1e24, 'min_viscosity': 1e10,
                    'diff_E': 300e3, 'diff_V': 6e-6, 'diff_A': 7.1671e10,
                    'disl_n': 3.5, 'disl_E': 540e3, 'disl_V': param_Vdisl, 'disl_A': param_Adisl,
                    'shear_modulus_high': param_Gw, 'shear_modulus_low': param_Gw,
                    'cohesion': 100e6, 'max_yield_stress': param_ymax }

east_plate_dict = { 'density': 3300., 'diffusivity': 1e-6, 'alpha': 1.,
                    'max_viscosity': 1e24, 'min_viscosity': 1e10,
                    'diff_E': 300e3, 'diff_V': 6e-6, 'diff_A': 7.1671e10,
                    'disl_n': 3.5, 'disl_E': 540e3, 'disl_V': param_Vdisl, 'disl_A': param_Adisl,
                    'shear_modulus_high': param_Ge, 'shear_modulus_low': param_Ge,
                    'cohesion': 100e6, 'max_yield_stress': param_ymax }

lmantle_dict    = { 'density': 3300., 'diffusivity': 1e-6, 'alpha': 1.,
                    'max_viscosity': 1e24, 'min_viscosity': 1e10,
                    'diff_E': 300e3, 'diff_V': 3e-6, 'diff_A': 1.7241e12,
                    'disl_n': 1.0, 'disl_E': 0e3, 'disl_V': 0e-6, 'disl_A': 1e24,
                    'shear_modulus_high': 120e9, 'shear_modulus_low': 120e9,
                    'cohesion': 100e6, 'max_yield_stress': param_ymax }

air        = Material(index=0, label='air', **air_dict)
west_plate = Material(index=1, label='west_plate', **west_plate_dict)
east_plate = Material(index=2, label='east_plate', **east_plate_dict)
shear_zone = Material(index=3, label='shear_zone', **shear_zone_dict)
mantle     = Material(index=4, label='mantle', **mantle_dict)
west_crust = Material(index=5, label='west_crust', **west_plate_dict)
east_crust = Material(index=6, label='east_crust', **east_plate_dict)
lmantle    = Material(index=7, label='lmantle', **lmantle_dict)


if((restart_step < -1) or initial_run):
    if(yres > 0):
        mesh = uw.mesh.FeMesh_Cartesian( elementType = ("Q1/dQ0"), 
                                        elementRes  = (xres, yres, zres), 
                                        minCoord    = (-domain_width_west, 0, -domain_depth), 
                                        maxCoord    = ( domain_width_east, domain_length,  air_depth) )
    else:
        mesh = uw.mesh.FeMesh_Cartesian( elementType = ("Q1/dQ0"), 
                                        elementRes  = (xres, zres), 
                                        minCoord    = (-domain_width_west, -domain_depth), 
                                        maxCoord    = ( domain_width_east,  air_depth) )

    refine_left_x = -50e3 / reference_length
    refine_right_x = 450e3 / reference_length
    refine_buffer_x = 0e3 / reference_length
    refine_ratio_x = 4.

    refine_left_z = np.maximum(west_lithos_bottom_coords.min(), east_lithos_bottom_coords.min())
    refine_right_z = air_depth
    refine_buffer_z = 0e3 / reference_length
    refine_ratio_z = 4.

    mesh_deform_data = mesh.data.copy()
    mesh_deform_data[:, 0] = deform_map(mesh.data[:, 0], -domain_width_west, domain_width_east, 
                                        refine_left_x, refine_right_x, refine_buffer_x, refine_ratio_x)
    mesh_deform_data[:, -1] = deform_map(mesh.data[:, -1], -domain_depth, air_depth, 
                                        refine_left_z, refine_right_z, refine_buffer_z, refine_ratio_z)
    with mesh.deform_mesh(isRegular=False):
        mesh.data[:] = mesh_deform_data
    if(uw.mpi.rank == 0):
        logger.debug(f'Mesh initialized')
else:
    if(yres > 0):
        mesh = uw.mesh.FeMesh_Cartesian( elementType = ("Q1/dQ0"), 
                                         elementRes  = (xres, yres, zres),
                                         minCoord    = (0., 0., 0.),
                                         maxCoord    = (1., 1., 1.) )
    else:
        mesh = uw.mesh.FeMesh_Cartesian( elementType = ("Q1/dQ0"), 
                                         elementRes  = (xres, zres) )
    mesh.load(str(basicinput_dir / 'mesh_0.h5'))
    if(uw.mpi.rank == 0):
        logger.debug(f'Mesh loaded')

velocityField    = mesh.add_variable(         nodeDofCount=mesh.dim )
pressureField    = mesh.subMesh.add_variable( nodeDofCount=1 )
temperatureField = mesh.add_variable(         nodeDofCount=1 )

swarm = uw.swarm.Swarm( mesh=mesh )
advector = uw.systems.SwarmAdvector( swarm=swarm, velocityField=velocityField, order=2 )
swarm.allow_parallel_nn = True

if((restart_step < -1) or initial_run):
    velocityField.data[:] = 0.
    if(uw.mpi.rank == 0):
        logger.debug(f'Velocity initialized')
    pressureField.data[:] = 0.
    if(uw.mpi.rank == 0):
        logger.debug(f'Pressure initialized')

    bend_buffer_width = 80e3 / reference_length
    slab_front_buffer_width = 80e3 / reference_length

    slab_top_x = slab_top_zxfunc(mesh.data[:, -1])
    slab_top_z = slab_top_xzfunc(mesh.data[:, 0])
    slab_front_z = slab_front_xzfunc(mesh.data[:, 0])
    for idx in range(mesh.data.shape[0]):
        xm = mesh.data[idx, 0]
        zm = mesh.data[idx, -1]
        if(zm > 0.):
            temperatureField.data[idx] = 0.
        elif(xm < ridge_loc):
            temperatureField.data[idx] = 1.
        elif(xm >= ridge_loc and xm < 0.):
            tx = np.minimum((xm-ridge_loc)/vel_west_plate, age_max_cut)
            temperatureField.data[idx] = plate_temp_func([zm, tx])
        elif(xm >= 0
        and xm <= slab_front_coords[:, 0].max()*1.5
        and zm <= 0):
            xs = slab_top_x[idx]
            zs = slab_top_z[idx]
            zf = slab_front_z[idx]
            if(xm <= xs
            and zm <= zs
            and zm >= zf):
                pt_m = shg.Point(xm, zm)
                xd = -slab_top_line.distance(pt_m)
                tp = slab_top_line.project(pt_m)/vel_west_plate
                if(xm < bend_buffer_width):
                    buffer_coef = xm/bend_buffer_width
                    xd = (zm-zs)*(1.-buffer_coef) + xd*buffer_coef
                    pt_tmp = shg.Point(xm, zs)
                    tp = slab_top_line.project(pt_tmp)/vel_west_plate*(1-buffer_coef) + tp*buffer_coef
                temperatureField.data[idx] = slab_temp_func([xd, tp])
            elif(xm > xs
                and xm <= xs + x_arr_max
                and zm >= zs
                and zm >= slab_top_coords[:, 1].min()):
                xd = xm-xs
                pt_tmp = shg.Point(xs, zm)
                tp = slab_top_line.project(pt_tmp)/vel_west_plate
                temperatureField.data[idx] = slab_temp_func([xd, tp])
            elif(xm <= xs
                and zm <= zs
                and zm < zf):
                xd = (xm-xs)*np.cos(slab_front_slope_rad)
                tp = slab_top_line.length/vel_west_plate
                buffer_coef = np.minimum((zf-zm)*np.cos(slab_front_slope_rad)/slab_front_buffer_width, 1.)
                temperatureField.data[idx] = slab_temp_func([xd, tp])*(1.-buffer_coef) + buffer_coef
            elif(xm > xs
                and xm <= xs + x_arr_max
                and zm >= zs
                and zm < slab_top_coords[:, 1].min()):
                xd = xm-xs
                tp = slab_top_line.length/vel_west_plate
                buffer_coef = np.minimum((slab_top_coords[:, 1].min()-zm)/np.cos(slab_front_slope_rad)/slab_front_buffer_width, 1.)
                temperatureField.data[idx] = slab_temp_func([xd, tp])*(1.-buffer_coef) + buffer_coef
            else:
                tx = np.minimum(age_east_plate, age_max_cut)
                temperatureField.data[idx] = plate_temp_func([zm, tx])
        elif(xm > slab_front_coords[:, 0].max()*1.2):
            tx = np.minimum(age_east_plate, age_max_cut)
            temperatureField.data[idx] = plate_temp_func([zm, tx])
        else:
            temperatureField.data[idx] = 1.

    temperatureField.data[temperatureField.data > 0.99] = 1.
    temperatureField.data[:] = np.clip(temperatureField.data, 0., 1.)

    adiabatic = 0.3e-3/(reference_T1-reference_T0)*reference_length
    temperatureField.data[:, 0] = temperatureField.data[:, 0] - adiabatic*mesh.data[:, -1]
    
    if(uw.mpi.rank == 0):
        logger.debug(f'Temperature initialized')

    swarmLayout = uw.swarm.layouts.PerCellSpaceFillerLayout( swarm=swarm, particlesPerCell=16 )
    swarm.populate_using_layout( layout=swarmLayout )
    if(uw.mpi.rank == 0):
        logger.debug(f'Swarm initialized')

    materialIndex = swarm.add_variable( dataType="int", count=1 )
    material_list = [air, west_plate, east_plate, shear_zone, mantle, west_crust, east_crust, lmantle]

    west_lithos_bottom_z = west_lithos_bottom_xzfunc(swarm.data[:, 0])
    shear_zone_bottom_z = shear_zone_bottom_xzfunc(swarm.data[:, 0])
    slab_top_x = slab_top_zxfunc(swarm.data[:, -1])
    slab_top_z = slab_top_xzfunc(swarm.data[:, 0])
    slab_front_z = slab_front_xzfunc(swarm.data[:, 0])
    east_lithos_bottom_z = east_lithos_bottom_xzfunc(swarm.data[:, 0])
    for idx in range(swarm.data.shape[0]):
        xp = swarm.data[idx, 0]
        zp = swarm.data[idx, -1]
        if(zp > 0.):
            materialIndex.data[idx] = air.index
        elif(zp >= shear_zone_bottom_z[idx]
            and zp <= slab_top_z[idx]
            and zp >= slab_front_z[idx]
            and zp >= -param_szzmax / reference_length
            and xp >= 0):
            materialIndex.data[idx] = shear_zone.index
        elif(xp >= ridge_loc
            and xp >= -param_wpxmax / reference_length
            and zp > -west_crust_depth
            and xp < 0):
            materialIndex.data[idx] = west_crust.index
        elif(xp >= ridge_loc
            and zp < shear_zone_bottom_z[idx]
            and zp >= -param_wpzmax / reference_length
            and xp >= -param_wpxmax / reference_length):
            materialIndex.data[idx] = west_plate.index
        elif(xp >= slab_top_x[idx]
            and xp <= param_epxmax / reference_length
            and zp > -east_crust_depth):
            materialIndex.data[idx] = east_crust.index
        elif(xp >= slab_top_x[idx] 
            and zp >= -param_epzmax / reference_length
            and xp <= param_epxmax / reference_length):
            materialIndex.data[idx] = east_plate.index
        elif(zp < -lmantle_top_depth):
            materialIndex.data[idx] = lmantle.index
        else:
            materialIndex.data[idx] = mantle.index
    if(uw.mpi.rank == 0):
        logger.debug(f'Material index initialized')

    previousStress = swarm.add_variable( dataType="double", count=int(mesh.dim*(mesh.dim+1)/2) )
    previousStress.data[:] = 0.
    if(uw.mpi.rank == 0):
        logger.debug(f'Previous stress initialized')
else:
    velocityField.load(str(input_dir / f'velocity_{restart_flag}{restart_step}.h5'))
    if(uw.mpi.rank == 0):
        logger.debug(f'Velocity loaded')
    pressureField.load(str(input_dir / f'pressure_{restart_flag}{restart_step}.h5'))
    if(uw.mpi.rank == 0):
        logger.debug(f'Pressure loaded')
    temperatureField.load(str(basicinput_dir / f'temperature_0.h5'))
    if(uw.mpi.rank == 0):
        logger.debug(f'Temperature loaded')

    swarm.load(str(basicinput_dir / f'swarm_0.h5'))
    if(uw.mpi.rank == 0):
        logger.debug(f'Swarm loaded')

    materialIndex = swarm.add_variable( dataType="int", count=1 )
    material_list = [air, west_plate, east_plate, shear_zone, mantle, west_crust, east_crust, lmantle]

    materialIndex.load(str(basicinput_dir / f'material_0.h5'))
    if(uw.mpi.rank == 0):
        logger.debug(f'Material index loaded')

    previousStress = swarm.add_variable( dataType="double", count=int(mesh.dim*(mesh.dim+1)/2) )
    previousStress.load(str(input_dir / f'prestress_{restart_flag}{restart_step}.h5'))
    if(uw.mpi.rank == 0):
        logger.debug(f'Previous stress loaded')

if(vp_initial_guess and start_viscous and (restart_step < -1) and (not initial_run)):
    velocityField.load(v_file)
    if(uw.mpi.rank == 0):
        logger.debug(f'Velocity initial guess loaded')
    pressureField.load(p_file)
    if(uw.mpi.rank == 0):
        logger.debug(f'Pressure initial guess loaded')

dx_local = np.diff(np.unique(mesh.data[:, 0]))
dz_local = np.diff(np.unique(mesh.data[:, -1]))
if(yres > 0):
    dy_local = np.diff(np.unique(mesh.data[:, 1]))
else:
    dy_local = np.array([0.])
dxdydz_local_min = np.array([dx_local.min(), dy_local.min(), dz_local.min()])
dxdydz_local_max = np.array([dx_local.max(), dy_local.max(), dz_local.max()])
dxdydz_min = np.zeros(3)
MPI.COMM_WORLD.Allreduce(dxdydz_local_min, dxdydz_min, op=MPI.MIN)
dxdydz_max = np.zeros(3)
MPI.COMM_WORLD.Allreduce(dxdydz_local_max, dxdydz_max, op=MPI.MAX)

if(uw.mpi.rank == 0):
    msg = f'xres: {xres}, dx: {l_km(dxdydz_min[0]):.3e} - {l_km(dxdydz_max[0]):.3e} km\n'
    msg += f'yres: {yres}, dy: {l_km(dxdydz_min[1]):.3e} - {l_km(dxdydz_max[1]):.3e} km\n'
    msg += f'zres: {zres}, dz: {l_km(dxdydz_min[2]):.3e} - {l_km(dxdydz_max[2]):.3e} km'
    logger.debug(msg)

step = 0
time = 0.


appliedTraction = uw.mesh.MeshVariable( mesh=mesh, nodeDofCount=mesh.dim )
appliedTraction.data[:] = 0.

if(yres > 0):
    topWall = mesh.specialSets["MaxK_VertexSet"]
    bottomWall = mesh.specialSets["MinK_VertexSet"]
    leftWall = mesh.specialSets["MinI_VertexSet"]
    rightWall = mesh.specialSets["MaxI_VertexSet"]
    frontWall = mesh.specialSets["MinJ_VertexSet"]
    backWall = mesh.specialSets["MaxJ_VertexSet"]
    iWalls = leftWall + rightWall
    jWalls = frontWall + backWall
    kWalls = topWall + bottomWall

    velBC = uw.conditions.DirichletCondition( variable        = velocityField, 
                                              indexSetsPerDof = (iWalls, jWalls, kWalls) )

    trBC = uw.conditions.NeumannCondition( fn_flux=appliedTraction, 
                                           variable=velocityField,
                                           indexSetsPerDof=(jWalls+kWalls, iWalls+kWalls, iWalls+jWalls) )
else:
    topWall = mesh.specialSets["MaxJ_VertexSet"]
    bottomWall = mesh.specialSets["MinJ_VertexSet"]
    leftWall = mesh.specialSets["MinI_VertexSet"]
    rightWall = mesh.specialSets["MaxI_VertexSet"]
    iWalls = leftWall + rightWall
    jWalls = topWall + bottomWall

    velBC = uw.conditions.DirichletCondition( variable        = velocityField, 
                                              indexSetsPerDof = (iWalls, jWalls) )

    trBC = uw.conditions.NeumannCondition( fn_flux=appliedTraction, 
                                           variable=velocityField,
                                           indexSetsPerDof=(jWalls, iWalls) )


surfaceSwarm = uw.swarm.Swarm(mesh)
advector_surface = uw.systems.SwarmAdvector( swarm=surfaceSwarm, velocityField=velocityField, order=2 )
xres_surface = 501
if(yres > 0):
    yres_surface = yres+1
    surfacePoints = np.zeros((xres_surface*yres_surface, mesh.dim))
    x_surface = np.linspace(-domain_width_west, domain_width_east, xres_surface)
    y_surface = np.linspace(0, domain_length, yres_surface)
    xx_surface, yy_surface = np.meshgrid(x_surface, y_surface)
    surfacePoints[:, 0] = xx_surface.ravel()
    surfacePoints[:, 1] = yy_surface.ravel()
else:
    surfacePoints = np.zeros((xres_surface, mesh.dim))
    surfacePoints[:, 0] = np.linspace(-domain_width_west, domain_width_east, xres_surface)
if(air_depth > 0.):
    surfacePoints[:, -1] = -1e3 / reference_length
else:
    surfacePoints[:, -1] = 0.
surface_particles = surfaceSwarm.add_particles_with_coordinates( surfacePoints )

tracerSwarm = uw.swarm.Swarm(mesh)
tracer_particles = tracerSwarm.add_particles_with_coordinates( mesh.data[:] )
tracerMaterial = tracerSwarm.add_variable( dataType="int", count=1 )
tracerModulus = tracerSwarm.add_variable( dataType="double", count=1 )
tracerTemperature = tracerSwarm.add_variable( dataType="double", count=1 )
tracerVelocity = tracerSwarm.add_variable( dataType="double", count=mesh.dim )
tracerViscosity = tracerSwarm.add_variable( dataType="double", count=1 )
tracerStrainRate = tracerSwarm.add_variable( dataType="double", count=1 )
tracerStress = tracerSwarm.add_variable( dataType="double", count=1 )

boussinesq = 1.

densityMap = { material.index : material.density - boussinesq for material in material_list }
densityFn  = fn.branching.map( fn_key=materialIndex, mapping=densityMap )

diffusivityMap = { material.index : material.diffusivity for material in material_list }
diffusivityFn = fn.branching.map( fn_key=materialIndex, mapping=diffusivityMap )

alphaMap = { material.index : material.alpha for material in material_list }
alphaFn = fn.branching.map( fn_key=materialIndex, mapping=alphaMap )

unit_z = np.zeros(mesh.dim)
unit_z[-1] = -1.
buoyancyFn = (Rb * densityFn - Ra * temperatureField * alphaFn) * unit_z


maxViscosityMap = { material.index : material.max_viscosity for material in material_list }
maxViscosity = fn.branching.map( fn_key=materialIndex, mapping=maxViscosityMap )

minViscosityMap = { material.index : material.min_viscosity for material in material_list }
minViscosity = fn.branching.map( fn_key=materialIndex, mapping=minViscosityMap )

diffEMap = { material.index : material.diff_E for material in material_list }
diffE = fn.branching.map( fn_key=materialIndex, mapping=diffEMap )

diffVMap = { material.index : material.diff_V for material in material_list }
diffV = fn.branching.map( fn_key=materialIndex, mapping=diffVMap )

diffAMap = { material.index : material.diff_A for material in material_list }
diffA = fn.branching.map( fn_key=materialIndex, mapping=diffAMap )

dislNMap = { material.index : material.disl_n for material in material_list }
dislN = fn.branching.map( fn_key=materialIndex, mapping=dislNMap )

dislEMap = { material.index : material.disl_E for material in material_list }
dislE = fn.branching.map( fn_key=materialIndex, mapping=dislEMap )

dislVMap = { material.index : material.disl_V for material in material_list }
dislV = fn.branching.map( fn_key=materialIndex, mapping=dislVMap )

dislAMap = { material.index : material.disl_A for material in material_list }
dislA = fn.branching.map( fn_key=materialIndex, mapping=dislAMap )

shearModulusHighMap = { material.index : material.shear_modulus_high for material in material_list }
shearModulusHigh = fn.branching.map( fn_key=materialIndex, mapping=shearModulusHighMap )

shearModulusLowMap = { material.index : material.shear_modulus_low for material in material_list }
shearModulusLow = fn.branching.map( fn_key=materialIndex, mapping=shearModulusLowMap )

cohesionMap = { material.index : material.cohesion for material in material_list }
cohesionFn = fn.branching.map( fn_key=materialIndex, mapping=cohesionMap )

maxYieldStressMap = { material.index : material.max_yield_stress for material in material_list }
maxYieldStress = fn.branching.map( fn_key=materialIndex, mapping=maxYieldStressMap )

strainRate = fn.tensor.symmetric( velocityField.fn_gradient )
strainRate_2nd_Invariant = fn.tensor.second_invariant(strainRate)
tiny_strain_rate = 1e-25 * reference_time
slipVelocity = 2. * strainRate_2nd_Invariant * shear_zone_width

depth = fn.misc.max(-fn.input()[mesh.dim-1], 0.)
lithostaticPressure = fn.misc.max(Rb*depth, 0.)
trueLithostaticPressure = reference_density * gravity * depth*reference_length
trueTemperature = reference_T0 + temperatureField * (reference_T1-reference_T0)

factor_lithos = 0.
factor_shear_zone = 0.95
factor_trans_depth = np.array([60e3, 80e3]) / reference_length
factorTransition = ((depth-factor_trans_depth[0])
                     / (factor_trans_depth[1]-factor_trans_depth[0])
                     * (factor_lithos-factor_shear_zone)) + factor_shear_zone
porePressureFactor = fn.branching.conditional( [ (materialIndex < shear_zone.index, factor_lithos),
                                                 (materialIndex > shear_zone.index, factor_lithos),
                                                 (depth <= factor_trans_depth[0], factor_shear_zone),
                                                 (depth <= factor_trans_depth[1], factorTransition),
                                                 (True, factor_lithos) ] )

normalStressLithos = (1. - porePressureFactor) * lithostaticPressure
normal_pressure_shear_zone = 30e6 / reference_stress
normalStressShearZone = fn.misc.min(normal_pressure_shear_zone, normalStressLithos)
normalStressTransition = ((depth-factor_trans_depth[0])
                     / (factor_trans_depth[1]-factor_trans_depth[0])
                     * (normalStressLithos-normalStressShearZone)) + normal_pressure_shear_zone
normalStress = fn.branching.conditional( [ (materialIndex < shear_zone.index, normalStressLithos),
                                            (materialIndex > shear_zone.index, normalStressLithos),
                                            (depth <= factor_trans_depth[0], normalStressShearZone),
                                            (depth <= factor_trans_depth[1], normalStressTransition),
                                            (True, normalStressLithos) ] )

truePressure = normalStress * reference_stress
diffViscosity = diffA * fn.math.exp((diffE+truePressure*diffV) / (R_const*trueTemperature))
dislViscosity = dislA * (fn.math.exp((dislE+truePressure*dislV) / (dislN*R_const*trueTemperature)) 
                         * fn.math.pow(strainRate_2nd_Invariant+tiny_strain_rate, 1./dislN-1.))
pureViscosity = 1./(1./diffViscosity + 1./dislViscosity)
viscosityFn = fn.misc.max(fn.misc.min(pureViscosity, maxViscosity), minViscosity)

shearModulusFn = shearModulusHigh + (shearModulusLow - shearModulusHigh) * fn.math.pow(temperatureField, 2.)
maxwellTime = viscosityFn / shearModulusFn


def diagnose(**kwargs):
    surf_vars = getGlobalSwarmVar(np.hstack((surfaceSwarm.data,
                                             velocityField.evaluate(surfaceSwarm.data))))
    surf_loc = surf_vars[:, 0:mesh.dim]
    surf_vel = surf_vars[:, mesh.dim:2*mesh.dim]

    pt_loc, pt_vars = getGlobalMeshVar(mesh, np.hstack((materialIndex.evaluate(mesh),
                                                        slipVelocity.evaluate(mesh))))

    pt_material = pt_vars[:, 0]
    pt_slip_vel = pt_vars[:, 1]

    if(uw.mpi.rank == 0):
        arg_surf = np.argsort(surf_loc[:, 0])
        surf_vel = surf_vel[arg_surf]
        surf_loc = surf_loc[arg_surf]
        plate_vel_quantity = plate_quantity(surf_vel[:, 0], surf_loc[:, 0],
                                            ridge_loc, 0.,
                                            return_std=True,
                                            return_max=True,
                                            return_rms=True)

        slip_vel_quantity = shear_zone_quantity(pt_slip_vel, pt_material,
                                                shear_zone.index,
                                                return_std=True,
                                                return_max=True,
                                                return_rms=True)

        msg = f'Step, time (yr)\n'
        msg += f'{step:d} {t_yr(time):.3f}\n'
        msg += f'Plate velocity: mean, std, max, rms (mm/yr)\n'
        for i in range(len(plate_vel_quantity)):
            msg += f'{v_mm_yr(plate_vel_quantity[i]):.3e} '
        msg += '\n'
        msg += f'Slip velocity: mean, std, max, rms (mm/yr)\n'
        for i in range(len(slip_vel_quantity)):
            msg += f'{v_mm_yr(slip_vel_quantity[i]):.3e} '
        logger.debug(msg)

        arg_surf = np.argsort(surf_loc[:, 0])
        surf_vel = surf_vel[arg_surf]
        surf_loc = surf_loc[arg_surf]

        if(kwargs.get('plot') is not None):
            fig, ax = plt.subplots(figsize=(12, 3))
            ax.plot(l_km(surf_loc[:, 0]), v_mm_yr(surf_vel[:, 0]))
            ax.scatter(l_km(surf_loc[:, 0]), v_mm_yr(surf_vel[:, 0]), s=1)
            ax.set_title(f'Step = {step}')
            ax.set_xlim(-l_km(domain_width_west), l_km(domain_width_east))
            ax.set_xlabel('x (km)')
            ax.set_ylabel('vx (mm/yr)')
            ax.grid()
            ax.minorticks_on()
            plt.tight_layout()
            plt.savefig(figure_dir / kwargs.get('plot'))
            plt.close()

        diagnostics = np.r_[plate_vel_quantity, slip_vel_quantity]
    else:
        diagnostics = np.zeros(8)

    MPI.COMM_WORLD.Bcast(diagnostics, root=0)
    return diagnostics

def checkpoint(step, flag='', **kwargs):
    if(kwargs.get('mesh', False)):
        mesh.save(str(data_dir / f'mesh_{flag}{step:d}.h5'))
    if(kwargs.get('velocity', False)):
        velocityField.save(str(data_dir / f'velocity_{flag}{step:d}.h5'))
    if(kwargs.get('pressure', False)):
        pressureField.save(str(data_dir / f'pressure_{flag}{step:d}.h5'))
    if(kwargs.get('temperature', False)):
        temperatureField.save(str(data_dir / f'temperature_{flag}{step:d}.h5'))
    if(kwargs.get('swarm', False)):
        swarm.save(str(data_dir / f'swarm_{flag}{step:d}.h5'))
    if(kwargs.get('material', False)):
        materialIndex.save(str(data_dir / f'material_{flag}{step:d}.h5'))
    if(kwargs.get('prestress', False)):
        previousStress.save(str(data_dir / f'prestress_{flag}{step:d}.h5'))

def trace(step, flag='', **kwargs):
    if(kwargs.get('swarm', False)):
        tracerSwarm.save(str(data_dir / f'tracer_swarm_{flag}{step:d}.h5'))
    if(kwargs.get('material', False)):
        tracerMaterial.data[:] = materialIndex.evaluate(tracerSwarm.data)
        tracerMaterial.save(str(data_dir / f'tracer_material_{flag}{step:d}.h5'))
    if(kwargs.get('modulus', False)):
        tracerModulus.data[:] = shearModulusFn.evaluate(tracerSwarm.data)
        tracerModulus.save(str(data_dir / f'tracer_modulus_{flag}{step:d}.h5'))
    if(kwargs.get('temperature', False)):
        tracerTemperature.data[:] = temperatureField.evaluate(tracerSwarm.data)
        tracerTemperature.save(str(data_dir / f'tracer_temperature_{flag}{step:d}.h5'))
    if(kwargs.get('velocity', False)):
        tracerVelocity.data[:] = velocityField.evaluate(tracerSwarm.data)
        tracerVelocity.save(str(data_dir / f'tracer_velocity_{flag}{step:d}.h5'))
    if(kwargs.get('viscosity', False)):
        tracerViscosity.data[:] = viscosityFn.evaluate(tracerSwarm.data)
        tracerViscosity.save(str(data_dir / f'tracer_viscosity_{flag}{step:d}.h5'))
    if(kwargs.get('strainrate', False)):
        tracerStrainRate.data[:] = strainRate_2nd_Invariant.evaluate(tracerSwarm.data)
        tracerStrainRate.save(str(data_dir / f'tracer_strainrate_{flag}{step:d}.h5'))
    if(kwargs.get('stress', False)):
        stress_2nd_Invariant = fn.tensor.second_invariant(stressFn)
        tracerStress.data[:] = stress_2nd_Invariant.evaluate(tracerSwarm.data)
        tracerStress.save(str(data_dir / f'tracer_stress_{flag}{step:d}.h5'))

if(initial_run):
    checkpoint(0, mesh=True, temperature=True, swarm=True, material=True)
    exit()

trace(0, swarm=True, material=True, modulus=True, temperature=True)


#################################################################################################
friction_st = param_fst
friction_dyn = np.array([friction_st, friction_st, friction_st])
friction_dyn_lithos = 0.6
friction_trans_true_temp = np.array([373., 423., 623., 723.])
frictionTransitionUpper = ((trueTemperature-friction_trans_true_temp[0])
                            / (friction_trans_true_temp[1]-friction_trans_true_temp[0])
                            * (friction_dyn[1]-friction_dyn[0])) + friction_dyn[0]
frictionTransitionLower = ((trueTemperature-friction_trans_true_temp[2])
                            / (friction_trans_true_temp[3]-friction_trans_true_temp[2])
                            * (friction_dyn[2]-friction_dyn[1])) + friction_dyn[1]
frictionDynCenter = fn.branching.conditional( [ (materialIndex < shear_zone.index, friction_dyn_lithos),
                                                (materialIndex > shear_zone.index, friction_dyn_lithos),
                                                (trueTemperature <= friction_trans_true_temp[0], friction_dyn[0]),
                                                (trueTemperature <= friction_trans_true_temp[1], frictionTransitionUpper),
                                                (trueTemperature <= friction_trans_true_temp[2], friction_dyn[1]),
                                                (trueTemperature <= friction_trans_true_temp[3], frictionTransitionLower),
                                                (True, friction_dyn[2]) ] )
friction_dyn_edge = friction_st
yFactorFriction = fn.branching.conditional( [ (fn.input()[1] < param_l2,
                                                fn.math.exp(-0.5*fn.math.pow(fn.input()[1]/param_l1, 2.)) ),
                                                (True, 0.) ] )
frictionDynEdge = fn.branching.conditional( [ (materialIndex < shear_zone.index, friction_dyn_lithos),
                                              (materialIndex > shear_zone.index, friction_dyn_lithos),
                                              (True, friction_dyn_edge) ] )
frictionFn = frictionDynCenter * yFactorFriction + frictionDynEdge * (1.-yFactorFriction)


#################################################################################################
if(restart_step < -1):
    if(start_viscous):
        flag = 'v'

        yieldStress = fn.misc.min(cohesionFn + frictionFn * normalStress, maxYieldStress)

        plastViscosity = yieldStress / (2.*strainRate_2nd_Invariant+tiny_strain_rate)
        viscosityType = fn.branching.conditional( [ (plastViscosity < viscosityFn, 0),
                                                    (dislViscosity < diffViscosity, 1), 
                                                    (True, 2) ] )
        viscosityEff = fn.exception.SafeMaths(fn.misc.min(viscosityFn, plastViscosity))

        viscousStress      = 2. * viscosityEff * strainRate
        stressFn           = viscousStress

        if(solve_viscous):
            stokes = uw.systems.Stokes( velocityField = velocityField,
                                        pressureField = pressureField,
                                        conditions    = [velBC, trBC],
                                        fn_viscosity  = viscosityEff,
                                        fn_bodyforce  = buoyancyFn )

            solver = uw.systems.Solver( stokes )

            solver.set_inner_method(inner_method)
            solver.set_penalty(1e3)
            solver.set_inner_rtol(1e-4)

            if(uw.mpi.rank == 0):
                logger.debug(f'Started solving viscous...')

            step = 0

            solver.solve( nonLinearIterate=True, nonLinearTolerance=1e-2 )

            if(yres <= 64):
                diagnostics = diagnose(plot=f'vx_{flag}{step}.png')
            else:
                if(uw.mpi.rank == 0):
                    logger.debug(f'Step, time (yr)\n{step:d} {t_yr(time):.3f}')

        trace(step, flag, velocity=True, viscosity=True, strainrate=True, stress=True)

        stress_data = stressFn.evaluate(swarm)
        previousStress.data[:] = stress_data[:]

        # checkpoint(step, flag, velocity=True, pressure=True, prestress=True)

    flag = 'l'
    dt_e = west_plate.max_viscosity/west_plate.shear_modulus_high * dt_e_ratio_long

    if(nsteps_long > 0):
        yieldStress = fn.misc.min(cohesionFn + frictionFn * normalStress, maxYieldStress)

        strainRateEff = strainRate + previousStress / (2. * shearModulusFn * dt_e)
        strainRateEff_2nd_Invariant = fn.tensor.second_invariant(strainRateEff)

        elastViscosity = dt_e / (1./shearModulusFn + dt_e/viscosityFn)
        plastViscosity = yieldStress / (2.*strainRateEff_2nd_Invariant+tiny_strain_rate)
        viscosityType = fn.branching.conditional( [ (plastViscosity < elastViscosity, 0),
                                                    (dislViscosity < diffViscosity, 1), 
                                                    (True, 2) ] )
        viscosityEff = fn.exception.SafeMaths(fn.misc.min(elastViscosity, plastViscosity))

        viscousStress      = 2. * viscosityEff * strainRate
        tauHistoryFn       = viscosityEff / ( shearModulusFn * dt_e ) * previousStress
        viscoelasticStress = 2. * viscosityEff * strainRateEff
        stressFn           = viscoelasticStress

        stokes = uw.systems.Stokes( velocityField = velocityField,
                                    pressureField = pressureField,
                                    voronoi_swarm = swarm,
                                    conditions    = [velBC, trBC],
                                    fn_viscosity  = viscosityEff,
                                    fn_bodyforce  = buoyancyFn,
                                    fn_stresshistory = tauHistoryFn )

        solver = uw.systems.Solver( stokes )

        phi_dt = 1.
        dt = dt_e*phi_dt

        step = 0
        time = 0.

        solver.set_inner_method(inner_method)
        solver.set_penalty(1e3)
        solver.set_inner_rtol(1e-4)

        if(uw.mpi.rank == 0):
            logger.debug(f'Started solving long-term...')

        while(step < nsteps_long):
            solver.solve( nonLinearIterate=True, nonLinearTolerance=1e-2 )

            if(yres <= 64):
                diagnostics = diagnose(plot=f'vx_{flag}{step}.png')
            else:
                if(uw.mpi.rank == 0):
                    logger.debug(f'Step, time (yr)\n{step:d} {t_yr(time):.3f}')
            
            dt = dt_e*phi_dt

            if(step == nsteps_long-1):
                trace(step, flag, velocity=True, viscosity=True, strainrate=True, stress=True)
            elif(step % 10 == 0):
                trace(step, flag, velocity=True, viscosity=True, strainrate=True, stress=True)

            stress_data = stressFn.evaluate(swarm)
            previousStress.data[:] = dt/dt_e * stress_data[:] + (1.-dt/dt_e) * previousStress.data[:]

            # if(step == nsteps_long-1):
            #     checkpoint(step, flag, velocity=True, pressure=True, prestress=True)
            # elif(step % 10 == 0):
                # checkpoint(step, flag, velocity=True, pressure=True, prestress=True)

            step = step + 1
            time = time + dt


#################################################################################################
flag = 's'
dt_e = dt_e_short

if(nsteps_short > 0):
    # checkpoint(-1, flag, velocity=True, pressure=True, prestress=True)

    yieldStress = fn.misc.min(cohesionFn + frictionFn * normalStress, maxYieldStress)

    strainRateEff = strainRate + previousStress / (2. * shearModulusFn * dt_e)
    strainRateEff_2nd_Invariant = fn.tensor.second_invariant(strainRateEff)

    elastViscosity = dt_e / (1./shearModulusFn + dt_e/viscosityFn)
    plastViscosity = yieldStress / (2.*strainRateEff_2nd_Invariant+tiny_strain_rate)
    viscosityType = fn.branching.conditional( [ (plastViscosity < elastViscosity, 0),
                                                (dislViscosity < diffViscosity, 1), 
                                                (True, 2) ] )
    viscosityEff = fn.exception.SafeMaths(fn.misc.min(elastViscosity, plastViscosity))

    viscousStress      = 2. * viscosityEff * strainRate
    tauHistoryFn       = viscosityEff / ( shearModulusFn * dt_e ) * previousStress
    viscoelasticStress = 2. * viscosityEff * strainRateEff
    stressFn           = viscoelasticStress

    stokes = uw.systems.Stokes( velocityField = velocityField,
                                pressureField = pressureField,
                                voronoi_swarm = swarm,
                                conditions    = [velBC, trBC],
                                fn_viscosity  = viscosityEff,
                                fn_bodyforce  = buoyancyFn,
                                fn_stresshistory = tauHistoryFn )

    solver = uw.systems.Solver( stokes )

    phi_dt = 1.
    dt = dt_e*phi_dt

    step = 0
    time = 0.

    solver.set_inner_method(inner_method)
    solver.set_penalty(1e1)
    solver.set_inner_rtol(1e-4)

    while(step < nsteps_short):
        solver.solve( nonLinearIterate=True, nonLinearTolerance=1e-2 )

        if(yres <= 64):
            diagnostics = diagnose(plot=f'vx_{flag}{step}.png')
        else:
            if(uw.mpi.rank == 0):
                logger.debug(f'Step, time (yr)\n{step:d} {t_yr(time):.3f}')
        
        dt = dt_e*phi_dt

        if(step == nsteps_short-1):
            trace(step, flag, velocity=True, viscosity=True, strainrate=True, stress=True)
        elif(step % 5 == 0):
            trace(step, flag, velocity=True, viscosity=True, strainrate=True, stress=True)

        stress_data = stressFn.evaluate(swarm)
        previousStress.data[:] = dt/dt_e * stress_data[:] + (1.-dt/dt_e) * previousStress.data[:]

        # if(step == nsteps_short-1):
        #     checkpoint(step, flag, velocity=True, pressure=True, prestress=True)
        # elif(step % 5 == 0):
            # checkpoint(step, flag, velocity=True, pressure=True, prestress=True)

        step = step + 1
        time = time + dt


#################################################################################################
flag = 'i'
dt_e = dt_e_short

if(nsteps_ic > 0):
    friction_st = param_fstic
    friction_dyn = np.array([friction_st, friction_st, friction_st])
    friction_dyn_lithos = 0.6
    friction_trans_true_temp = np.array([373., 423., 623., 723.])
    frictionTransitionUpper = ((trueTemperature-friction_trans_true_temp[0])
                                / (friction_trans_true_temp[1]-friction_trans_true_temp[0])
                                * (friction_dyn[1]-friction_dyn[0])) + friction_dyn[0]
    frictionTransitionLower = ((trueTemperature-friction_trans_true_temp[2])
                                / (friction_trans_true_temp[3]-friction_trans_true_temp[2])
                                * (friction_dyn[2]-friction_dyn[1])) + friction_dyn[1]
    frictionDynCenter = fn.branching.conditional( [ (materialIndex < shear_zone.index, friction_dyn_lithos),
                                                    (materialIndex > shear_zone.index, friction_dyn_lithos),
                                                    (trueTemperature <= friction_trans_true_temp[0], friction_dyn[0]),
                                                    (trueTemperature <= friction_trans_true_temp[1], frictionTransitionUpper),
                                                    (trueTemperature <= friction_trans_true_temp[2], friction_dyn[1]),
                                                    (trueTemperature <= friction_trans_true_temp[3], frictionTransitionLower),
                                                    (True, friction_dyn[2]) ] )
    friction_dyn_edge = friction_st
    yFactorFriction = fn.branching.conditional( [ (fn.input()[1] < param_l2,
                                                fn.math.exp(-0.5*fn.math.pow(fn.input()[1]/param_l1, 2.)) ),
                                                (True, 0.) ] )
    frictionDynEdge = fn.branching.conditional( [ (materialIndex < shear_zone.index, friction_dyn_lithos),
                                                (materialIndex > shear_zone.index, friction_dyn_lithos),
                                                (True, friction_dyn_edge) ] )
    frictionFn = frictionDynCenter * yFactorFriction + frictionDynEdge * (1.-yFactorFriction)
    yieldStress = fn.misc.min(cohesionFn + frictionFn * normalStress, maxYieldStress)

    strainRateEff = strainRate + previousStress / (2. * shearModulusFn * dt_e)
    strainRateEff_2nd_Invariant = fn.tensor.second_invariant(strainRateEff)

    elastViscosity = dt_e / (1./shearModulusFn + dt_e/viscosityFn)
    plastViscosity = yieldStress / (2.*strainRateEff_2nd_Invariant+tiny_strain_rate)
    viscosityType = fn.branching.conditional( [ (plastViscosity < elastViscosity, 0),
                                                (dislViscosity < diffViscosity, 1), 
                                                (True, 2) ] )
    viscosityEff = fn.exception.SafeMaths(fn.misc.min(elastViscosity, plastViscosity))

    viscousStress      = 2. * viscosityEff * strainRate
    tauHistoryFn       = viscosityEff / ( shearModulusFn * dt_e ) * previousStress
    viscoelasticStress = 2. * viscosityEff * strainRateEff
    stressFn           = viscoelasticStress

    stokes = uw.systems.Stokes( velocityField = velocityField,
                                pressureField = pressureField,
                                voronoi_swarm = swarm,
                                conditions    = [velBC, trBC],
                                fn_viscosity  = viscosityEff,
                                fn_bodyforce  = buoyancyFn,
                                fn_stresshistory = tauHistoryFn )

    solver = uw.systems.Solver( stokes )

    phi_dt = 1.
    dt = dt_e*phi_dt

    step = 0
    time = 0.

    solver.set_inner_method(inner_method)
    solver.set_penalty(1e1)
    solver.set_inner_rtol(1e-4)

    while(step < nsteps_ic):
        solver.solve( nonLinearIterate=True, nonLinearTolerance=1e-2 )

        if(yres <= 64):
            diagnostics = diagnose(plot=f'vx_{flag}{step}.png')
        else:
            if(uw.mpi.rank == 0):
                logger.debug(f'Step, time (yr)\n{step:d} {t_yr(time):.3f}')

        
        dt = dt_e*phi_dt

        if(step == nsteps_ic-1):
            trace(step, flag, velocity=True, viscosity=True, strainrate=True, stress=True)
        elif(step % 5 == 0):
            trace(step, flag, velocity=True, viscosity=True, strainrate=True, stress=True)

        stress_data = stressFn.evaluate(swarm)
        previousStress.data[:] = dt/dt_e * stress_data[:] + (1.-dt/dt_e) * previousStress.data[:]

        if(step == nsteps_ic-1):
            checkpoint(step, flag, velocity=True, pressure=True, prestress=True)
        # elif(step % 5 == 0):
            # checkpoint(step, flag, velocity=True, pressure=True, prestress=True)

        step = step + 1
        time = time + dt


#################################################################################################
if(impose_event):
    flag = 'c'
    dt_e = dt_e_event
    # checkpoint(-1, flag, velocity=True, pressure=True, prestress=True)

    friction_st = param_fstic
    friction_dyn = np.array([param_fdynup, param_fdynmid, friction_st])
    friction_dyn_lithos = 0.6
    friction_trans_true_temp = np.array([373., 423., 623., 723.])
    frictionTransitionUpper = ((trueTemperature-friction_trans_true_temp[0])
                                / (friction_trans_true_temp[1]-friction_trans_true_temp[0])
                                * (friction_dyn[1]-friction_dyn[0])) + friction_dyn[0]
    frictionTransitionLower = ((trueTemperature-friction_trans_true_temp[2])
                                / (friction_trans_true_temp[3]-friction_trans_true_temp[2])
                                * (friction_dyn[2]-friction_dyn[1])) + friction_dyn[1]
    frictionDynCenter = fn.branching.conditional( [ (materialIndex < shear_zone.index, friction_dyn_lithos),
                                                    (materialIndex > shear_zone.index, friction_dyn_lithos),
                                                    (trueTemperature <= friction_trans_true_temp[0], friction_dyn[0]),
                                                    (trueTemperature <= friction_trans_true_temp[1], frictionTransitionUpper),
                                                    (trueTemperature <= friction_trans_true_temp[2], friction_dyn[1]),
                                                    (trueTemperature <= friction_trans_true_temp[3], frictionTransitionLower),
                                                    (True, friction_dyn[2]) ] )
    friction_dyn_edge = friction_st
    yFactorFriction = fn.branching.conditional( [ (fn.input()[1] < param_l2,
                                                fn.math.exp(-0.5*fn.math.pow(fn.input()[1]/param_l1, 2.)) ),
                                                (True, 0.) ] )
    frictionDynEdge = fn.branching.conditional( [ (materialIndex < shear_zone.index, friction_dyn_lithos),
                                                (materialIndex > shear_zone.index, friction_dyn_lithos),
                                                (True, friction_dyn_edge) ] )
    frictionFn = frictionDynCenter * yFactorFriction + frictionDynEdge * (1.-yFactorFriction)
    yieldStress = fn.misc.min(cohesionFn + frictionFn * normalStress, maxYieldStress)

    strainRateEff = strainRate + previousStress / (2. * shearModulusFn * dt_e)
    strainRateEff_2nd_Invariant = fn.tensor.second_invariant(strainRateEff)

    elastViscosity = dt_e / (1./shearModulusFn + dt_e/viscosityFn)
    plastViscosity = yieldStress / (2.*strainRateEff_2nd_Invariant+tiny_strain_rate)
    viscosityType = fn.branching.conditional( [ (plastViscosity < elastViscosity, 0),
                                                (dislViscosity < diffViscosity, 1), 
                                                (True, 2) ] )
    viscosityEff = fn.exception.SafeMaths(fn.misc.min(elastViscosity, plastViscosity))

    viscousStress      = 2. * viscosityEff * strainRate
    tauHistoryFn       = viscosityEff / ( shearModulusFn * dt_e ) * previousStress
    viscoelasticStress = 2. * viscosityEff * strainRateEff
    stressFn           = viscoelasticStress

    stokes = uw.systems.Stokes( velocityField = velocityField,
                                pressureField = pressureField,
                                voronoi_swarm = swarm,
                                conditions    = [velBC, trBC],
                                fn_viscosity  = viscosityEff,
                                fn_bodyforce  = buoyancyFn,
                                fn_stresshistory = tauHistoryFn )

    solver = uw.systems.Solver( stokes )

    phi_dt = 1.
    dt = dt_e*phi_dt

    step = 0
    time = 0.

    solver.set_inner_method(inner_method)
    solver.set_penalty(1e1)
    solver.set_inner_rtol(1e-4)

    if(uw.mpi.rank == 0):
        logger.debug(f'Started solving coseismic...')

    solver.solve( nonLinearIterate=True, nonLinearTolerance=1e-3 )

    if(yres <= 64):
        diagnostics = diagnose(plot=f'vx_{flag}{step}.png')
    else:
        if(uw.mpi.rank == 0):
            logger.debug(f'Step, time (yr)\n{step:d} {t_yr(time):.3f}')

    dt = dt_e*phi_dt

    trace(step, flag, velocity=True, viscosity=True, strainrate=True, stress=True)

    stress_data = stressFn.evaluate(swarm)
    previousStress.data[:] = dt/dt_e * stress_data[:] + (1.-dt/dt_e) * previousStress.data[:]

    checkpoint(step, flag, velocity=True, pressure=True, prestress=True)


#################################################################################################
flag = 'p'
dt_e = dt_e_event

phi_dt = 1.
dt = dt_e*phi_dt

step = 0
time = 0.

if(uw.mpi.rank == 0):
    logger.debug(f'Started solving postseismic...')

while(step < nsteps_post):
    friction_st = param_fstic
    up_fric = np.maximum(param_fdynup - (step+1)*param_fdynupr, friction_st)
    mid_fric = np.minimum(param_fdynmid + (step+1)*param_fdynmidr, friction_st)
    friction_dyn = np.array([up_fric, mid_fric, friction_st])
    friction_dyn_lithos = 0.6
    friction_trans_true_temp = np.array([373., 423., 623., 723.])
    frictionTransitionUpper = ((trueTemperature-friction_trans_true_temp[0])
                                / (friction_trans_true_temp[1]-friction_trans_true_temp[0])
                                * (friction_dyn[1]-friction_dyn[0])) + friction_dyn[0]
    frictionTransitionLower = ((trueTemperature-friction_trans_true_temp[2])
                                / (friction_trans_true_temp[3]-friction_trans_true_temp[2])
                                * (friction_dyn[2]-friction_dyn[1])) + friction_dyn[1]
    frictionDynCenter = fn.branching.conditional( [ (materialIndex < shear_zone.index, friction_dyn_lithos),
                                                    (materialIndex > shear_zone.index, friction_dyn_lithos),
                                                    (trueTemperature <= friction_trans_true_temp[0], friction_dyn[0]),
                                                    (trueTemperature <= friction_trans_true_temp[1], frictionTransitionUpper),
                                                    (trueTemperature <= friction_trans_true_temp[2], friction_dyn[1]),
                                                    (trueTemperature <= friction_trans_true_temp[3], frictionTransitionLower),
                                                    (True, friction_dyn[2]) ] )
    friction_dyn_edge = friction_st
    yFactorFriction = fn.branching.conditional( [ (fn.input()[1] < param_l2,
                                                fn.math.exp(-0.5*fn.math.pow(fn.input()[1]/param_l1, 2.)) ),
                                                (True, 0.) ] )
    frictionDynEdge = fn.branching.conditional( [ (materialIndex < shear_zone.index, friction_dyn_lithos),
                                                (materialIndex > shear_zone.index, friction_dyn_lithos),
                                                (True, friction_dyn_edge) ] )
    frictionFn = frictionDynCenter * yFactorFriction + frictionDynEdge * (1.-yFactorFriction)
    yieldStress = fn.misc.min(cohesionFn + frictionFn * normalStress, maxYieldStress)

    strainRateEff = strainRate + previousStress / (2. * shearModulusFn * dt_e)
    strainRateEff_2nd_Invariant = fn.tensor.second_invariant(strainRateEff)

    elastViscosity = dt_e / (1./shearModulusFn + dt_e/viscosityFn)
    plastViscosity = yieldStress / (2.*strainRateEff_2nd_Invariant+tiny_strain_rate)
    viscosityType = fn.branching.conditional( [ (plastViscosity < elastViscosity, 0),
                                                (dislViscosity < diffViscosity, 1), 
                                                (True, 2) ] )
    viscosityEff = fn.exception.SafeMaths(fn.misc.min(elastViscosity, plastViscosity))

    viscousStress      = 2. * viscosityEff * strainRate
    tauHistoryFn       = viscosityEff / ( shearModulusFn * dt_e ) * previousStress
    viscoelasticStress = 2. * viscosityEff * strainRateEff
    stressFn           = viscoelasticStress

    stokes = uw.systems.Stokes( velocityField = velocityField,
                                pressureField = pressureField,
                                voronoi_swarm = swarm,
                                conditions    = [velBC, trBC],
                                fn_viscosity  = viscosityEff,
                                fn_bodyforce  = buoyancyFn,
                                fn_stresshistory = tauHistoryFn )

    solver = uw.systems.Solver( stokes )

    solver.set_inner_method(inner_method)
    solver.set_penalty(1e1)
    solver.set_inner_rtol(1e-4)

    solver.solve( nonLinearIterate=True, nonLinearTolerance=1e-3 )

    if(yres <= 64):
        diagnostics = diagnose(plot=f'vx_{flag}{step}.png')
    else:
        if(uw.mpi.rank == 0):
            logger.debug(f'Step, time (yr)\n{step:d} {t_yr(time):.3f}')

    
    dt = dt_e*phi_dt

    if(step == 0 or step == nsteps_post-1):
        trace(step, flag, velocity=True, viscosity=True, strainrate=True, stress=True)
    elif(step % 1 == 0):
        trace(step, flag, velocity=True, viscosity=True, strainrate=True, stress=True)

    stress_data = stressFn.evaluate(swarm)
    previousStress.data[:] = dt/dt_e * stress_data[:] + (1.-dt/dt_e) * previousStress.data[:]

    # if(step == 0 or step == nsteps_post-1):
    #     checkpoint(step, flag, velocity=True, pressure=True, prestress=True)
    # elif(step % 1 == 0):
        # checkpoint(step, flag, velocity=True, pressure=True, prestress=True)

    step = step + 1
    time = time + dt

if(uw.mpi.rank == 0):
    logger.debug(f'Completed using {uw.mpi.size} threads\n')
