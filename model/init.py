import underworld as uw
import numpy as np
from references import *
from domain import *
from materials import *
from structures import *
import sys
sys.path.append('./utils/')
from mesh_utils import *

restart_step = -1
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

refine_left_x = -250e3 / reference_length
refine_right_x = 250e3 / reference_length
refine_buffer_x = 0e3 / reference_length
refine_ratio_x = 4.

refine_left_z = np.maximum(west_lithos_bottom_coords.min(), east_lithos_bottom_coords.min())
refine_right_z = air_depth
refine_buffer_z = 0e3 / reference_length
refine_ratio_z = 3.

mesh_deform_data = mesh.data.copy()
mesh_deform_data[:, 0] = deform_map(mesh.data[:, 0], -domain_width_west, domain_width_east, 
                                    refine_left_x, refine_right_x, refine_buffer_x, refine_ratio_x)
mesh_deform_data[:, -1] = deform_map(mesh.data[:, -1], -domain_depth, air_depth, 
                                     refine_left_z, refine_right_z, refine_buffer_z, refine_ratio_z)
with mesh.deform_mesh(isRegular=False):
    mesh.data[:] = mesh_deform_data


velocityField    = mesh.add_variable(         nodeDofCount=mesh.dim )
pressureField    = mesh.subMesh.add_variable( nodeDofCount=1 )
temperatureField = mesh.add_variable(         nodeDofCount=1 )
# temperatureDotField = mesh.add_variable(      nodeDofCount=1 )

velocityField.data[:] = 0.
pressureField.data[:] = 0.
temperatureField.data[:] = 0. 
# temperatureDotField.data[:] = 0.


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


swarm = uw.swarm.Swarm( mesh=mesh )
advector = uw.systems.SwarmAdvector( swarm=swarm, velocityField=velocityField, order=2 )
swarm.allow_parallel_nn = True

swarmLayout = uw.swarm.layouts.PerCellSpaceFillerLayout( swarm=swarm, particlesPerCell=16 )
swarm.populate_using_layout( layout=swarmLayout )

materialIndex = swarm.add_variable( dataType="int", count=1 )
material_list = [air, west_plate, east_plate, shear_zone, mantle, west_crust, east_crust]

# shear_zone_bottom_z = shear_zone_bottom_xzfunc(swarm.data[:, 0])
# slab_top_z = slab_top_xzfunc(swarm.data[:, 0])
# slab_front_z = slab_front_xzfunc(swarm.data[:, 0])
# for idx in range(swarm.data.shape[0]):
#     xp, zp = swarm.data[idx]
#     if(zp > 0.):
#         materialIndex.data[idx] = air.index
#     elif(zp >= shear_zone_bottom_z[idx]
#          and zp <= slab_top_z[idx]
#          and zp >= slab_front_z[idx]):
#         materialIndex.data[idx] = shear_zone.index
#     else:
#         materialIndex.data[idx] = mantle.index

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
        and zp >= east_lithos_bottom_coords.min()):
        # and zp >= east_lithos_bottom_coords.min()
        # and xp >= 0):
        materialIndex.data[idx] = shear_zone.index
    elif(xp >= ridge_loc
        and zp > west_lithos_bottom_z[idx]*0.5
        and zp > -west_crust_depth
        and zp < shear_zone_bottom_z[idx]):
        # and xp < 0):
        materialIndex.data[idx] = west_crust.index
    elif(xp >= ridge_loc
        and zp >= west_lithos_bottom_z[idx]
        and zp >= slab_front_z[idx]
        and zp < shear_zone_bottom_z[idx]):
        materialIndex.data[idx] = west_plate.index
    elif(xp >= slab_top_x[idx]
        and zp > -east_crust_depth):
        materialIndex.data[idx] = east_crust.index
    elif(xp >= slab_top_x[idx] 
        and zp >= east_lithos_bottom_z[idx]):
        materialIndex.data[idx] = east_plate.index
    else:
        materialIndex.data[idx] = mantle.index

previousStress = swarm.add_variable( dataType="double", count=int(mesh.dim*(mesh.dim+1)/2) )
previousStress.data[:] = 0.