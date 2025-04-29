import underworld as uw
import numpy as np
from references import *
from domain import *
from materials import *

restart_step = 10
input_dir = Path('./input/')
restart_flag = 's'
if(yres > 0):
    mesh = uw.mesh.FeMesh_Cartesian( elementType = ("Q1/dQ0"), 
                                     elementRes  = (xres, yres, zres),
                                     minCoord    = (0., 0., 0.),
                                     maxCoord    = (1., 1., 1.) )
else:
    mesh = uw.mesh.FeMesh_Cartesian( elementType = ("Q1/dQ0"), 
                                     elementRes  = (xres, zres) )
mesh.load(str(input_dir / 'mesh_0.h5'))


velocityField    = mesh.add_variable(         nodeDofCount=mesh.dim )
pressureField    = mesh.subMesh.add_variable( nodeDofCount=1 )
temperatureField = mesh.add_variable(         nodeDofCount=1 )
# temperatureDotField = mesh.add_variable(      nodeDofCount=1 )

# velocityField.data[:] = 0.
# pressureField.data[:] = 0.
velocityField.load(str(input_dir / f'velocity_{restart_flag}{restart_step}.h5'))
pressureField.load(str(input_dir / f'pressure_{restart_flag}{restart_step}.h5'))
temperatureField.load(str(input_dir / f'temperature_0.h5'))
# temperatureField.load(str(input_dir / f'temperature_{restart_step}.h5'))


swarm = uw.swarm.Swarm( mesh=mesh )
advector = uw.systems.SwarmAdvector( swarm=swarm, velocityField=velocityField, order=2 )
swarm.allow_parallel_nn = True

swarm.load(str(input_dir / f'swarm_0.h5'))
# swarm.load(str(input_dir / f'swarm_{restart_step}.h5'))

materialIndex = swarm.add_variable( dataType="int", count=1 )
# material_list = [air, shear_zone, mantle]
material_list = [air, west_plate, east_plate, shear_zone, mantle, west_crust, east_crust]

materialIndex.load(str(input_dir / f'material_0.h5'))
# materialIndex.load(str(input_dir / f'material_{restart_step}.h5'))

previousStress = swarm.add_variable( dataType="double", count=int(mesh.dim*(mesh.dim+1)/2) )
# previousStress.data[:] = 0.
previousStress.load(str(input_dir / f'prestress_{restart_flag}{restart_step}.h5'))