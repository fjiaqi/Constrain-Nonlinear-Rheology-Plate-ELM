from mpi4py import MPI
from domain import *
if(restart_step < -1):
    from init import *
else:
    from restart import *

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