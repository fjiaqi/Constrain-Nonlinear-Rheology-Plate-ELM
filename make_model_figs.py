# %%
import numpy as np
import h5py
import pyvista as pv
pv.set_jupyter_backend('static')
pv.global_theme.font.family = 'arial'

# %%
output_dir = '/home/x-jqfang/scratch/model_2504_bkp/basics/data'

with h5py.File(f'{output_dir}/mesh_0.h5', 'r') as file:
    vertices = np.array(file['vertices'])
    en_map = np.array(file['en_map'])

with h5py.File(f'{output_dir}/temperature_0.h5', 'r') as file:
    temperature = np.array(file['data'])

# %%
nodes_per_cell = en_map.shape[1]
if(nodes_per_cell == 8):
    cell_type = pv.CellType.HEXAHEDRON
else:
    raise ValueError(f'Unsupported number of nodes per cell: {nodes_per_cell}')

# Create cells array in VTK format: [n_nodes, node1, node2, ...]
cells = np.insert(en_map, 0, nodes_per_cell, axis=1).ravel()

# Create cell types array (all cells are the same type)
cell_types = np.full(en_map.shape[0], cell_type, dtype=np.uint8)

# Create unstructured grid
grid = pv.UnstructuredGrid(cells, cell_types, vertices)
grid.point_data['temperature'] = temperature

# %%
# Create a point cloud from the vertices
point_cloud = pv.PolyData(vertices)
point_cloud['temperature'] = temperature

# Plot the point cloud
pl = pv.Plotter()
pl.add_mesh(point_cloud, scalars='temperature', cmap='coolwarm', point_size=2)
# pl.add_scalar_bar(title='Temperature')

# Adjust the view
pl.camera_position = [(10, -10, 10), (0, 0, 0), (0, 0, 1)]  # Custom camera position
# pl.camera.zoom(1.5)  # Zoom in
pl.show_axes()  # Add axes
# pl.show_grid()  # Add a grid
# pl.set_background('white')  # Set background color

# pl.show()
_ = pl.screenshot('fig2a.png', transparent_background=True)
# pl.save_graphic("fig2a.eps")

# %%
