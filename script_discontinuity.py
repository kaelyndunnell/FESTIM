from mpi4py import MPI
import dolfinx
import dolfinx.fem.petsc
import ufl
import numpy as np
from petsc4py import PETSc
import basix


def transfer_meshtags_to_submesh(
    mesh, entity_tag, submesh, sub_vertex_to_parent, sub_cell_to_parent
):
    """
    Transfer a meshtag from a parent mesh to a sub-mesh.
    """

    tdim = mesh.topology.dim
    cell_imap = mesh.topology.index_map(tdim)
    num_cells = cell_imap.size_local + cell_imap.num_ghosts
    mesh_to_submesh = np.full(num_cells, -1)
    mesh_to_submesh[sub_cell_to_parent] = np.arange(
        len(sub_cell_to_parent), dtype=np.int32
    )
    sub_vertex_to_parent = np.asarray(sub_vertex_to_parent)

    submesh.topology.create_connectivity(entity_tag.dim, 0)

    num_child_entities = (
        submesh.topology.index_map(entity_tag.dim).size_local
        + submesh.topology.index_map(entity_tag.dim).num_ghosts
    )
    submesh.topology.create_connectivity(submesh.topology.dim, entity_tag.dim)

    c_c_to_e = submesh.topology.connectivity(submesh.topology.dim, entity_tag.dim)
    c_e_to_v = submesh.topology.connectivity(entity_tag.dim, 0)

    child_markers = np.full(num_child_entities, 0, dtype=np.int32)

    mesh.topology.create_connectivity(entity_tag.dim, 0)
    mesh.topology.create_connectivity(entity_tag.dim, mesh.topology.dim)
    p_f_to_v = mesh.topology.connectivity(entity_tag.dim, 0)
    p_f_to_c = mesh.topology.connectivity(entity_tag.dim, mesh.topology.dim)
    sub_to_parent_entity_map = np.full(num_child_entities, -1, dtype=np.int32)
    for facet, value in zip(entity_tag.indices, entity_tag.values):
        facet_found = False
        for cell in p_f_to_c.links(facet):
            if facet_found:
                break
            if (child_cell := mesh_to_submesh[cell]) != -1:
                for child_facet in c_c_to_e.links(child_cell):
                    child_vertices = c_e_to_v.links(child_facet)
                    child_vertices_as_parent = sub_vertex_to_parent[child_vertices]
                    is_facet = np.isin(
                        child_vertices_as_parent, p_f_to_v.links(facet)
                    ).all()
                    if is_facet:
                        child_markers[child_facet] = value
                        facet_found = True
                        sub_to_parent_entity_map[child_facet] = facet
    tags = dolfinx.mesh.meshtags(
        submesh,
        entity_tag.dim,
        np.arange(num_child_entities, dtype=np.int32),
        child_markers,
    )
    tags.name = entity_tag.name
    return tags, sub_to_parent_entity_map


def bottom_boundary(x):
    return np.isclose(x[1], 0.0)


def top_boundary(x):
    return np.isclose(x[1], 1.0)


def half(x):
    return x[1] <= 0.5 + 1e-14


mesh = dolfinx.mesh.create_unit_square(
    MPI.COMM_WORLD, 20, 20, dolfinx.mesh.CellType.triangle
)

# Split domain in half and set an interface tag of 5
gdim = mesh.geometry.dim
tdim = mesh.topology.dim
fdim = tdim - 1
top_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, top_boundary)
bottom_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, bottom_boundary)
num_facets_local = (
    mesh.topology.index_map(fdim).size_local + mesh.topology.index_map(fdim).num_ghosts
)
facets = np.arange(num_facets_local, dtype=np.int32)
values = np.full_like(facets, 0, dtype=np.int32)
values[top_facets] = 1
values[bottom_facets] = 2

bottom_cells = dolfinx.mesh.locate_entities(mesh, tdim, half)
num_cells_local = (
    mesh.topology.index_map(tdim).size_local + mesh.topology.index_map(tdim).num_ghosts
)
cells = np.full(num_cells_local, 4, dtype=np.int32)
cells[bottom_cells] = 3
ct = dolfinx.mesh.meshtags(
    mesh, tdim, np.arange(num_cells_local, dtype=np.int32), cells
)
all_b_facets = dolfinx.mesh.compute_incident_entities(
    mesh.topology, ct.find(3), tdim, fdim
)
all_t_facets = dolfinx.mesh.compute_incident_entities(
    mesh.topology, ct.find(4), tdim, fdim
)
interface = np.intersect1d(all_b_facets, all_t_facets)
values[interface] = 5

mt = dolfinx.mesh.meshtags(mesh, mesh.topology.dim - 1, facets, values)

submesh_b, submesh_b_to_mesh, b_v_map = dolfinx.mesh.create_submesh(
    mesh, tdim, ct.find(3)
)[0:3]
submesh_t, submesh_t_to_mesh, t_v_map = dolfinx.mesh.create_submesh(
    mesh, tdim, ct.find(4)
)[0:3]
parent_to_sub_b = np.full(num_facets_local, -1, dtype=np.int32)
parent_to_sub_b[submesh_b_to_mesh] = np.arange(len(submesh_b_to_mesh), dtype=np.int32)
parent_to_sub_t = np.full(num_facets_local, -1, dtype=np.int32)
parent_to_sub_t[submesh_t_to_mesh] = np.arange(len(submesh_t_to_mesh), dtype=np.int32)

# We need to modify the cell maps, as for `dS` integrals of interfaces between submeshes, there is no entity to map to.
# We use the entity on the same side to fix this (as all restrictions are one-sided)

# Transfer meshtags to submesh
ft_b, b_facet_to_parent = transfer_meshtags_to_submesh(
    mesh, mt, submesh_b, b_v_map, submesh_b_to_mesh
)
ft_t, t_facet_to_parent = transfer_meshtags_to_submesh(
    mesh, mt, submesh_t, t_v_map, submesh_t_to_mesh
)

t_parent_to_facet = np.full(num_facets_local, -1)
t_parent_to_facet[t_facet_to_parent] = np.arange(len(t_facet_to_parent), dtype=np.int32)

import festim as F

my_problem = F.HydrogenTransportProblem()
my_problem.mesh = F.MeshWithMeshtags(mesh, ct, mt)