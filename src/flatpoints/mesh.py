"""Meshes: Poisson reconstruction, cut into disks, projection of the points."""
import heapq

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

POISSON_DEPTH = 8  # octree depth of the Poisson reconstruction
KNN_NORMALS = 16   # neighbours used to estimate and orient the normals
TRIM = 2.0         # open surface: mesh kept within TRIM x the point spacing of the cloud


# === Reconstruction ===

def poisson_mesh(points, depth=POISSON_DEPTH, trim=None):
    """(V, F) mesh of the surface sampled by `points`, screened Poisson: its largest piece.

    Poisson always returns a closed surface. For an open surface, `trim` removes what it
    invents away from the samples: the vertices farther from the cloud than `trim` times
    the local point spacing.
    """
    import open3d as o3d
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(KNN_NORMALS))
    try:
        cloud.orient_normals_consistent_tangent_plane(KNN_NORMALS)
    except RuntimeError:  # exactly planar cloud: its Delaunay (qhull) is degenerate
        jitter = 1e-6 * np.ptp(points, 0).max() * np.random.default_rng(0).standard_normal(points.shape)
        shaken = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points + jitter))
        shaken.normals = cloud.normals
        shaken.orient_normals_consistent_tangent_plane(KNN_NORMALS)
        cloud.normals = shaken.normals
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cloud, depth=depth)
    if trim is not None:
        tree = cKDTree(points)
        spacing = tree.query(points, k=7)[0][:, 1:].mean(1)
        d, i = tree.query(np.asarray(mesh.vertices))
        mesh.remove_vertices_by_mask(d > trim * spacing[i])
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    piece, sizes, _ = mesh.cluster_connected_triangles()
    mesh.remove_triangles_by_mask(np.asarray(piece) != np.argmax(sizes))
    mesh.remove_unreferenced_vertices()
    return np.asarray(mesh.vertices), np.asarray(mesh.triangles)


# === Cuts ===

def disk_cut(V, F):
    """Cut the open mesh (V, F) into one disk: (F,) 0 for the triangles of the disk, -1
    for the others.

    The disk grows from the triangle farthest from the boundary (along the edges of the
    mesh), the closest triangles first; handles and holes are left open (see `grow`).
    """
    E = np.unique(np.sort(F[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), 1), axis=0)
    graph = coo_matrix((np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1), E.T),
                       shape=(len(V),) * 2).tocsr()
    nbr = neighbours(F)
    border = np.unique(F[nbr < 0])
    depth = dijkstra(graph, directed=False, indices=border, min_only=True)
    seed = np.argmax(depth[F].min(1))
    d = dijkstra(graph, directed=False, indices=F[seed], min_only=True)[F].mean(1)
    side = np.full(len(F), -1)
    side[seed] = 0
    side = grow(F, nbr, side, [d, d], [np.ones(len(F), bool), np.zeros(len(F), bool)])
    return drop_ears(nbr, side)


def hemisphere_cut(V, F, normal, threshold):
    """(F,) 0 (left), 1 (right) or -1 (out) for each triangle of the closed mesh:
    two disks, either side of `threshold` on the axis `normal`. Each grows from its extreme
    triangle (see `grow`); triangles of a handle stay out of both.
    """
    x = V[F, normal].mean(1)
    nbr = neighbours(F)
    side = np.full(len(F), -1)
    side[np.argmin(x)], side[np.argmax(x)] = 0, 1
    right = x > threshold
    side = grow(F, nbr, side, [x, -x], [~right, right])
    d = np.abs(x - threshold)
    side = grow(F, nbr, side, [d, d], [side < 0, side < 0])
    return drop_ears(nbr, side)


def grow(F, nbr, side, keys, allowed):
    """Grow faces 0 and 1 from `side` (-1: free), keeping each a topological disk.

    A free triangle joins face s if it shares two edges with it, or one edge and a third
    vertex not yet in it. Candidates come in increasing `keys[s]`, among `allowed[s]`.
    Returns the new `side`.
    """
    side = side.copy()
    Fl, nbl, sl = F.tolist(), nbr.tolist(), side.tolist()
    inside = [np.zeros(F.max() + 1, bool) for _ in (0, 1)]
    for s in (0, 1):
        inside[s][F[side == s]] = True
    inside = [m.tolist() for m in inside]
    keys = [k.tolist() for k in keys]
    allowed = [a.tolist() for a in allowed]

    queue = []
    for s in (0, 1):  # the free neighbours of each face
        for t in np.flatnonzero(side == s).tolist():
            queue += [(keys[s][u], s, u) for u in nbl[t] if u >= 0 and allowed[s][u]]
    heapq.heapify(queue)
    while queue:
        _, s, t = heapq.heappop(queue)
        if sl[t] >= 0:
            continue
        shared = [k for k, u in enumerate(nbl[t]) if u >= 0 and sl[u] == s]
        if len(shared) == 1:
            if inside[s][Fl[t][(shared[0] + 2) % 3]]:  # third vertex already in the face
                continue
        elif len(shared) != 2:
            continue
        sl[t] = s
        for v in Fl[t]:
            inside[s][v] = True
        for u in nbl[t]:
            if u >= 0 and sl[u] < 0 and allowed[s][u]:
                heapq.heappush(queue, (keys[s][u], s, u))
    return np.array(sl)


def drop_ears(nbr, side):
    """Free the triangles with two or three edges on the boundary of their face (their
    tip would be a boundary vertex of a single triangle, which the rectangular conformal
    map does not allow), until there is none."""
    side = side.copy()
    while True:
        across = np.where(nbr >= 0, side[nbr], -2)
        ear = (side >= 0) & ((across != side[:, None]).sum(1) >= 2)
        if not ear.any():
            return side
        side[ear] = -1


def neighbours(F):
    """(nF, 3) the triangle across each edge (k, k+1) of each triangle, -1 on a border."""
    he = np.stack([F, np.roll(F, -1, axis=1)], -1).reshape(-1, 2).astype(np.int64)
    n = he.max() + 1
    key, twin = he[:, 0] * n + he[:, 1], he[:, 1] * n + he[:, 0]
    order = np.argsort(key)
    pos = np.minimum(np.searchsorted(key[order], twin), len(key) - 1)
    found = key[order][pos] == twin
    return np.where(found, order[pos] // 3, -1).reshape(-1, 3)


# === Points on the mesh ===

def closest_on_mesh(V, F, X):
    """For each point of X, the triangle of its closest point on (V, F) and the
    barycentric coordinates of that closest point, (M,) and (M, 3)."""
    import open3d as o3d
    mesh = o3d.t.geometry.TriangleMesh()
    mesh.vertex.positions = o3d.core.Tensor(V.astype(np.float32))
    mesh.triangle.indices = o3d.core.Tensor(F.astype(np.int32))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)
    hit = scene.compute_closest_points(o3d.core.Tensor(X.astype(np.float32)))
    w = hit["primitive_uvs"].numpy().astype(float)
    bary = np.clip(np.c_[1 - w.sum(1), w], 0, 1)  # float32 round-off
    return hit["primitive_ids"].numpy().astype(int), bary / bary.sum(1, keepdims=True)


def submesh(F, keep):
    """(F', vertices) the triangles F[keep] renumbered on their own vertices, and the
    index of these vertices in the full mesh."""
    vertices, Fk = np.unique(F[keep], return_inverse=True)
    return Fk.reshape(-1, 3), vertices
