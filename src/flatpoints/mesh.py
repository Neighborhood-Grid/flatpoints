"""Meshes of flatpoints: surface reconstruction, hemisphere cut, points on the mesh.

A closed surface (two faces) is reconstructed from the cloud by screened Poisson (open3d),
then cut in two topological disks across the median of the lateral axis. An open surface
(single face, e.g. a terrain) is triangulated in its two non-lateral coordinates.
"""
import heapq

import numpy as np
from scipy.spatial import Delaunay

POISSON_DEPTH = 8  # octree depth of the Poisson reconstruction
KNN_NORMALS = 16   # neighbours used to estimate and orient the normals
SLIVER = 4.0       # planar mesh: boundary triangles longer than SLIVER x median edge go


# === Reconstruction ===

def poisson_mesh(points, depth=POISSON_DEPTH):
    """(V, F) closed mesh of the surface sampled by `points`: its largest piece."""
    import open3d as o3d
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(KNN_NORMALS))
    cloud.orient_normals_consistent_tangent_plane(KNN_NORMALS)
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cloud, depth=depth)
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    piece, sizes, _ = mesh.cluster_connected_triangles()
    mesh.remove_triangles_by_mask(np.asarray(piece) != np.argmax(sizes))
    mesh.remove_unreferenced_vertices()
    return np.asarray(mesh.vertices), np.asarray(mesh.triangles)


def planar_mesh(points, ab):
    """(points, F, side): Delaunay triangulation of the points in their 2D coordinates
    `ab`, and the disk it keeps (side 0) without the long thin triangles along the convex
    hull (side -1)."""
    F = Delaunay(ab).simplices
    centre = ab[F].mean(1)
    d = np.linalg.norm(centre - np.median(ab, 0), axis=1)
    edge = np.linalg.norm(ab[F] - ab[F[:, [1, 2, 0]]], axis=2).max(1)
    side = np.full(len(F), -1)
    side[np.argmin(d)] = 0
    nbr = neighbours(F)
    side = grow(F, nbr, side, [d, d], [edge <= SLIVER * np.median(edge), np.zeros(len(F), bool)])
    return points, F, drop_ears(nbr, side)


# === Hemisphere cut ===

def hemisphere_cut(V, F, lateral, threshold):
    """Split the closed mesh (V, F) in two disks across `threshold` on the lateral axis.

    Each face grows from its extreme triangle as a sweep along the lateral axis, first on
    its own side of the threshold, then both take what remains, closest to the threshold
    first. The growth keeps each face a disk (see `grow`), whatever the genus of the mesh:
    triangles of a handle, that would close a loop, stay out of both faces.

    Returns:
        (F,) 0 (left), 1 (right) or -1 (out) for each triangle.
    """
    x = V[F, lateral].mean(1)
    nbr = neighbours(F)
    side = np.full(len(F), -1)
    side[np.argmin(x)], side[np.argmax(x)] = 0, 1
    right = x > threshold
    side = grow(F, nbr, side, [x, -x], [~right, right])
    d = np.abs(x - threshold)
    side = grow(F, nbr, side, [d, d], [side < 0, side < 0])
    return drop_ears(nbr, side)


def grow(F, nbr, side, keys, allowed):
    """Grow the faces 0 and 1 of the mesh, keeping each one a topological disk.

    A free triangle joins face s if it shares two edges with it, or one edge and its third
    vertex is not yet in the face (otherwise it would pinch the face or enclose a hole).
    Candidates are taken in increasing keys[s], both faces in turn from a single queue.

    Args:
        F: (nF, 3) triangles; nbr: (nF, 3) their neighbours (see `neighbours`).
        side: (nF,) current face of each triangle (-1: free); each face must be a disk.
        keys: two (nF,) priorities, one per face.
        allowed: two (nF,) masks, the triangles each face may take.

    Returns:
        (nF,) the new face of each triangle.
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
