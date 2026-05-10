import itertools
import math
import os
import sys
import env
import src.utils.engine
import src.utils.utils as utils
import numpy as np
import networkx as nx
from scipy.optimize import least_squares, linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy import sparse

from src.calibrate_camera import *
from src.image_rectification import *
from src.threeD_reconstruction import *


def _coerce_xyz_vector(point_like):
    """Return a length-3 float64 vector (for ``np.asarray([...])`` stacks of per-track points)."""
    vector = np.asarray(point_like, dtype=np.float64).reshape(-1)
    if vector.size >= 3:
        return vector[:3].astype(np.float64, copy=True)
    result = np.zeros(3, dtype=np.float64)
    result[: vector.size] = vector
    return result


def initial_world_point_along_viewing_ray(projection_matrix, uv, depth_scale=1.0):
    """
    World-frame point on the ray from the camera center through pixel uv.
    depth_scale multiplies a focal-length-based step so the point sits in front of the camera.
    """
    P = np.asarray(projection_matrix, dtype=np.float64)
    uv = np.asarray(uv, dtype=np.float64).ravel()
    _, intrinsic_matrix, rotation_world_to_cam, translation_column, *_ = cv2.decomposeProjectionMatrix(
        P
    )
    intrinsic_matrix = np.asarray(intrinsic_matrix, dtype=np.float64)
    rotation_world_to_cam = np.asarray(rotation_world_to_cam, dtype=np.float64)
    intrinsics_3x3 = intrinsic_matrix[:3, :3]
    # OpenCV may return 3x3 or 4x4 rotation; we need a proper 3x3 world-to-camera rotation.
    rotation_3x3 = rotation_world_to_cam[:3, :3]

    translation_homogeneous = np.asarray(translation_column, dtype=np.float64).reshape(-1)
    if translation_homogeneous.size >= 4:
        translation_vector = translation_homogeneous[:3] / (translation_homogeneous[3] + 1e-12)
    else:
        translation_vector = translation_homogeneous[:3]
    translation_vector = np.asarray(translation_vector, dtype=np.float64).reshape(3)

    camera_center_world = (-rotation_3x3.T @ translation_vector.reshape(3, 1)).flatten()

    pixel_homogeneous = np.array([[uv[0]], [uv[1]], [1.0]], dtype=np.float64)
    bearing_in_camera = np.linalg.inv(intrinsics_3x3) @ pixel_homogeneous
    bearing_in_camera = bearing_in_camera / (np.linalg.norm(bearing_in_camera) + 1e-12)
    ray_direction_world = (rotation_3x3.T @ bearing_in_camera).flatten()

    focal_length_estimate = 0.5 * (abs(intrinsics_3x3[0, 0]) + abs(intrinsics_3x3[1, 1]))
    distance_along_ray = depth_scale * max(focal_length_estimate, 1.0)
    initial_point_world = camera_center_world + distance_along_ray * ray_direction_world

    if not np.all(np.isfinite(initial_point_world)):
        return _coerce_xyz_vector([uv[0], uv[1], 1.0])
    return _coerce_xyz_vector(initial_point_world)


def _projection_matrix_from_pose(camera_intrinsics, rotation_matrix, translation_vector):
    return camera_intrinsics @ np.hstack(
        [rotation_matrix, np.asarray(translation_vector, dtype=np.float64).reshape(3, 1)]
    )


def _rotation_translation_world_to_camera_from_projection_matrix(
    camera_intrinsics, projection_matrix
):
    """World-to-camera ``R, t`` with ``X_cam = R @ X_world + t``, matching BA decomposition."""
    inverse_intrinsics = np.linalg.inv(np.asarray(camera_intrinsics, dtype=np.float64))
    extrinsics = inverse_intrinsics @ np.asarray(projection_matrix, dtype=np.float64)
    rotation_linear = extrinsics[:, :3].astype(np.float64)
    translation_vector = extrinsics[:, 3].astype(np.float64)
    U_matrix, _, vt_matrix = np.linalg.svd(rotation_linear)
    orthogonal_rotation = U_matrix @ vt_matrix
    if np.linalg.det(orthogonal_rotation) < 0:
        U_matrix = U_matrix.copy()
        U_matrix[:, -1] *= -1.0
        orthogonal_rotation = U_matrix @ vt_matrix
    return orthogonal_rotation, translation_vector


def _decompose_projection_matrices_to_extrinsics(camera_intrinsics, projection_matrices):
    rotation_matrices = []
    translation_vectors = []
    for projection_matrix in projection_matrices:
        orthogonal_rotation, translation_vector = (
            _rotation_translation_world_to_camera_from_projection_matrix(
                camera_intrinsics, projection_matrix
            )
        )
        rotation_matrices.append(orthogonal_rotation)
        translation_vectors.append(translation_vector)
    return rotation_matrices, translation_vectors


def camera_centers_world_from_projection_matrices(camera_intrinsics, projection_matrices):
    """Camera centers ``C = -R.T @ t`` for each ``P = K [R | t]`` (same convention as BA)."""
    rotation_matrices, translation_vectors = _decompose_projection_matrices_to_extrinsics(
        camera_intrinsics, projection_matrices
    )
    centers = np.empty((len(projection_matrices), 3), dtype=np.float64)
    for index, (rotation_matrix, translation_vector) in enumerate(
        zip(rotation_matrices, translation_vectors)
    ):
        centers[index] = (-rotation_matrix.T @ translation_vector.reshape(3, 1)).flatten()
    return centers


def print_sparse_reconstruction_diagnostics(
    diagnostic_label,
    points_3d,
    camera_intrinsics,
    projection_matrices,
):
    """
    Log numeric shape of the point cloud and camera centers (helps distinguish true 1D/planar
    collapse from visualization quirks). Safe to call on initial guesses or BA-refined results.
    """
    points_xyz = np.asarray(points_3d, dtype=np.float64).reshape(-1, 3)
    valid_rows = np.all(np.isfinite(points_xyz), axis=1)
    points_xyz = points_xyz[valid_rows]
    num_points = points_xyz.shape[0]
    if num_points < 2:
        print(f"[diagnostics:{diagnostic_label}] too few finite points ({num_points}).")
        return
    centered = points_xyz - np.mean(points_xyz, axis=0, keepdims=True)
    singular_values_1d = np.linalg.svd(centered, compute_uv=False)
    singular_values = np.zeros(3, dtype=np.float64)
    singular_values[: min(3, len(singular_values_1d))] = singular_values_1d[:3]
    rotation_list, translation_list = _decompose_projection_matrices_to_extrinsics(
        camera_intrinsics, projection_matrices
    )
    camera_centers_world = []
    for rotation_matrix, translation_vector in zip(rotation_list, translation_list):
        center = (-rotation_matrix.T @ translation_vector.reshape(3, 1)).flatten()
        camera_centers_world.append(center)
    camera_centers_world = np.asarray(camera_centers_world, dtype=np.float64)
    pairwise_offsets = (
        camera_centers_world[:, np.newaxis, :] - camera_centers_world[np.newaxis, :, :]
    )
    center_spread = float(np.max(np.linalg.norm(pairwise_offsets, axis=2)))
    print(
        f"[diagnostics:{diagnostic_label}] points={num_points} | "
        f"covariance SVD s={singular_values} | "
        f"s2/s1={singular_values[1]/(singular_values[0]+1e-12):.4g} "
        f"s3/s2={singular_values[2]/(singular_values[1]+1e-12):.4g} | "
        f"max camera-center pairwise distance={center_spread:.6g}"
    )


def reprojection_error(
    points_3d,
    rotation_matrices,
    translation_vectors,
    camera_intrinsics,
    matches,
    z_floor=1e-10,
    large_error=1e6,
):
    """
    Stacked reprojection residuals for bundle adjustment: one (rx, ry) pair per observation.
    Sum of squares of this vector is the total BA objective (up to ordering).
    """
    points_3d = np.asarray(points_3d, dtype=np.float64).reshape(-1, 3)
    camera_intrinsics = np.asarray(camera_intrinsics, dtype=np.float64)
    residual_components = []
    for point_index, track in enumerate(matches):
        point_homogeneous = np.append(points_3d[point_index], 1.0)
        for image_index, true_2d_pt in track:
            projection_matrix = _projection_matrix_from_pose(
                camera_intrinsics,
                rotation_matrices[image_index],
                translation_vectors[image_index],
            )
            homogeneous_2d = projection_matrix @ point_homogeneous
            z_depth = homogeneous_2d[2]
            observed_xy = np.asarray(true_2d_pt, dtype=np.float64).ravel()
            if not np.isfinite(z_depth) or abs(z_depth) < z_floor:
                residual_components.extend([large_error, large_error])
                continue
            projected_xy = homogeneous_2d[:2] / z_depth
            if not np.all(np.isfinite(projected_xy)):
                residual_components.extend([large_error, large_error])
                continue
            residual_2d = projected_xy - observed_xy
            residual_components.extend([residual_2d[0], residual_2d[1]])
    return np.asarray(residual_components, dtype=np.float64)


def _bundle_adjustment_jacobian_sparsity(matches, num_points, num_cameras, fix_first_camera_pose):
    """
    Sparse structure for d(residuals)/d(params): each observation depends only on
    its 3D point (3) and the camera that saw it (6 extrinsics), so finite-difference
    Jacobians stay tractable (see scipy least_squares jac_sparsity).
    """
    num_residuals = sum(len(track) * 2 for track in matches)
    if fix_first_camera_pose:
        num_camera_scalar_params = 6 * max(0, num_cameras - 1)
    else:
        num_camera_scalar_params = 6 * num_cameras
    num_params = 3 * num_points + num_camera_scalar_params

    sparsity_pattern = sparse.lil_matrix((num_residuals, num_params), dtype=np.float64)
    residual_row = 0
    for point_index, track in enumerate(matches):
        for image_index, _ in track:
            point_column_start = 3 * point_index
            for delta in range(3):
                sparsity_pattern[residual_row, point_column_start + delta] = 1.0
                sparsity_pattern[residual_row + 1, point_column_start + delta] = 1.0
            if fix_first_camera_pose:
                if image_index >= 1:
                    camera_column_start = 3 * num_points + 6 * (image_index - 1)
                    for delta in range(6):
                        sparsity_pattern[residual_row, camera_column_start + delta] = 1.0
                        sparsity_pattern[residual_row + 1, camera_column_start + delta] = 1.0
            else:
                camera_column_start = 3 * num_points + 6 * image_index
                for delta in range(6):
                    sparsity_pattern[residual_row, camera_column_start + delta] = 1.0
                    sparsity_pattern[residual_row + 1, camera_column_start + delta] = 1.0
            residual_row += 2
    return sparsity_pattern.tocsr()


def bundle_adjustment(
    camera_intrinsics,
    projection_matrices_initial,
    matches,
    points_3d_initial,
    fix_first_camera_pose=True,
    least_squares_options=None,
):
    """
    Jointly refine all 3D points and camera poses (extrinsics) by minimizing
    sum over all points and all observations of squared reprojection error.

    First camera is fixed to the identity pose (reference frame) unless all cameras
    are optimized by setting fix_first_camera_pose=False (gauge not fixed then).

    Uses ``jac_sparsity`` so SciPy does sparse finite differences (fast). Without this,
    dense Jacobians scale roughly with the number of parameters and become unusable.

    Args:
        camera_intrinsics: 3x3 K (fixed).
        projection_matrices_initial: list of 3x4 P matrices aligned with image indices.
        matches: list of tracks; matches[k] are observations of point k.
        points_3d_initial: array (N, 3), same N as len(matches).

    Returns:
        points_3d_refined (N, 3),
        projection_matrices_refined: list of 3x4 P,
        result: scipy OptimizeResult from least_squares.
    """
    points_3d_initial = np.asarray(points_3d_initial, dtype=np.float64).reshape(-1, 3)
    num_points = points_3d_initial.shape[0]
    num_cameras = len(projection_matrices_initial)
    if num_points != len(matches):
        raise ValueError(
            f"points_3d_initial has {num_points} rows but matches has {len(matches)} tracks."
        )

    rotation_matrices, translation_vectors = _decompose_projection_matrices_to_extrinsics(
        camera_intrinsics, projection_matrices_initial
    )

    def pack_parameters(point_xyz, rotation_list, translation_list):
        blocks = [point_xyz.ravel()]
        camera_range = range(1, num_cameras) if fix_first_camera_pose else range(0, num_cameras)
        for camera_index in camera_range:
            rvec, _ = cv2.Rodrigues(np.asarray(rotation_list[camera_index], dtype=np.float64))
            blocks.append(rvec.flatten())
            blocks.append(np.asarray(translation_list[camera_index], dtype=np.float64).ravel())
        return np.concatenate(blocks)

    def unpack_parameters(packed):
        point_xyz = packed[: 3 * num_points].reshape(num_points, 3)
        offset = 3 * num_points
        rotation_out = []
        translation_out = []
        camera_range = range(1, num_cameras) if fix_first_camera_pose else range(0, num_cameras)
        for _ in camera_range:
            rvec = packed[offset : offset + 3]
            offset += 3
            translation_vec = packed[offset : offset + 3]
            offset += 3
            rotation_matrix, _ = cv2.Rodrigues(rvec.reshape(3, 1))
            rotation_out.append(rotation_matrix)
            translation_out.append(translation_vec)
        if fix_first_camera_pose:
            rotation_all = [rotation_matrices[0]] + rotation_out
            translation_all = [translation_vectors[0]] + translation_out
        else:
            rotation_all = rotation_out
            translation_all = translation_out
        return point_xyz, rotation_all, translation_all

    initial_packed = pack_parameters(
        points_3d_initial, rotation_matrices, translation_vectors
    )

    def residuals(packed):
        point_xyz, rotation_all, translation_all = unpack_parameters(packed)
        return reprojection_error(
            point_xyz,
            rotation_all,
            translation_all,
            camera_intrinsics,
            matches,
        )

    jacobian_sparsity = _bundle_adjustment_jacobian_sparsity(
        matches, num_points, num_cameras, fix_first_camera_pose
    )

    default_options = {
        "method": "trf",
        "max_nfev": 100,
        "ftol": 1e-6,
        "xtol": 1e-6,
        "gtol": 1e-6,
        "jac_sparsity": jacobian_sparsity,
        "loss": "huber",
        "f_scale": 10.0,
    }
    if least_squares_options:
        default_options.update(least_squares_options)

    optimization_result = least_squares(
        residuals,
        initial_packed,
        **default_options,
    )

    refined_points, refined_rotations, refined_translations = unpack_parameters(
        optimization_result.x
    )
    projection_matrices_refined = [
        _projection_matrix_from_pose(camera_intrinsics, rotation_matrix, translation_vector)
        for rotation_matrix, translation_vector in zip(
            refined_rotations, refined_translations
        )
    ]
    return refined_points, projection_matrices_refined, optimization_result


def _per_observation_residual_norms_flat(residual_vector):
    """Map stacked (rx, ry, ...) from ``reprojection_error`` to one L2 norm per observation."""
    residual_vector = np.asarray(residual_vector, dtype=np.float64).ravel()
    num_observations = len(residual_vector) // 2
    norms = np.empty(num_observations, dtype=np.float64)
    for observation_index in range(num_observations):
        rx = residual_vector[2 * observation_index]
        ry = residual_vector[2 * observation_index + 1]
        norms[observation_index] = float(np.hypot(rx, ry))
    return norms


def _ransac_dropout_track_metrics(matches, observation_norms, threshold_px):
    """
    Per-track reprojection norms (same order as ``reprojection_error`` over matches).

    Returns:
        tracks_with_two_or_more_inliers: count of tracks having at least two observations
            with norm strictly below ``threshold_px``.
        sum_mean_norm_per_track: sum over tracks of (mean norm in that track) — lower is better fit.
        total_observations_below_threshold: count of individual observations below threshold.
    """
    cursor = 0
    tracks_with_two_or_more_inliers = 0
    sum_mean_norm_per_track = 0.0
    total_observations_below_threshold = 0
    for track in matches:
        slice_norms = observation_norms[cursor : cursor + len(track)]
        cursor += len(slice_norms)
        below_threshold_mask = slice_norms < threshold_px
        total_observations_below_threshold += int(np.sum(below_threshold_mask))
        if int(np.sum(below_threshold_mask)) >= 2:
            tracks_with_two_or_more_inliers += 1
        sum_mean_norm_per_track += float(np.mean(slice_norms))
    return (
        tracks_with_two_or_more_inliers,
        sum_mean_norm_per_track,
        total_observations_below_threshold,
    )


def bundle_adjustment_with_ransac_dropout(
    camera_intrinsics,
    projection_matrices_initial,
    matches,
    points_3d_initial,
    *,
    num_ransac_iterations=12,
    subset_fraction_low=0.22,
    subset_fraction_high=0.55,
    min_tracks_in_subset=40,
    outlier_reprojection_pixels=6.0,
    fix_first_camera_pose=True,
    least_squares_options=None,
    least_squares_options_subset=None,
    rng=None,
    run_final_bundle_adjustment=True,
):
    """
    RANSAC over subset bundle adjustments (matches are **never** filtered here):

    1. Repeatedly sample a subset of tracks and run ``bundle_adjustment`` on that subset only.
    2. With the fitted projection matrices, compute per-observation reprojection norms on
       **all** tracks using the **initial** 3D points.
    3. Rank hypotheses by this tuple (lexicographic, higher is better):

       - **Maximize** ``tracks_with_two_or_more_inliers``: number of tracks that have at least
         two observations with norm ``< outlier_reprojection_pixels``.
       - **Minimize** aggregate per-track error: use ``-sum_mean_norm_per_track`` where the sum is
         over tracks of (mean reprojection norm in that track).
       - **Maximize** ``total_observations_below_threshold`` as a last tie-break.

    4. Keep only the winning iteration's **projection matrices** (best cameras).
    5. If ``run_final_bundle_adjustment``, run **one** full ``bundle_adjustment`` on the original
       unfiltered ``matches`` and ``points_3d_initial``, starting from those cameras.

    Returns:
        Tuple ``(points_refined, projection_matrices_refined, optimization_result, stats_dict)``.
        If ``run_final_bundle_adjustment`` is False, the first and third entries are ``None``.
    """
    rng = rng if rng is not None else np.random.default_rng()
    points_3d_initial = np.asarray(points_3d_initial, dtype=np.float64).reshape(-1, 3)
    num_tracks = len(matches)
    if num_tracks == 0:
        stats = {"skipped": True, "reason": "no tracks"}
        return (
            np.zeros((0, 3), dtype=np.float64),
            list(projection_matrices_initial),
            None,
            stats,
        )

    total_observations = sum(len(track) for track in matches)
    if total_observations == 0:
        stats = {"skipped": True, "reason": "no observations"}
        return (
            np.zeros((0, 3), dtype=np.float64),
            list(projection_matrices_initial),
            None,
            stats,
        )

    subset_ba_options = dict(least_squares_options) if least_squares_options else {}
    if least_squares_options_subset is not None:
        subset_ba_options = {**subset_ba_options, **least_squares_options_subset}

    best_rank = (-1, float("-inf"), -1)
    best_projection_matrices = None
    best_tracks_two_plus_inliers = -1
    best_sum_mean_norm_per_track = np.inf
    best_total_obs_below = -1
    attempts_used = 0

    for _ in range(num_ransac_iterations):
        if num_tracks <= 2:
            subset_size = num_tracks
        else:
            low_bound = max(2, int(math.floor(subset_fraction_low * num_tracks)))
            high_bound = min(num_tracks - 1, max(low_bound, int(math.ceil(subset_fraction_high * num_tracks))))
            if high_bound < low_bound:
                low_bound, high_bound = high_bound, low_bound
            subset_size = int(rng.integers(low_bound, high_bound + 1))
            subset_size = max(subset_size, min(min_tracks_in_subset, num_tracks - 1))
            subset_size = min(subset_size, num_tracks - 1)
            subset_size = max(subset_size, 2)

        subset_track_indices = rng.choice(num_tracks, size=subset_size, replace=False)

        matches_subset = [matches[i] for i in subset_track_indices]
        points_subset = points_3d_initial[subset_track_indices].copy()

        try:
            _, projection_matrices_fit, _subset_result = bundle_adjustment(
                camera_intrinsics,
                projection_matrices_initial,
                matches_subset,
                points_subset,
                fix_first_camera_pose=fix_first_camera_pose,
                least_squares_options=subset_ba_options,
            )
        except Exception:
            continue

        attempts_used += 1
        rotation_matrices_fit, translation_vectors_fit = _decompose_projection_matrices_to_extrinsics(
            camera_intrinsics,
            projection_matrices_fit,
        )
        residual_vector = reprojection_error(
            points_3d_initial,
            rotation_matrices_fit,
            translation_vectors_fit,
            camera_intrinsics,
            matches,
        )
        observation_norms = _per_observation_residual_norms_flat(residual_vector)
        tracks_two_plus, sum_mean_track, total_below = _ransac_dropout_track_metrics(
            matches,
            observation_norms,
            outlier_reprojection_pixels,
        )
        candidate_rank = (tracks_two_plus, -sum_mean_track, total_below)
        if candidate_rank > best_rank:
            best_rank = candidate_rank
            best_projection_matrices = projection_matrices_fit
            best_tracks_two_plus_inliers = tracks_two_plus
            best_sum_mean_norm_per_track = sum_mean_track
            best_total_obs_below = total_below

    if best_projection_matrices is None:
        stats = {
            "ransac_success": False,
            "attempts_used": attempts_used,
            "num_tracks_in": num_tracks,
            "total_observations_in": total_observations,
            "note": "no successful subset BA; returning unpruned single BA",
        }
        return bundle_adjustment(
            camera_intrinsics,
            projection_matrices_initial,
            matches,
            points_3d_initial,
            fix_first_camera_pose=fix_first_camera_pose,
            least_squares_options=least_squares_options,
        ) + (stats,)

    stats = {
        "ransac_success": True,
        "attempts_used": attempts_used,
        "num_iterations_requested": num_ransac_iterations,
        "best_tracks_with_two_plus_inliers": best_tracks_two_plus_inliers,
        "best_sum_mean_norm_per_track_px": float(best_sum_mean_norm_per_track),
        "best_total_observations_below_threshold": best_total_obs_below,
        "outlier_reprojection_pixels": outlier_reprojection_pixels,
        "num_tracks_in": num_tracks,
        "total_observations_in": total_observations,
        "matches_filtered": False,
    }

    if not run_final_bundle_adjustment:
        return None, best_projection_matrices, None, stats

    final_points, final_projection_matrices, final_result = bundle_adjustment(
        camera_intrinsics,
        best_projection_matrices,
        matches,
        points_3d_initial,
        fix_first_camera_pose=fix_first_camera_pose,
        least_squares_options=least_squares_options,
    )
    return final_points, final_projection_matrices, final_result, stats


def _minimum_incident_lowe_ratio(subgraph, graph_node):
    """Smaller Lowe ratio is a stronger (more distinctive) match."""
    incident_ratios = []
    for _, _, edge_data in subgraph.edges(graph_node, data=True):
        incident_ratios.append(edge_data.get("lowe_ratio", 1.0))
    return min(incident_ratios) if incident_ratios else 1.0


def deduplicate_connected_component_track(connected_component_nodes, kp_desc_list, matches_graph):
    """
    Collapse to at most one keypoint per image from this connected component.

    If several keypoints from the same image appear in one CC, keep the one with the
    **best (smallest) minimum incident Lowe ratio** among edges inside this component—i.e.
    the correspondence that was most distinctive in its pairwise match. Tie-break with
    higher subgraph degree, then lower keypoint index.
    """
    subgraph = matches_graph.subgraph(connected_component_nodes)
    nodes_grouped_by_image = {}
    for graph_node in connected_component_nodes:
        image_index, keypoint_index = graph_node
        nodes_grouped_by_image.setdefault(image_index, []).append(graph_node)

    track_observations = []
    for image_index in sorted(nodes_grouped_by_image.keys()):
        candidate_nodes = nodes_grouped_by_image[image_index]
        chosen_node = min(
            candidate_nodes,
            key=lambda node: (
                _minimum_incident_lowe_ratio(subgraph, node),
                -subgraph.degree(node),
                node[1],
            ),
        )
        _, chosen_keypoint_index = chosen_node
        uv_point = kp_desc_list[image_index][0][chosen_keypoint_index].pt
        track_observations.append((image_index, uv_point))
    return track_observations


def _descriptor_matcher_for_descriptor_array(descriptor_array):
    """Return an OpenCV matcher suited to binary (uint8) vs floating descriptors."""
    if descriptor_array.dtype == np.uint8:
        return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
    search_params = dict(checks=50)
    return cv2.FlannBasedMatcher(index_params, search_params)


def _full_descriptor_distance_matrix(descriptor_rows_a, descriptor_rows_b):
    """Pairwise Euclidean distances for Hungarian costs (uint8 descriptors cast to float32)."""
    descriptor_rows_a = np.asarray(descriptor_rows_a)
    descriptor_rows_b = np.asarray(descriptor_rows_b)
    return cdist(
        descriptor_rows_a.astype(np.float32),
        descriptor_rows_b.astype(np.float32),
        metric="euclidean",
    )


def _hungarian_descriptor_matches_lowe_filtered(
    descriptor_rows_left,
    descriptor_rows_right,
    *,
    lowe_ratio_threshold=0.75,
    max_pair_distance=None,
):
    """
    One-to-one bipartite assignment minimizing total descriptor distance (Hungarian), then filter
    pairs by Lowe ratio (assignment distance vs second-best in the query row).

    Returns list of ``(query_index, train_index, lowe_ratio)``.
    """
    descriptor_rows_left = np.asarray(descriptor_rows_left)
    descriptor_rows_right = np.asarray(descriptor_rows_right)
    num_left = len(descriptor_rows_left)
    num_right = len(descriptor_rows_right)
    if num_left < 1 or num_right < 1:
        return []

    cost_matrix = _full_descriptor_distance_matrix(descriptor_rows_left, descriptor_rows_right)
    row_indices, column_indices = linear_sum_assignment(cost_matrix)

    accepted_triplets = []
    for row_index, column_index in zip(row_indices, column_indices):
        assignment_distance = float(cost_matrix[row_index, column_index])
        if max_pair_distance is not None and assignment_distance > max_pair_distance:
            continue
        row_costs = cost_matrix[row_index]
        if num_right < 2:
            lowe_ratio = 0.0
        else:
            alternative_mask = np.ones(num_right, dtype=bool)
            alternative_mask[column_index] = False
            second_best_distance = float(np.min(row_costs[alternative_mask]))
            lowe_ratio = assignment_distance / max(second_best_distance, 1e-12)
        if lowe_ratio >= lowe_ratio_threshold:
            continue
        accepted_triplets.append((row_index, column_index, lowe_ratio))
    return accepted_triplets


def make_dense_sift_detect_and_compute(
    step_pixels=10,
    keypoint_diameter_pixels=16.0,
    sift_constructor_kwargs=None,
):
    """
    Build a ``detect_and_compute`` callable compatible with ``find_correspondences_for_n_images``
    that evaluates **dense** SIFT descriptors on a regular grid (DSIFT-style): same 128-D float
    vectors as normal SIFT, so matchers and downstream graph / BA logic stay unchanged.

    Downstream code only needs ``(keypoints, descriptors)`` with ``KeyPoint.pt`` set — no API
    change required. **Cost**: many more descriptors → slower matching and larger graphs unless you
    increase ``step_pixels`` or subsample.

    Args:
        step_pixels: spacing between grid centers.
        keypoint_diameter_pixels: ``cv2.KeyPoint`` size (SIFT support region scale).
        sift_constructor_kwargs: forwarded to ``cv2.SIFT_create(**...)``.
    """
    sift_constructor_kwargs = sift_constructor_kwargs or {}
    sift_instance = cv2.SIFT_create(**sift_constructor_kwargs)

    def detect_and_compute(image, mask=None):
        if image.ndim == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image
        height, width = gray_image.shape[:2]
        mask_array = None
        if mask is not None:
            mask_array = np.asarray(mask)
            if mask_array.ndim == 3:
                mask_array = cv2.cvtColor(mask_array, cv2.COLOR_BGR2GRAY)
            mask_array = mask_array.astype(np.uint8)

        half_step = max(1, step_pixels // 2)
        grid_keypoints = []
        for y_pixel in range(half_step, height, step_pixels):
            for x_pixel in range(half_step, width, step_pixels):
                if mask_array is not None:
                    mask_y = min(y_pixel, mask_array.shape[0] - 1)
                    mask_x = min(x_pixel, mask_array.shape[1] - 1)
                    if mask_array[mask_y, mask_x] == 0:
                        continue
                grid_keypoints.append(
                    cv2.KeyPoint(
                        float(x_pixel),
                        float(y_pixel),
                        float(keypoint_diameter_pixels),
                    )
                )
        return sift_instance.compute(gray_image, grid_keypoints)

    return detect_and_compute


def find_correspondences_for_n_images(
    images,
    detect_and_compute=None,
    descriptor_matcher=None,
    match_consecutive_frames_only=True,
    pairwise_matching_mode="bipartite_hungarian",
    lowe_ratio_threshold=0.75,
    bipartite_max_pair_distance=None,
):
    """
    Multi-view feature matching; edges link keypoints across image pairs.
    Each connected component yields one track of (image_idx, xy); tracks are
    deduplicated to at most one observation per image (see deduplicate_connected_component_track).

    detect_and_compute: optional callable with the same contract as ``cv2.Feature2D.detectAndCompute``
    — ``(image, mask) -> (keypoints, descriptors)``. If omitted, uses ``cv2.SIFT_create().detectAndCompute``.
    For dense grid SIFT without changing downstream logic, pass
    ``detect_and_compute=make_dense_sift_detect_and_compute(step_pixels=...)``.

    descriptor_matcher: optional ``cv2.DescriptorMatcher``. If omitted, Hamming BF is used for uint8
    descriptors (ORB/BRISK-style); FLANN KD-tree is used for floating descriptors (e.g. SIFT).

    match_consecutive_frames_only: if True (default), match descriptors only between successive
    frames ``(0,1), (1,2), …`` — chain-shaped graph, typical for video sequences. If False, match
    every unordered pair (dense, quadratic cost). Consecutive-only removes long-range mismatches
    but still allows multiple keypoints per frame in one CC when parallel paths merge along the
    chain; deduplication inside ``deduplicate_connected_component_track`` remains necessary.

    pairwise_matching_mode: ``"knn_ratio"`` uses FLANN/BF kNN + Lowe ratio per query (many-to-one
    allowed). ``"bipartite_hungarian"`` builds a full distance matrix per pair and runs optimal
    one-to-one assignment (Hungarian), then applies the same Lowe ratio filter on accepted pairs.
    Bipartite is O(n m) memory per pair; best suited to moderate keypoint counts / consecutive pairs.

    bipartite_max_pair_distance: if set, drop Hungarian pairs whose assignment distance exceeds
    this (same units as descriptor L2 for SIFT).
    """
    if detect_and_compute is None:
        sift_feature_detector = cv2.SIFT_create(
            contrastThreshold=0.01,
            edgeThreshold=50,
        )
        detect_and_compute = sift_feature_detector.detectAndCompute

    kp_desc_list = [detect_and_compute(img, None) for img in images]

    if pairwise_matching_mode not in ("knn_ratio", "bipartite_hungarian"):
        raise ValueError(
            "pairwise_matching_mode must be 'knn_ratio' or 'bipartite_hungarian'."
        )

    if descriptor_matcher is None:
        sample_descriptors = next(
            (descriptors for _, descriptors in kp_desc_list if descriptors is not None and len(descriptors) >= 2),
            None,
        )
        if sample_descriptors is None:
            descriptor_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            descriptor_matcher = _descriptor_matcher_for_descriptor_array(sample_descriptors)

    matches_graph = nx.Graph()

    if match_consecutive_frames_only:
        frame_pair_indices = [
            (frame_index, frame_index + 1)
            for frame_index in range(len(images) - 1)
        ]
    else:
        frame_pair_indices = [
            (frame_index_i, frame_index_j)
            for frame_index_i in range(len(images))
            for frame_index_j in range(len(images))
            if frame_index_i != frame_index_j
        ]

    for image_idx_i, image_idx_j in frame_pair_indices:
        desc1 = kp_desc_list[image_idx_i][1]
        desc2 = kp_desc_list[image_idx_j][1]
        if desc1 is None or desc2 is None or len(desc1) < 1 or len(desc2) < 1:
            continue
        if pairwise_matching_mode != "bipartite_hungarian" and (
            len(desc1) < 2 or len(desc2) < 2
        ):
            continue

        if pairwise_matching_mode == "bipartite_hungarian":
            accepted_triplets = _hungarian_descriptor_matches_lowe_filtered(
                desc1,
                desc2,
                lowe_ratio_threshold=lowe_ratio_threshold,
                max_pair_distance=bipartite_max_pair_distance,
            )
            for query_idx, train_idx, lowe_ratio in accepted_triplets:
                node_image_i = (image_idx_i, query_idx)
                node_image_j = (image_idx_j, train_idx)
                if matches_graph.has_edge(node_image_i, node_image_j):
                    lowe_ratio = min(
                        lowe_ratio,
                        matches_graph[node_image_i][node_image_j]["lowe_ratio"],
                    )
                matches_graph.add_edge(node_image_i, node_image_j, lowe_ratio=lowe_ratio)
            continue

        matches_one_way = descriptor_matcher.knnMatch(desc1, desc2, k=2)
        for match_pair in matches_one_way:
            if len(match_pair) < 2:
                continue
            match_best, match_second = match_pair[0], match_pair[1]
            if match_best.distance >= lowe_ratio_threshold * match_second.distance:
                continue
            lowe_ratio = match_best.distance / max(match_second.distance, 1e-12)
            node_image_i = (image_idx_i, match_best.queryIdx)
            node_image_j = (image_idx_j, match_best.trainIdx)
            if matches_graph.has_edge(node_image_i, node_image_j):
                lowe_ratio = min(
                    lowe_ratio,
                    matches_graph[node_image_i][node_image_j]["lowe_ratio"],
                )
            matches_graph.add_edge(node_image_i, node_image_j, lowe_ratio=lowe_ratio)

    final_matches_list = []
    for cc in nx.connected_components(matches_graph):
        matched_xy = deduplicate_connected_component_track(cc, kp_desc_list, matches_graph)
        final_matches_list.append(matched_xy)

    return final_matches_list


def estimate_camera_projection_matrices(images, camera_matrix):
    """
    Incremental two-view estimates between consecutive frames. For pair ``(k-1, k)``,
    ``find_best_RT`` gives the pose of camera ``k`` when the **world frame matches camera k-1**:
    ``X_ck = R_local @ X_{c_{k-1}} + t_local`` (same as ``P = K[R|t]``).

    In the global frame fixed to camera 0: ``R_k = R_local @ R_{k-1}`` and
    ``t_k = R_local @ t_{k-1} + t_local`` (same order as ``cv2.composeRT(r_{k-1}, t_{k-1}, r_{local}, t_{local})``).
    """
    P_matrices_list = [get_identity_projection_matrix(camera_matrix)]
    R_global_prev = np.eye(3, dtype=np.float64)
    t_global_prev = np.zeros((3,), dtype=np.float64)
    for im1, im2 in zip(images[:-1], images[1:]):
        kp1, kp2, good_matches = find_matches(im1, im2)
        F, mask, pts1, pts2 = recover_fundamental_matrix(kp1, kp2, good_matches)
        inlier_pts1, inlier_pts2 = get_inliers(mask, pts1, pts2)
        E = compute_essential_matrix(camera_matrix, F)
        R_candidates, t_candidates = estimate_initial_RT(E)
        R_local, t_local = find_best_RT(R_candidates, t_candidates, inlier_pts1, inlier_pts2, camera_matrix)
        R_global_curr = R_local @ R_global_prev
        t_global_curr = R_local @ t_global_prev + t_local
        P_matrices_list.append(get_local_projection_matrix(camera_matrix, R_global_curr, t_global_curr).copy())
        R_global_prev = R_global_curr
        t_global_prev = t_global_curr
    return P_matrices_list


def get_initial_guess(
    single_point_matches,
    cameras_P_matrices,
    camera_intrinsics,
    camera_centers_world,
):
    """
    Triangulate an initial 3D point from two views.

    Among all pairs of observations in the track, prefer the pair whose camera centers have the
    **largest baseline** (stronger intersection geometry than fixed indices 0 and 2). Try pairs in
    descending baseline order until DLT gives a finite point with **positive depth** in both chosen
    cameras. Single-view tracks fall back to a point on the viewing ray.
    """
    camera_intrinsics = np.asarray(camera_intrinsics, dtype=np.float64)
    camera_centers_world = np.asarray(camera_centers_world, dtype=np.float64).reshape(-1, 3)
    num_observations = len(single_point_matches)
    if num_observations == 1:
        uv_only = np.asarray(single_point_matches[0][1], dtype=np.float64).ravel()
        image_idx_fallback = single_point_matches[0][0]
        return _coerce_xyz_vector(
            initial_world_point_along_viewing_ray(cameras_P_matrices[image_idx_fallback], uv_only)
        )

    pair_infos = []
    for observation_index_a in range(num_observations):
        image_idx_a = single_point_matches[observation_index_a][0]
        for observation_index_b in range(observation_index_a + 1, num_observations):
            image_idx_b = single_point_matches[observation_index_b][0]
            if image_idx_a == image_idx_b:
                continue
            baseline_length = float(
                np.linalg.norm(
                    camera_centers_world[image_idx_a] - camera_centers_world[image_idx_b]
                )
            )
            pair_infos.append(
                (baseline_length, observation_index_a, observation_index_b)
            )
    pair_infos.sort(key=lambda item: item[0], reverse=True)

    depth_floor = 1e-6
    for _baseline_length, observation_index_a, observation_index_b in pair_infos:
        image_idx_0 = single_point_matches[observation_index_a][0]
        image_idx_1 = single_point_matches[observation_index_b][0]
        uv_0 = np.asarray(single_point_matches[observation_index_a][1], dtype=np.float64).ravel()
        uv_1 = np.asarray(single_point_matches[observation_index_b][1], dtype=np.float64).ravel()
        pts4d = cv2.triangulatePoints(
            cameras_P_matrices[image_idx_0],
            cameras_P_matrices[image_idx_1],
            uv_0.reshape(2, 1),
            uv_1.reshape(2, 1),
        )
        w = pts4d[3, 0]
        if abs(w) < 1e-12:
            continue
        point_candidate = (pts4d[:3, 0] / w).astype(np.float64)
        if not np.all(np.isfinite(point_candidate)):
            continue
        rotation_a, translation_a = _rotation_translation_world_to_camera_from_projection_matrix(
            camera_intrinsics, cameras_P_matrices[image_idx_0]
        )
        rotation_b, translation_b = _rotation_translation_world_to_camera_from_projection_matrix(
            camera_intrinsics, cameras_P_matrices[image_idx_1]
        )
        depth_a = float((rotation_a @ point_candidate + translation_a)[2])
        depth_b = float((rotation_b @ point_candidate + translation_b)[2])
        if depth_a > depth_floor and depth_b > depth_floor:
            return _coerce_xyz_vector(point_candidate)

    uv_fallback = np.asarray(single_point_matches[0][1], dtype=np.float64).ravel()
    image_idx_fallback = single_point_matches[0][0]
    return _coerce_xyz_vector(
        initial_world_point_along_viewing_ray(cameras_P_matrices[image_idx_fallback], uv_fallback)
    )
        
        
def optimize_single_point_3d_location(
    single_point_matches,
    cameras_P_matrices,
    camera_intrinsics,
    camera_centers_world,
):
    def _reprojection_residuals(point_3d):
        residual_components = []
        z_floor = 1e-10
        large_error = 1e6
        for image_idx, true_2d_pt in single_point_matches:
            P = cameras_P_matrices[image_idx]
            homogenous_2d_pt = P @ np.hstack([point_3d, 1])
            z = homogenous_2d_pt[2]
            observed_xy = np.asarray(true_2d_pt, dtype=np.float64).ravel()
            if not np.isfinite(z) or abs(z) < z_floor:
                residual_components.extend([large_error, large_error])
                continue
            projected_xy = homogenous_2d_pt[:2] / z
            if not np.all(np.isfinite(projected_xy)):
                residual_components.extend([large_error, large_error])
                continue
            residual_2d = projected_xy - observed_xy
            residual_components.extend([residual_2d[0], residual_2d[1]])
        return np.asarray(residual_components, dtype=np.float64)

    point_3d_initial = get_initial_guess(
        single_point_matches,
        cameras_P_matrices,
        camera_intrinsics,
        camera_centers_world,
    )
    initial_residuals = _reprojection_residuals(point_3d_initial)
    if not np.all(np.isfinite(initial_residuals)):
        uv_fallback = np.asarray(single_point_matches[0][1], dtype=np.float64).ravel()
        image_idx_fallback = single_point_matches[0][0]
        point_3d_initial = initial_world_point_along_viewing_ray(
            cameras_P_matrices[image_idx_fallback], uv_fallback
        )
    result = least_squares(
        _reprojection_residuals,
        point_3d_initial,
        method="lm",
    )
    # cost = (1/2) * sum(residual**2); sum of squares = sum over views of ||e_i||^2 in pixel^2
    sum_squared_reprojection = 2.0 * result.cost
    mean_squared_error_per_view = sum_squared_reprojection / len(single_point_matches)
    # RMS in pixels (same rough scale as the old mean L2 norm per view, not pixel^2)
    rms_reprojection_error = float(np.sqrt(mean_squared_error_per_view))
    return result.x, rms_reprojection_error


def optimize_3d_location_for_matched_points(matches, cameras_P_matrices, camera_intrinsics):
    """
    Refine each track with optimize_single_point_3d_location (same order as matches).
    Returns one 3D point per track for bundle-adjustment warm start — no loss filtering.
    """
    camera_centers_world = camera_centers_world_from_projection_matrices(
        camera_intrinsics, cameras_P_matrices
    )
    collected_points_3d = []
    for match in matches:
        point_3d, _ = optimize_single_point_3d_location(
            match, cameras_P_matrices, camera_intrinsics, camera_centers_world
        )
        collected_points_3d.append(point_3d)
    return collected_points_3d
    

def main():
    if not os.path.exists(env.p8.output):
        os.makedirs(env.p8.output)
    chessboard_size = (16, 10)  # columns, rows
    images_folder = env.p8.statue_images
    chessboard_path = env.p7.chessboard

    camera_matrix, _ = calibrate_camera_from_chessboard(chessboard_path, chessboard_size)

    # # USE THIS CAMERA MATRIX FOR THE REST OF THE PIPELINE
    # camera_matrix = np.eye(3)
    # focal_length = 719.5459
    # camera_matrix[0,0] = focal_length
    # camera_matrix[1,1] = focal_length

    image_files = sorted([
        f for f in os.listdir(images_folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    images = [utils.load_image(os.path.join(images_folder, image_file)) for image_file in image_files]
    P_matrices = estimate_camera_projection_matrices(images, camera_matrix)
    print(P_matrices)
    matches = find_correspondences_for_n_images(images)
    print(matches)

    # Full FLANN tracks can be huge; BA cost grows ~linearly with tracks × views.
    # Set to None to keep every track (can take a long time).
    ba_max_tracks = 8000
    if ba_max_tracks is not None and len(matches) > ba_max_tracks:
        rng = np.random.default_rng(0)
        subset_indices = rng.choice(len(matches), size=ba_max_tracks, replace=False)
        subset_indices = np.sort(subset_indices)
        matches = [matches[i] for i in subset_indices]
        print(f"Subsampled tracks for BA: using {len(matches)} / original count.")

    camera_centers_world = camera_centers_world_from_projection_matrices(
        camera_matrix, P_matrices
    )
    initial_points_3d = np.asarray(
        [
            get_initial_guess(track, P_matrices, camera_matrix, camera_centers_world)
            for track in matches
        ],
        dtype=np.float64,
    )
    print_sparse_reconstruction_diagnostics(
        "initial triangulation / guesses", initial_points_3d, camera_matrix, P_matrices
    )
    points_refined, _projection_matrices_refined, bundle_result, ba_ransac_stats = (
        bundle_adjustment_with_ransac_dropout(
            camera_matrix,
            P_matrices,
            matches,
            initial_points_3d,
            num_ransac_iterations=10,
            subset_fraction_low=0.2,
            subset_fraction_high=0.48,
            min_tracks_in_subset=min(40, max(3, len(matches) // 2)),
            outlier_reprojection_pixels=6.0,
            rng=np.random.default_rng(0),
            least_squares_options_subset={"max_nfev": 80},
        )
    )
    print(f"BA RANSAC dropout: {ba_ransac_stats}")
    if bundle_result is None:
        print("Bundle adjustment did not return a result (run_final_bundle_adjustment=False or empty run).")
        return
    sum_squared_total = 2.0 * bundle_result.cost
    num_residuals = len(bundle_result.fun)
    rms_all_observations = float(np.sqrt(sum_squared_total / max(num_residuals, 1)))
    print(
        f"Bundle adjustment: sum squared reprojection = {sum_squared_total:.6g}, "
        f"RMS per scalar residual ≈ {rms_all_observations:.4g} px"
    )
    print_sparse_reconstruction_diagnostics(
        "after BA", points_refined, camera_matrix, _projection_matrices_refined
    )
    show_points_matplotlib(points_refined)


if __name__ == '__main__':
    main()

        
        
