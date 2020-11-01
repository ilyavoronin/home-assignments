__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import cv2
import numpy as np

from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import (
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,
    eye3x4,
    triangulate_correspondences,
    TriangulationParameters,
    build_correspondences,
    pose_to_view_mat3x4,
    rodrigues_and_translation_to_view_mat3x4,
    check_baseline,
)

MAX_REPROJECTION_ERROR = 2.0
MIN_ANGLE = 1.0
MIN_DEPTH = 0
MIN_DIST = 0.1

def handle_frame(
        frame_num,
        corner_storage,
        pcbuilder,
        view_mat,
        intrinsic_mat,
        direction
):
    print("Processing frame ", frame_num)
    corners = corner_storage[frame_num]

    _, corner_indexes, points_indexes = \
        np.intersect1d(corners.ids, pcbuilder.ids, assume_unique=True, return_indices=True)

    corners = corners.points[corner_indexes]
    points = pcbuilder.points[points_indexes]

    x, r, t, inliers = \
        cv2.solvePnPRansac(points, corners, intrinsic_mat,
                           None, reprojectionError=MAX_REPROJECTION_ERROR, flags=cv2.SOLVEPNP_EPNP)

    good_corners = corners[inliers]
    good_points = points[inliers]
    x, r, t = \
        cv2.solvePnP(good_points, good_corners, intrinsic_mat, None, r, t,
                     useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)

    print("Number of inliers: ", len(good_points))
    print("Total points in cloud ", len(pcbuilder.ids))

    new_views = rodrigues_and_translation_to_view_mat3x4(r, t)

    if new_views is not None:
        view_mat[frame_num] = new_views
    else:
        view_mat[frame_num] = view_mat[frame_num + direction]

    for i in range(20):
        other_frame_num = frame_num + i * direction
        if check_baseline(view_mat[other_frame_num], view_mat[frame_num], MIN_DIST):
            cor = build_correspondences(corner_storage[other_frame_num], corner_storage[frame_num], ids_to_remove=pcbuilder.ids)
            if len(cor) == 0:
                continue
            new_points, ids, _ = triangulate_correspondences(cor,
                                                             view_mat[other_frame_num],
                                                             view_mat[frame_num],
                                                             intrinsic_mat,
                                                             TriangulationParameters(1.0, MIN_ANGLE, MIN_DIST)
                                                             )
            pcbuilder.add_points(ids, new_points)


def init_cloud(pcbuilder, corner_storage, intrinsic_mat, known_view1, known_view2):
    new_points = []
    ids = []
    max_error = MAX_REPROJECTION_ERROR
    min_angle = MIN_ANGLE
    correspondences = build_correspondences(corner_storage[known_view1[0]], corner_storage[known_view2[0]])
    while len(new_points) < 10:
        new_points, ids, _ = triangulate_correspondences(correspondences, pose_to_view_mat3x4(known_view1[1]),
                                                         pose_to_view_mat3x4(known_view2[1]), intrinsic_mat,
                                                         TriangulationParameters(max_error, min_angle, MIN_DEPTH)
                                                         )
        min_angle -= 0.3
        max_error += 0.5
    pcbuilder.add_points(ids, new_points)


def track_camera(corner_storage, intrinsic_mat, known_view1, known_view2):
    init_frame_num1 = known_view1[0]
    view1 = known_view1[1]
    init_frame_num2 = known_view2[0]
    view2 = known_view2[1]
    nframes = len(corner_storage)
    pcbuilder = PointCloudBuilder()
    view_mat = [eye3x4()] * nframes

    view_mat[init_frame_num1] = pose_to_view_mat3x4(view1)
    view_mat[init_frame_num2] = pose_to_view_mat3x4(view2)

    init_cloud(pcbuilder, corner_storage, intrinsic_mat, known_view1, known_view2)

    for i in range(init_frame_num1 - 1, -1, -1):
        handle_frame(i, corner_storage, pcbuilder, view_mat, intrinsic_mat, 1)

    for i in range(init_frame_num1 + 1, nframes):
        handle_frame(i, corner_storage, pcbuilder, view_mat, intrinsic_mat, -1)
    return view_mat, pcbuilder


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    if known_view_1 is None or known_view_2 is None:
        raise NotImplementedError()

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    view_mats, point_cloud_builder = track_camera(corner_storage, intrinsic_mat, known_view_1, known_view_2)

    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
