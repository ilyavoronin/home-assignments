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
    check_baseline, Correspondences,
)

MAX_REPROJECTION_ERROR = 7.0
MIN_ANGLE = 3.0
MIN_DEPTH = 0.1
MIN_DIST = 0.1
MAX_HOMOGRAPHY = 0.7


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

    for i in range(-30, 30):
        other_frame_num = frame_num + i
        if other_frame_num < 0 or other_frame_num >= len(view_mat):
            break
        if view_mat[other_frame_num] is None:
            continue
        if check_baseline(view_mat[other_frame_num], view_mat[frame_num], MIN_DIST):
            cor = build_correspondences(corner_storage[other_frame_num], corner_storage[frame_num])
            if len(cor.ids) < 3:
                continue
            new_points, ids, _ = triangulate_correspondences(cor,
                                                             view_mat[other_frame_num],
                                                             view_mat[frame_num],
                                                             intrinsic_mat,
                                                             TriangulationParameters(2.0, 1.0, MIN_DIST)
                                                             )
            pcbuilder.add_points(ids, new_points)


def init_cloud(pcbuilder, corner_storage, intrinsic_mat, known_view1, known_view2):
    new_points = []
    ids = []
    max_error = MAX_REPROJECTION_ERROR
    min_angle = MIN_ANGLE
    correspondences = build_correspondences(corner_storage[known_view1[0]], corner_storage[known_view2[0]])
    while len(new_points) < 10 and min_angle > 0 and max_error < 10:
        new_points, ids, _ = triangulate_correspondences(correspondences, pose_to_view_mat3x4(known_view1[1]),
                                                         pose_to_view_mat3x4(known_view2[1]), intrinsic_mat,
                                                         TriangulationParameters(max_error, min_angle, MIN_DEPTH)
                                                         )
        min_angle -= 0.3
        max_error += 0.5
    if len(new_points) < 3:
        raise Exception("Failed to initialize point cloud")

    pcbuilder.add_points(ids, new_points)


def track_camera(corner_storage, intrinsic_mat, known_view1, known_view2):
    init_frame_num1 = known_view1[0]
    view1 = known_view1[1]
    init_frame_num2 = known_view2[0]
    view2 = known_view2[1]
    nframes = len(corner_storage)
    pcbuilder = PointCloudBuilder()
    view_mat = [None] * nframes

    view_mat[init_frame_num1] = pose_to_view_mat3x4(view1)
    view_mat[init_frame_num2] = pose_to_view_mat3x4(view2)

    init_cloud(pcbuilder, corner_storage, intrinsic_mat, known_view1, known_view2)

    for i in range(init_frame_num1 - 1, -1, -1):
        handle_frame(i, corner_storage, pcbuilder, view_mat, intrinsic_mat, 1)

    for i in range(init_frame_num1 + 1, nframes):
        if i == init_frame_num2:
            continue
        handle_frame(i, corner_storage, pcbuilder, view_mat, intrinsic_mat, -1)
    return view_mat, pcbuilder


def find_best_init_frames(corner_storage, intrinsic_mat, nbest):

    frame_pairs = []

    for frame_num1 in range(len(corner_storage)):
        for frame_num2 in range(frame_num1 + 1, min(frame_num1 + 40, len(corner_storage))):
            cor = build_correspondences(corner_storage[frame_num1], corner_storage[frame_num2])
            if len(cor.ids) < 20:
                continue

            ret, mask = cv2.findEssentialMat(
                cor.points_1,
                cor.points_2,
                intrinsic_mat,
                cv2.RANSAC,
                0.99,
                1.0
            )

            if ret is None:
                continue

            mask = (mask == 1).flatten()
            cor = Correspondences(cor.ids[mask], cor.points_1[mask], cor.points_2[mask])

            _, mask1 = cv2.findHomography(cor.points_1, cor.points_2, method=cv2.RANSAC)
            if np.count_nonzero(mask1) / np.count_nonzero(mask) > MAX_HOMOGRAPHY:
                continue

            R1, R2, t = cv2.decomposeEssentialMat(ret)

            rotations = [R1, R2]
            translations = [t, -t]

            for R in rotations:
                for t in translations:
                    a, ids, b = triangulate_correspondences(
                        cor,
                        eye3x4(),
                        np.hstack((R, t)),
                        intrinsic_mat,
                        TriangulationParameters(MAX_REPROJECTION_ERROR, MIN_ANGLE, MIN_DEPTH)
                    )

                    frame_pairs.append((-len(ids), (frame_num1, frame_num2), np.hstack((R, t))))

    frame_pairs = sorted(frame_pairs, key=lambda x: x[0])[:nbest]

    final_res = []
    for p in frame_pairs:
        final_res.append(((p[1][0], view_mat3x4_to_pose(eye3x4())), (p[1][1], view_mat3x4_to_pose(p[2]))))

    return final_res


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )
    best_frames = None
    if known_view_1 is None:
        best_frames = find_best_init_frames(corner_storage, intrinsic_mat, 5)
    else:
        best_frames = [(known_view_1, known_view_2)]

    view_mats, point_cloud_builder = None, None
    for init_frames in best_frames:
        view_mats, point_cloud_builder = track_camera(corner_storage, intrinsic_mat, init_frames[0], init_frames[1])
        break

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
