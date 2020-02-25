#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import numpy as np
import cv2

from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import (
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,
    build_correspondences,
    pose_to_view_mat3x4,
    triangulate_correspondences,
    TriangulationParameters,
    rodrigues_and_translation_to_view_mat3x4,
    eye3x4
)


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

    poses = [None] * len(corner_storage)
    cloud_points = [None] * (corner_storage.max_corner_id() + 1)

    correspondences = build_correspondences(corner_storage[known_view_1[0]], corner_storage[known_view_2[0]])

    mat_0 = pose_to_view_mat3x4(known_view_1[1])
    mat_1 = pose_to_view_mat3x4(known_view_2[1])
    tri_params = TriangulationParameters(max_reprojection_error=1., min_triangulation_angle_deg=2., min_depth=0.1)
        
    print(f"Processing init frames {known_view_1[0]} and {known_view_2[0]}")
    points3d, correspondences_ids, median_cos = triangulate_correspondences(correspondences, mat_0, mat_1, intrinsic_mat, tri_params)
    
    cnt = 0
    cnt_all = 0
    for i in range(len(correspondences_ids)):
        id = correspondences_ids[i]
        if cloud_points[id] is None:
            cloud_points[id] = points3d[i]
            cnt += 1
            cnt_all += 1
    print(f"Cloud poinds added: {cnt}")
    print(f"Cloud poinds total: {cnt_all}")
    
    poses[0] = mat_0
    poses[1] = mat_1
   
    not_done = True
    while not_done:
        not_done = False
        for i in range(len(corner_storage)):
            if poses[i] is None:
                print(f"Processing frame {i}")
                cloud_points, poses[i] = _compute_pose(corner_storage[i], cloud_points, intrinsic_mat)
                if poses[i] is None:
                    continue

                for j in range(len(corner_storage)):
                    if poses[j] is not None and i != j:
                        correspondences = build_correspondences(corner_storage[i], corner_storage[j])

                        mat_0 = poses[i]
                        mat_1 = poses[j]
                        points3d, correspondences_ids, median_cos = triangulate_correspondences(correspondences, mat_0, mat_1, intrinsic_mat, tri_params)
                        print(f"{points3d.shape[0]} points triangulated")
                        cnt = 0
                        for k in range(len(correspondences_ids)):
                            id = correspondences_ids[k]
                            if cloud_points[id] is None:
                                cloud_points[id] = points3d[k]
                                cnt += 1
                                cnt_all += 1
                        print(f"Cloud poinds added: {cnt}")
                        print(f"Cloud poinds total: {cnt_all}")
                        not_done = True
                print(f"Frame {i} done")

    point_cloud_builder = PointCloudBuilder()
    point_cloud_builder.add_points(np.array([id for id in range(len(cloud_points)) if cloud_points[id] is not None]), np.array([point for point in cloud_points if point is not None]))
    
    point_cloud = point_cloud_builder.build_point_cloud()
    return [view_mat3x4_to_pose(pose) for pose in poses if pose is not None], point_cloud

def _compute_pose(corners, cloud_points, intrinsic_mat):
        corners_ids = corners.ids.squeeze(-1)
        
        mask = np.ones_like(corners_ids)
        ids = []
        points = []

        for i in range(len(corners_ids)):
            id = corners_ids[i]
            if cloud_points[id] is not None:
                ids.append(id)
                points.append(corners.points[i])
        
        if len(ids) < 6:
            return cloud_points, None

        points = np.array([point for point in points])
        points3D = np.array([cloud_points[ind] for ind in ids if cloud_points[ind] is not None])
        success, rvec, tvec, inliers = cv2.solvePnPRansac(points3D, points, intrinsic_mat, None)

        if not success:
            return cloud_points, None

        inliers = inliers.squeeze(-1)
        print(f"Inliers number: {inliers.shape[0]}")
        for id in corners_ids:
            if id not in inliers:
                cloud_points[id] = None

        return cloud_points, rodrigues_and_translation_to_view_mat3x4(rvec, tvec)

if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
