#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import numpy as np
import cv2
import operator

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

class CameraTracker():
    def __init__(self, corner_storage, intrinsic_mat, known_view_1, known_view_2):
        self.intrinsic_mat = intrinsic_mat
        self.corner_storage = corner_storage
        self.tracks = [None] * len(corner_storage)
        self.cloud_points = [None] * (corner_storage.max_corner_id() + 1)
        self.tri_params = TriangulationParameters(max_reprojection_error=3., min_triangulation_angle_deg=3., min_depth=0.1)

        if known_view_1 is None or known_view_2 is None:
            self.init_no_know_view()
        else:
            self.tracks[known_view_1[0]] = pose_to_view_mat3x4(known_view_1[1])
            self.tracks[known_view_2[0]] = pose_to_view_mat3x4(known_view_2[1])
            print(f"Processing init frames {known_view_1[0]} and {known_view_2[0]}")
            self.do_triangulation(known_view_1[0], known_view_2[0])

    def init_no_know_view(self):
        best_pose_id = None
        best_size = 0
        best_pose = None
        print(f"Finding best init frames")
        for i in range(1, len(self.corner_storage)):
            print(f"Trying init frames 0 and {i}")
            pose, cloud_size = self.init_with_two_frames(0, i)
            if pose is not None and cloud_size is not None:
                if best_pose_id is None or cloud_size > best_size:
                    best_pose_id = i
                    best_pose = pose
                    best_size = cloud_size

        self.tracks[0] = eye3x4()
        self.tracks[best_pose_id] = pose_to_view_mat3x4(best_pose)
        print(f"Processing init frames {0} and {best_pose_id}")
        self.do_triangulation(0, best_pose_id)


    def init_with_two_frames(self, frame_1, frame_2):
        correspondences = build_correspondences(self.corner_storage[frame_1], self.corner_storage[frame_2])
        
        if correspondences.points_1.shape[0] <= 5:
            return None, None

        essential_mat, inliers_mask = cv2.findEssentialMat(correspondences.points_1, correspondences.points_2, self.intrinsic_mat, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, homography_inliers = cv2.findHomography(correspondences.points_1, correspondences.points_2, method=cv2.RANSAC, confidence=0.999, ransacReprojThreshold=self.tri_params.max_reprojection_error)
        
        if np.count_nonzero(inliers_mask) < 1.5 * np.count_nonzero(homography_inliers):
            return None, None

        ids_to_remove = np.where(inliers_mask == 0)[0]
        correspondences = build_correspondences(self.corner_storage[frame_1], self.corner_storage[frame_2], ids_to_remove=ids_to_remove)

        R1, R2, t = cv2.decomposeEssentialMat(essential_mat)
        pose_1 = view_mat3x4_to_pose(np.hstack((R1, t)))
        pose_2 = view_mat3x4_to_pose(np.hstack((R2, t)))
        pose_3 = view_mat3x4_to_pose(np.hstack((R1, -t)))
        pose_4 = view_mat3x4_to_pose(np.hstack((R2, -t)))
        best_pose = None
        best_size = 0

        for pose in [pose_1, pose_2, pose_3, pose_4]:
            points, ids, median_cos = triangulate_correspondences(correspondences, eye3x4(), pose_to_view_mat3x4(pose), self.intrinsic_mat, self.tri_params)
            if best_pose is None or points.shape[0] > best_size:
                best_pose = pose
                best_size = points.shape[0]

        return best_pose, best_size
        

    def do_triangulation(self, frame_1, frame_2):
        mat_0 = self.tracks[frame_1]
        mat_1 = self.tracks[frame_2]
        
        correspondences = build_correspondences(self.corner_storage[frame_1], self.corner_storage[frame_2])
                
        points3d, correspondences_ids, median_cos = triangulate_correspondences(correspondences, mat_0, mat_1, self.intrinsic_mat, self.tri_params)
    
        cnt = 0
        for i in range(len(correspondences_ids)):
            id = correspondences_ids[i]
            if self.cloud_points[id] is None:
                self.cloud_points[id] = points3d[i]
                cnt += 1
        return cnt

    def update_track(self, frame_id):
        print(f"Processing frame {frame_id}")
        self.compute_track(frame_id)

        if self.tracks[frame_id] is None:
            return False
        cnt = 0
        for j in range(len(self.corner_storage)):
            if self.tracks[j] is not None and frame_id != j:
                cnt += self.do_triangulation(frame_id, j)
       
        print(f"Cloud points added: {cnt}")
        
        cnt_all = len([p for p in self.cloud_points if p is not None])
        print(f"Cloud points total: {cnt_all}")
        print(f"Frame {frame_id} done")
        return True

    def compute_track(self, frame_id):
        corners_ids = self.corner_storage[frame_id].ids.squeeze(-1)
        
        ids = []
        points = []

        for i in range(len(corners_ids)):
            id = corners_ids[i]
            if self.cloud_points[id] is not None:
                ids.append(id)
                points.append(self.corner_storage[frame_id].points[i])
        
        if len(ids) < 4:
            return

        points = np.array([point for point in points])
        points3D = np.array([self.cloud_points[ind] for ind in ids if self.cloud_points[ind] is not None])
        success, rvec, tvec, inliers = cv2.solvePnPRansac(points3D, points, self.intrinsic_mat, None, flags=cv2.SOLVEPNP_EPNP)

        if not success:
            return

        inliers = inliers.squeeze(-1)
        print(f"Inliers number: {inliers.shape[0]}")
        for id in corners_ids:
            if id not in inliers:
                self.cloud_points[id] = None

        self.tracks[frame_id] = rodrigues_and_translation_to_view_mat3x4(rvec, tvec)

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

    tracker = CameraTracker(corner_storage, intrinsic_mat, known_view_1, known_view_2)
   
    not_done = True
    while not_done:
        not_done = False
        for i in range(len(corner_storage)):
            if tracker.tracks[i] is None:
                upd = tracker.update_track(i)
                if (upd):
                    not_done = True
                
    point_cloud_builder = PointCloudBuilder()

    ids_to_add = np.array([id for id in range(len(tracker.cloud_points)) if tracker.cloud_points[id] is not None])
    points_to_add = np.array([point for point in tracker.cloud_points if point is not None])
    if len(points_to_add) > 0:
        point_cloud_builder.add_points(ids_to_add, points_to_add)

    unsuccesful = [track for track in tracker.tracks if track is None]
    if len(unsuccesful) > 0:
        print(f"Failed to find views for {len(unsuccesful)} frames.")

    view_mats = [track for track in tracker.tracks if track is not None]

    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    poses = list(map(view_mat3x4_to_pose, view_mats))
    point_cloud = point_cloud_builder.build_point_cloud()


    return poses, point_cloud



if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()