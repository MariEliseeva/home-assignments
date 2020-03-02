#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims
from scipy.spatial.distance import cdist

from _corners import FrameCorners, CornerStorage, StorageImpl
from _corners import dump, load, draw, without_short_tracks, create_cli


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    MAX_CORNERS = 1000
    QUALITY_LEVEL = 0.005
    MIN_DISTANCE = 10

    PARAMS_CORNERS = dict(blockSize=3,
                      gradientSize=3,
                      useHarrisDetector=False,
                      k=0.04)

    PARAMS_MOVED = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
                     minEigThreshold=0.001)

    image_0 = frame_sequence[0]
    image_0 = np.round(image_0 * 255).astype('uint8')
    corners_points = np.array(cv2.goodFeaturesToTrack(image_0, MAX_CORNERS, QUALITY_LEVEL, MIN_DISTANCE, None, **PARAMS_CORNERS))
    corners_ids = np.array(range(len(corners_points)))

    corners = FrameCorners(corners_ids, corners_points, np.array([10] * len(corners_points)))
    builder.set_corners_at_frame(0, corners)

    free_id = len(corners_points)

    for frame, image_1 in enumerate(frame_sequence[1:], 1):
        
        image_1 = np.round(image_1 * 255).astype('uint8')

        points_1, status, _ = cv2.calcOpticalFlowPyrLK(image_0, image_1, corners.points, None, **PARAMS_MOVED)
        
        status = status.reshape(-1)
        moved_points = points_1[status == 1]
        ids = corners.ids[status == 1]

        new_points = cv2.goodFeaturesToTrack(image_1, MAX_CORNERS, QUALITY_LEVEL, MIN_DISTANCE, None, **PARAMS_CORNERS)
        new_points = new_points.reshape(-1, 2)
        
        dists = cdist(new_points, moved_points).min(axis=1)
        
        points_to_add = max(len(new_points), MAX_CORNERS) - len(moved_points)
        new_points_indices = np.argsort(dists)[-points_to_add:]

        corners = FrameCorners(
                np.concatenate([ids.reshape(-1), free_id + np.arange(points_to_add)]),
                np.concatenate([moved_points, new_points[new_points_indices]]),
                np.array([10] * (len(moved_points) + len(new_points_indices))))

        free_id += points_to_add

        builder.set_corners_at_frame(frame, corners)
        image_0 = image_1


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
