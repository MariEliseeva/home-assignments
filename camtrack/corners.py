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
    QUALITY_LEVEL = 0.01
    MIN_DISTANCE = 10

    image_0 = frame_sequence[0]
    image_0 = np.round(image_0 * 255).astype('uint8')
    corners_points = np.array(cv2.goodFeaturesToTrack(image_0, MAX_CORNERS, QUALITY_LEVEL, MIN_DISTANCE))
    corners_ids = np.array(range(len(corners_points)))

    corners = FrameCorners(corners_ids, corners_points, np.array([10] * len(corners_points)))
    builder.set_corners_at_frame(0, corners)

    free_id = len(corners_points)

    for frame, image_1 in enumerate(frame_sequence[1:], 1):
        
        image_1 = np.round(image_1 * 255).astype('uint8')

        points, status, _ = cv2.calcOpticalFlowPyrLK(image_0, image_1, corners.points, None, winSize=(10, 10))
        
        #print(points[0])
        status = status.reshape(-1)
        corners_ids = []
        moved_corner_points = []
        for i in range(len(status)):
            if status[i] == 1:
                corners_ids.append(corners.ids[i][0])
                moved_corner_points.append([points[i]])
        corners_ids = (np.array(corners_ids))
        
        if len(moved_corner_points) < MAX_CORNERS:

            mask = np.full_like(image_1, 255)
            for elem in moved_corner_points:
                x, y = elem[0][0], elem[0][1]
                cv2.circle(mask, (x, y), MIN_DISTANCE, 0, -1)

            new_corners_points = np.array(cv2.goodFeaturesToTrack(image_1, MAX_CORNERS - len(moved_corner_points), QUALITY_LEVEL, MIN_DISTANCE, mask=mask))
            corners_ids = np.append(np.array(corners_ids), np.array(range(free_id, free_id + len(new_corners_points))))
            free_id += len(new_corners_points)
            corners_points = np.append(moved_corner_points, new_corners_points, axis=0)
        
        corners = FrameCorners(corners_ids, corners_points, np.array([10] * len(corners_points)))
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
