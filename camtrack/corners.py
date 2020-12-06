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

QUALITY_LEVEL = 0.005
MAX_CORNERS = 4000
MIN_DISTANCE = 7
BLOCK_SIZE = 13
MAX_LEVEL = 3
MAX_ITERS = 10
LK_EPS = 0.01
MIN_TRACK_LENGTH = 3


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


class Corner:
    def __init__(self, corner, index, size, min_eigen: int, prev_dist: float):
        self.corner = [corner[0], corner[1]]
        self.index = index
        self.size = size
        self.min_eigen = min_eigen
        self.prev_dist = prev_dist


class NormalCornerStorage:
    max_index = 0

    def __init__(self):
        self.corners = []
        self.corner_mask = np.zeros((2000, 2000))

    def fill_mask(self, i, j):
        d = MIN_DISTANCE
        i = int(i)
        j = int(j)
        for k1 in range(i - d, i + d + 1):
            for k2 in range(j - d, j + d + 1):
                if (i - k1) ** 2 + (j - k2) ** 2 < d ** 2:
                    self.corner_mask[k1][k2] = 1

    def add_corner(self, corner):
        if not self.can_add(corner):
            return
        self.fill_mask(corner.corner[0], corner.corner[1])
        self.corners.append(corner)

    def add_corner1(self, corner, size, eigen, dist):
        ncorn = Corner(corner, NormalCornerStorage.max_index, size, eigen, dist)
        if not self.can_add(ncorn):
            return
        self.add_corner(ncorn)
        self.fill_mask(corner[0], corner[1])
        NormalCornerStorage.max_index += 1

    def add_new_corners(self, corners, size, image):
        for corner in corners.reshape(-1, 2):
            self.add_corner1(corner, size, calc_min_eigen(image, corner), 0)

    def get_corner_array(self):
        return np.asarray(list(map(lambda x: x.corner, self.corners)))

    @staticmethod
    def is_valid_corner(img, corner):
        return corner[1] < img.shape[0] - 1 and corner[0] < img.shape[1] - 1

    def to_frame_corners(self):
        sizes = []
        corners = []
        indexes = []
        for corner in self.corners:
            sizes.append(corner.size)
            corners.append(corner.corner)
            indexes.append(corner.index)

        return FrameCorners(
            np.array(indexes),
            np.array(corners),
            np.array(sizes)
        )

    def add_existing_corners(self, new_corners, image, filter_close=False):
        for ncorn in new_corners.corners:
            self.add_corner(ncorn)

    def can_add(self, corner):
        return self.corner_mask[int(corner.corner[0])][int(corner.corner[1])] == 0


def get_new_corners(image):
    def get_corners(image):
        return cv2.goodFeaturesToTrack(
            image,
            mask=None,
            maxCorners=MAX_CORNERS,
            minDistance=MIN_DISTANCE,
            blockSize=BLOCK_SIZE,
            qualityLevel=QUALITY_LEVEL
        )

    my_corners = NormalCornerStorage()
    lvls, pyr0 = get_pyramid(image)
    for lvl in range(0, lvls * 2, 2):
        pyr = pyr0[lvl]
        p1 = get_corners(pyr)
        nscale = image.shape[0] / pyr.shape[0]
        new_block_size = int(BLOCK_SIZE * nscale)
        my_corners.add_new_corners(scale_corners(p1, nscale), new_block_size, image)
    return my_corners


def get_pyramid(image):
    return cv2.buildOpticalFlowPyramid(
        image,
        winSize=(BLOCK_SIZE, BLOCK_SIZE),
        maxLevel=MAX_LEVEL
    )


def calc_flow(old_img, new_img, old_points):
    criteria = cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT
    return cv2.calcOpticalFlowPyrLK(
            old_img,
            new_img,
            old_points,
            None,
            maxLevel=MAX_LEVEL,
            winSize=(BLOCK_SIZE, BLOCK_SIZE),
            criteria=(criteria, MAX_ITERS, LK_EPS)
        )


def calc_min_eigen(img, corner):
    if not NormalCornerStorage.is_valid_corner(img, corner):
        return
    a = np.rint(corner).astype(np.int32)
    return img[a[1]][a[0]]


def calc_dist(corner, prev_corner):
    return np.linalg.norm(corner - prev_corner)


def scale_corners(corners, sc):
    return corners * sc


def tou8(frame):
    return (frame * 255).astype(np.uint8)


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:

    image_0 = tou8(frame_sequence[0])

    my_corners = get_new_corners(image_0)

    corners = my_corners.to_frame_corners()
    builder.set_corners_at_frame(0, corners)
    for frame, image_1 in enumerate(frame_sequence[1:], 1):
        if frame % 10 == 0:
            print(frame + 1, "out of", len(frame_sequence))
        p0 = my_corners.get_corner_array()
        rimage_1 = tou8(image_1)

        p1, status, _ = calc_flow(image_0, rimage_1, p0)
        p0rev, status_rev, _ = calc_flow(rimage_1, image_0, p1)

        d = np.linalg.norm((p0 - p0rev).reshape(-1, 2), axis=1)
        new_my_corners = NormalCornerStorage()

        eigen_img = cv2.cornerMinEigenVal(rimage_1, BLOCK_SIZE)
        for i, (old_corner, new_corner) in enumerate(zip(my_corners.corners, p1)):
            if status[i] == 1 and status_rev[i] == 1 and d[i] < 1.0:
                x1, y1 = new_corner.ravel()
                min_eigen = calc_min_eigen(eigen_img, new_corner)
                dist = calc_dist(new_corner, old_corner.corner)
                new_my_corners.add_corner(Corner([x1, y1], old_corner.index, old_corner.size, min_eigen, dist))

        just_new_corners = get_new_corners(rimage_1)
        new_my_corners.add_existing_corners(just_new_corners, rimage_1)

        corners = new_my_corners.to_frame_corners()

        builder.set_corners_at_frame(frame, corners)
        image_0 = rimage_1
        my_corners = new_my_corners


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
