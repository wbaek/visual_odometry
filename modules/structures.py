import logging
import time

import numpy as np
import cv2

from modules.utils import Utils

FLANN_INDEX_KDTREE=0
matcher = cv2.FlannBasedMatcher({'algorithm':FLANN_INDEX_KDTREE, 'tree':5}, {'checks':50})

class History(object):
    def __init__(self, original=None, image=None,
        keypoints=[], points=None, descriptions=None, matches=[],
        pose=[np.eye(3), np.zeros((3, 1))],
        elapsed=None):
        self.original = original
        self.image = image

        self.keypoints = keypoints
        self.points = points
        self.descriptions = descriptions
        self.matches = matches

        self.pose = pose
        self.elapsed = elapsed

    def add(self, keypoints, descriptions, distance_threshold=5.0):
        points = Utils.kp2np(keypoints)
        if len(self.keypoints) == 0:
            self.keypoints = keypoints
            self.points = points
            self.descriptions = descriptions
            return self

        _matches = matcher.radiusMatch(points, self.points, maxDistance=distance_threshold)
        status = np.array([1 if len(match) == 0 else 0 for match in _matches])

        self.keypoints += [kp for kp, s in zip(keypoints, status) if s>0]
        self.points = np.concatenate( [self.points, points[status>0]] )
        self.descriptions = np.concatenate( [self.descriptions, descriptions[status>0]] )
        return self

    def __repr__(self):
        return 'original:{} image:{} #keypoints:{} matches:{} pose:{}'.format(
            self.original.shape if self.original is not None else None,
            self.image.shape if self.image is not None else None,
            len(self.keypoints) if self.keypoints is not None else None,
            self.matches.shape if self.matches is not None else None,
            [p.shape for p in self.pose] if self.pose is not None else None,
        )

class Elapsed(object):
    def __init__(self):
        pass

    def clear(self):
        self.timestamps = [('total', time.time())]
        self.elapsed = {}

    def tic(self, name):
        self.timestamps.append((name, time.time()))

    def calc(self):
        self.elapsed = {'total':self.timestamps[-1][1] - self.timestamps[0][1]}
        self.elapsed.update({t[0]:t[1] - self.timestamps[i][1] for i, t in enumerate(self.timestamps[1:])})

    def __repr__(self):
        self.calc()
        return ' '.join(['{}:{:.3f}'.format(key, self.elapsed[key]) for key, value in self.timestamps])
