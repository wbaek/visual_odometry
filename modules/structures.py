from __future__ import absolute_import
from modules.utils import Utils

import copy
import time
import numpy as np
import cv2

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

    def set_matches(self, keypoints, points, descriptions, status):
        status = status.reshape(status.shape[0])
        self.keypoints = [kp for kp, st in zip(keypoints, status) if st>0]
        self.points = points[status>0]
        self.descriptions = descriptions[status>0]

        self.matches = []
        j = 0
        for i, st in enumerate(status):
            if st>0:
                self.matches.append( (i, j) )
                j += 1

    def remove_outlier(self, status):
        status = status.reshape(status.shape[0])
        self.keypoints = [kp for kp, st in zip(self.keypoints, status) if st>0]
        self.points = self.points[status>0]
        self.descriptions = self.descriptions[status>0]

        _matches = []
        j = 0
        for i, (match, st) in enumerate(zip(self.matches, status)):
            if st>0:
                _matches.append( (match[0], j) )
                j += 1
        self.matches = _matches


    def add(self, keypoints, descriptions, distance_threshold=3.0):
        if len(self.keypoints) == 0 or self.descriptions is None:
            self.keypoints = keypoints
            self.points = Utils.kp2np(keypoints)
            self.descriptions = descriptions
            return self
            
        target = np.array([kp.pt for kp in self.keypoints], dtype=np.float32)
        query = np.array([kp.pt for kp in keypoints], dtype=np.float32)
        _matches = matcher.radiusMatch(query, target, maxDistance=distance_threshold)
        status = np.array([1 if len(match) == 0 else 0 for match in _matches])

        delta = [kp for kp, s in zip(keypoints, status) if s>0]
        self.keypoints = self.keypoints + delta
        self.points = np.concatenate( [self.points, Utils.kp2np(delta)] )
        self.descriptions = np.concatenate( [self.descriptions, descriptions[status>0]] )
        return self

    def __repr__(self):
        return 'original:{} image:{} #keypoints:{} #points:{} descriptions:{} #matches:{} pose:{}'.format(
            self.original.shape if self.original is not None else None,
            self.image.shape if self.image is not None else None,
            len(self.keypoints) if self.keypoints is not None else None,
            len(self.points) if self.points is not None else None,
            self.descriptions.shape if self.descriptions is not None else None,
            len(self.matches) if self.matches is not None else None,
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
