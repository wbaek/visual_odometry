import logging
import time
import copy

import numpy as np
import cv2

from modules.utils import Utils

FLANN_INDEX_KDTREE=0
matcher = cv2.FlannBasedMatcher({'algorithm':FLANN_INDEX_KDTREE, 'tree':5}, {'checks':50})

class History(object):
    def __init__(self, original=None, image=None,
        keypoints=[], points=None, descriptions=None, matches=[],
        reconstructed=[],
        delta=[np.eye(3), np.zeros((3, 1))],
        pose=[np.eye(3), np.zeros((3, 1))],
        elapsed=None):
        self.original = original
        self.image = image

        self.keypoints = keypoints
        self.points = copy.deepcopy(points)
        self.descriptions = copy.deepcopy(descriptions)
        self.matches = matches
        self.reconstructed = reconstructed

        self.delta = delta
        self.pose = copy.deepcopy(pose)
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

    def update(self, delta):
        self.delta = delta
        self.pose[1] = self.pose[1] + self.pose[0].dot( delta[1] )
        self.pose[0] = self.pose[0].dot( delta[0] )

    def __repr__(self):
        rotation, _ = cv2.Rodrigues(self.pose[0])
        theta = np.linalg.norm(rotation)
        theta = theta if theta > 1e-5 else 1.0
        rotation = list(rotation/theta)
        return 'rotation:[{}]  translation:[{}] #keypoints:{} #points:{} matches:{} #reconstructed:{}'.format(
            ', '.join(['%.2f'%v for v in rotation+[theta*180.0/np.pi]]),
            ', '.join(['%.2f'%v for v in self.pose[1]]),
            len(self.keypoints) if self.keypoints is not None else None,
            len(self.points) if self.points is not None else None,
            len(self.matches),
            len(self.reconstructed)
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
