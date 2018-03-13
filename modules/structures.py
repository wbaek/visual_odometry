import time

import numpy as np

class History(object):
    def __init__(self, original=None, image=None, keypoints=None,
        points=None, tracked_pair=None, inlier_pair=None,
        pose=[np.eye(3), np.zeros((3, 1))],
        elapsed=None):
        self.original = original
        self.image = image
        self.keypoints = keypoints
        self.points = points
        self.tracked_pair = tracked_pair
        self.inlier_pair = inlier_pair
        self.pose = pose
        self.elapsed = elapsed

    def __repr__(self):
        return 'original:{} image:{} #keypoints:{} points:{} #tracked_pair:{} #inlier_pair:{} pose:{}'.format(
            self.original.shape if self.original is not None else None,
            self.image.shape if self.image is not None else None,
            len(self.keypoints) if self.keypoints is not None else None,
            self.points.shape if self.points is not None else None,
            [len(points) for points in self.tracked_pair] if self.tracked_pair is not None else None,
            [len(points) for points in self.inlier_pair] if self.inlier_pair is not None else None,
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
