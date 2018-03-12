import logging
import copy
import numpy as np
import cv2

class Preprocessor(object):
    def __init__(self, configs):
        self.configs = configs

    def process(self, image):
        _ = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _ = cv2.GaussianBlur(_, tuple(self.configs['gaussian_blur']['kernel_size']), 0)
        return _

class Detector(object):
    def __init__(self, configs):
        self.configs = configs
        self.detector = cv2.FastFeatureDetector_create(
            threshold=configs['threshold'], nonmaxSuppression=configs['nonmax_supression'])

    def detect(self, image):
        return self.detector.detect(image, None)

class Tracker(object):
    def __init__(self, configs):
        self.configs = configs

    def track(self, histories):
        if len(histories) <= 1:
            return [[], []]
        prev_image = histories[-2].image
        prev_points = histories[-2].points

        curr_image = histories[-1].image
        curr_points, status, err = cv2.calcOpticalFlowPyrLK(
            prev_image, curr_image, prev_points, None,
            winSize=tuple(self.configs['window_size']),
            maxLevel=self.configs['max_level'],
            minEigThreshold=self.configs['min_eigen_threshold'],
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        return Utils.remove_outlier(prev_points, curr_points, status)
        #return [prev_points[status==1], curr_points[status==1]]


class Utils(object):
    @staticmethod
    def remove_outlier(pt1, pt2, status):
        status = status.reshape(status.shape[0])
        return [pt1[status>0], pt2[status>0]]
        #return list(zip( *[[p, c] for p, c, s in zip(pt1, pt2, status) if s == 1] ) )

    @staticmethod
    def kp2np(keypoints):
        return np.array([kp.pt for kp in keypoints], dtype=np.float32)

    @staticmethod
    def nonmax_supression(points, scale=5.0):
        hashed = {hash( tuple([int(p/scale+0.5) for p in pt]) ):i for i, pt in enumerate(points)}
        return np.array([points[i] for k, i in hashed.items()])

    @staticmethod
    def append(points1, points2, threshold=5.0):
        #logging.info('%s %s', points1, points2)
        if len(points1) == 0:
            return points2
        if len(points2) == 0:
            return points1
        return np.concatenate((points1, points2), axis=0)

    @staticmethod
    def draw(image, histories):
        num_histories = len(histories)
        for i, history in enumerate(histories):
            ratio = 1.0 * i / num_histories
            inlier_pair = history.tracked_pair if history.inlier_pair is None else history.inlier_pair
            for pt1, pt2 in zip(*inlier_pair):
                cv2.line(image, tuple([int(v) for v in pt1]), tuple([int(v) for v in pt2]),
                    color=(0, 255*(ratio), 255*(1-ratio)), thickness=2)
        for pt2 in histories[-1].points:
            cv2.circle(image, tuple([int(v) for v in pt2]), radius=3,
                color=(255, 255, 255), thickness=-1)
            cv2.circle(image, tuple([int(v) for v in pt2]), radius=2,
                color=(0, 0, 0), thickness=-1)

