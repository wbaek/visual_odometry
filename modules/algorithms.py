from __future__ import absolute_import
from modules.utils import Utils

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

class PoseEstimator(object):
    def __init__(self, configs):
        self.configs = configs

    def estimate(self, points1, points2):
        if len(points1) < 8 or len(points2) < 8:
            raise ValueError('arguments points1 and points2 are must larger than 8 elements')
        if len(points1) != len(points2):
            raise ValueError('arguments points1 and points2 are same number of elements')

        focal_length = self.configs['focal_length']
        principle_point = tuple(self.configs['principle_point'])

        essential, status = cv2.findEssentialMat(points1, points2,
            focal=focal_length, pp=principle_point, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        inlier_pair = Utils.remove_outlier(points1, points2, status)
        _, rotation, translation, status = cv2.recoverPose(essential, inlier_pair[0], inlier_pair[1],
            focal=focal_length, pp=principle_point)
        inlier_pair = Utils.remove_outlier(inlier_pair[0], inlier_pair[1], status)
        return rotation, translation, inlier_pair


