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
        self.MIN_POINTS_THRESHOLD = 8
        self.EPSILON_THRESHOLD = 5.0

    def estimate(self, prev_points, curr_points):
        if len(prev_points) < self.MIN_POINTS_THRESHOLD or len(curr_points) < self.MIN_POINTS_THRESHOLD:
            raise ValueError('arguments points1 and points2 are must larger than %d elements'%self.MIN_POINTS_THRESHOLD)
        if len(prev_points) != len(curr_points):
            raise ValueError('arguments points1 and points2 are same number of elements')

        focal_length = self.configs['focal_length']
        principle_point = tuple(self.configs['principle_point'])

        inlier_pair = [prev_points, curr_points]
        essential, status = cv2.findEssentialMat(inlier_pair[1], inlier_pair[0],
            focal=focal_length, pp=principle_point, method=cv2.RANSAC, prob=0.95, threshold=3.0)
        inlier_pair = Utils.remove_outlier(inlier_pair[0], inlier_pair[1], status)
        essential, status = cv2.findEssentialMat(inlier_pair[1], inlier_pair[0],
            focal=focal_length, pp=principle_point, method=cv2.LMEDS, prob=0.90)
        inlier_pair = Utils.remove_outlier(inlier_pair[0], inlier_pair[1], status)

        num_inliers, rotation, translation, status = cv2.recoverPose(essential, inlier_pair[1], inlier_pair[0],
            focal=focal_length, pp=principle_point)

        eps = np.sum(np.abs(inlier_pair[0] - inlier_pair[1])/len(inlier_pair[0]))
        if num_inliers >= self.MIN_POINTS_THRESHOLD//2 and eps > self.EPSILON_THRESHOLD:
            inlier_pair = Utils.remove_outlier(inlier_pair[0], inlier_pair[1], status)
        else:
            logging.warning('inliers:%d num_inliers:%d eps:%.3f', len(inlier_pair[0]), num_inliers, eps)
            rotation, translation = np.eye(3), np.zeros((3, 1))
        return rotation, translation, inlier_pair


