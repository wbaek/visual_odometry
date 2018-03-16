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
        self.descriptor = cv2.ORB_create()

    def detect(self, image):
        return self.detector.detect(image, None)

    def compute(self, image, keypoints):
        return self.descriptor.compute(image, keypoints)

class Tracker(object):
    def __init__(self, configs):
        self.configs = configs
 
    def track(self, prev_image, curr_image, prev_points):
        curr_points, status, err = cv2.calcOpticalFlowPyrLK(
            prev_image, curr_image, prev_points, None,
            winSize=tuple(self.configs['window_size']),
            maxLevel=self.configs['max_level'],
            minEigThreshold=self.configs['min_eigen_threshold'],
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        return curr_points, status

    def update(self, keypoints, points):
        keypoints = [copy.deepcopy(kp) for kp in keypoints]
        results = []
        for kp, pt in zip(keypoints, points):
            kp.pt = tuple(pt)
            results.append(kp)
        return results

class PoseEstimator(object):
    def __init__(self, configs):
        self.configs = configs
        self.MIN_POINTS_THRESHOLD = 8
        self.EPSILON_THRESHOLD = 5.0
        self.focal_length = self.configs['focal_length']
        self.principle_point = tuple(self.configs['principle_point'])

    def essential(self, prev_points, curr_points):
        if len(prev_points) < self.MIN_POINTS_THRESHOLD or len(curr_points) < self.MIN_POINTS_THRESHOLD:
            raise ValueError('arguments points1 and points2 are must larger than %d elements'%self.MIN_POINTS_THRESHOLD)
        if len(prev_points) != len(curr_points):
            raise ValueError('arguments points1 and points2 are same number of elements')

        inlier_pair = [prev_points, curr_points]
        #'''
        essential, status = cv2.findEssentialMat(inlier_pair[1], inlier_pair[0],
            focal=self.focal_length, pp=self.principle_point, method=cv2.RANSAC, prob=0.95, threshold=3.0)
        '''
        essential, status = cv2.findEssentialMat(inlier_pair[1], inlier_pair[0],
            focal=self.focal_length, pp=self.principle_point, method=cv2.LMEDS, prob=0.95)
        '''
        return essential, status

    def pose(self, prev_points, curr_points, essential):
        inlier_pair = [prev_points, curr_points]
        num_inliers, rotation, translation, status = cv2.recoverPose(essential, inlier_pair[1], inlier_pair[0],
            focal=self.focal_length, pp=self.principle_point)
        inlier_pair = Utils.remove_outlier(inlier_pair[0], inlier_pair[1], status)

        eps = np.sum(np.abs(inlier_pair[0] - inlier_pair[1])/len(inlier_pair[0]))
        if num_inliers >= self.MIN_POINTS_THRESHOLD//2 and eps > self.EPSILON_THRESHOLD:
            pass
        else:
            logging.warning('inliers:%d num_inliers:%d eps:%.3f', len(inlier_pair[0]), num_inliers, eps)
            rotation, translation = np.eye(3), np.zeros((3, 1))
        return rotation, translation, status


