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
        status = status.reshape(status.shape[0])
        return curr_points, status

    def update(self, keypoints, points, status):
        keypoints = [copy.deepcopy(kp) for kp, st in zip(keypoints, status) if st]

        results = [None] * len(keypoints)
        for i, (kp, pt) in enumerate(zip(keypoints, points[status>0])):
            kp.pt = tuple(pt)
            results[i] = kp
        return results

class Matcher(object):
    def __init__(self, configs):
        self.configs = configs
        index_params = dict(algorithm=cv2.FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params,search_params)

    def match(self, description1, description2, threshold):
        matches = self.flann.radiusMatch(description1, description2, threshold)
        return matches

class PoseEstimator(object):
    def __init__(self, configs):
        self.configs = configs
        self.MIN_POINTS_THRESHOLD = 8
        self.EPSILON_THRESHOLD = 5.0

    def essential(self, prev_points, curr_points):
        if len(prev_points) < self.MIN_POINTS_THRESHOLD or len(curr_points) < self.MIN_POINTS_THRESHOLD:
            raise ValueError('arguments points1 and points2 are must larger than %d elements'%self.MIN_POINTS_THRESHOLD)
        if len(prev_points) != len(curr_points):
            raise ValueError('arguments points1 and points2 are same number of elements')

        focal_length = self.configs['focal_length']
        principle_point = tuple(self.configs['principle_point'])

        inlier_pair = [prev_points, curr_points]
        #'''
        essential, status = cv2.findEssentialMat(inlier_pair[1], inlier_pair[0],
            focal=focal_length, pp=principle_point, method=cv2.RANSAC, prob=0.995, threshold=1.0)
        status = status.reshape(status.shape[0])
        inlier_pair = [pt[status>0] for pt in inlier_pair]
        essential, _ = cv2.findEssentialMat(inlier_pair[1], inlier_pair[0],
            focal=focal_length, pp=principle_point, method=cv2.RANSAC, prob=0.95, threshold=1.0)

        '''
        essential, status = cv2.findEssentialMat(inlier_pair[1], inlier_pair[0],
            focal=focal_length, pp=principle_point, method=cv2.LMEDS, prob=0.995)
        #'''
        status = status.reshape(status.shape[0])
        return essential, status

    def pose(self, prev_points, curr_points, essential):
        focal_length = self.configs['focal_length']
        principle_point = tuple(self.configs['principle_point'])

        inlier_pair = [prev_points, curr_points] 
        num_inliers, rotation, translation, status = cv2.recoverPose(essential, inlier_pair[1], inlier_pair[0],
            focal=focal_length, pp=principle_point)
        status = status.reshape(status.shape[0])

        eps = np.sum(np.abs(inlier_pair[0] - inlier_pair[1])/len(inlier_pair[0]))
        if num_inliers >= self.MIN_POINTS_THRESHOLD//2 and eps > self.EPSILON_THRESHOLD:
            pass
        else:
            logging.warning('inliers:%d num_inliers:%d eps:%.3f', len(inlier_pair[0]), num_inliers, eps)
            rotation, translation = np.eye(3), np.zeros((3, 1))
        return rotation, translation, status

    def decompose(self, essential):
        rotation, rotation2, translation = cv2.decomposeEssentialMat(essential)
        return rotation2, translation

class Reconstructor(object):
    def __init__(self, configs):
        self.configs = configs

        focal_length = self.configs['focal_length']
        principle_point = tuple(self.configs['principle_point'])
        self.intrinsic = np.array([[focal_length, 0.0, principle_point[0]], [0.0, focal_length, principle_point[1]], [0.0, 0.0, 1.0]])
        logging.info(self.intrinsic)

    def match(self, match1, match2):
        m1 = np.array(match1)
        m2 = np.array(match2)
        target = m1[:,1]

        match = []
        for i, v in enumerate(m2[:,0]):
            if v in target:
                j = np.where(target == v)[0][0]
                match.append([j, i])
        return match

    def reconstruct(self, histories, frames=5):
        histories = histories[-frames:]

        matches = [[pair[1], pair[1]] for pair in histories[0].matches]
        for h in histories[1:]:
            if len(matches) == 0:
                raise Exception('not enough matched points')
            matches = self.match(matches, h.matches)
        matches = np.array(matches)

        history1 = histories[0]
        history2 = histories[-1]
        matches1 = matches[:,0]
        matches2 = matches[:,1]
        points1 = history1.points[matches1, :].astype(np.float64)
        points2 = history2.points[matches2, :].astype(np.float64)

        projection1 = self.intrinsic.dot( np.concatenate(history1.pose, axis=1) ).astype(np.float64)
        projection2 = self.intrinsic.dot( np.concatenate(history2.pose, axis=1) ).astype(np.float64)

        reconstructed = cv2.triangulatePoints(
            projection1, projection2,
            points1.T, points2.T)

        p1 = projection1.dot( reconstructed )
        p1 = p1[:2] / p1[2]

        p2 = projection2.dot( reconstructed )
        p2 = p2[:2] / p2[2]

        distance1 = np.linalg.norm(points1.T - p1, axis=0)
        distance2 = np.linalg.norm(points2.T - p2, axis=0)
        status = (distance1 + distance2) < 30.0
        logging.info('reconstruction: %d -> %d', len(status), len(status[status==True]))

        reconstruction = cv2.triangulatePoints(
            projection1, projection2,
            points1[status==True].T, points2[status==True].T)

        point3D =  reconstructed[:3] / reconstructed[3]
        return point3D.T[status == True]


