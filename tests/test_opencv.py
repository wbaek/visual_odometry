import os
import cv2
import numpy as np

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'datas')

def test_detector():
    image = cv2.imread(FIXTURE_DIR+'/000000.png', cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    detector = cv2.FastFeatureDetector_create(threshold=60, nonmaxSuppression=True)
    keypoints = detector.detect(gray, None)

    assert len(keypoints) == 817
    assert isinstance(keypoints[0], cv2.KeyPoint) == True
    assert keypoints[0].pt == (71.0, 3.0)
    assert keypoints[0].size == 7.0
    assert keypoints[0].angle == -1.0
    assert keypoints[0].response == 62.0
    assert keypoints[0].octave == 0.0
    assert keypoints[0].class_id == -1

def test_tracker():
    images = [cv2.imread(FIXTURE_DIR+'/%06d.png'%i, cv2.IMREAD_COLOR) for i in range(2)]
    grays = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]

    detector = cv2.FastFeatureDetector_create(threshold=60, nonmaxSuppression=True)
    keypoints = [detector.detect(gray, None) for gray in grays]

    prev_points = np.array([kp.pt for kp in keypoints[0]], dtype=np.float32)
    curr_points, status, err = cv2.calcOpticalFlowPyrLK(
        grays[0], grays[1], prev_points, None,
        winSize=(21, 21), maxLevel=3, minEigThreshold=0.001,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )
    status = status.reshape(status.shape[0])

    assert len(prev_points) == 817
    assert len(curr_points) == 817
    assert len(curr_points[status>0]) == 798

def test_pose_estimator():
    images = [cv2.imread(FIXTURE_DIR+'/%06d.png'%i, cv2.IMREAD_COLOR) for i in range(2)]
    grays = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]

    detector = cv2.FastFeatureDetector_create(threshold=60, nonmaxSuppression=True)
    keypoints = [detector.detect(gray, None) for gray in grays]

    prev_points = np.array([kp.pt for kp in keypoints[0]], dtype=np.float32)
    curr_points, status, err = cv2.calcOpticalFlowPyrLK(
        grays[0], grays[1], prev_points, None,
        winSize=(21, 21), maxLevel=3, minEigThreshold=0.001,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )
    assert len(prev_points) == 817
    assert len(curr_points) == 817

    status = status.reshape(status.shape[0])
    prev_points, curr_points = prev_points[status>0], curr_points[status>0]
    assert len(prev_points) == 798
    assert len(curr_points) == 798

    essential, status = cv2.findEssentialMat(curr_points, prev_points,
        focal=718.8560, pp=(607.1928, 185.2157), method=cv2.RANSAC, prob=0.9999, threshold=1.0)
    status = status.reshape(status.shape[0])
    prev_points, curr_points = prev_points[status>0], curr_points[status>0]
    assert len(prev_points) == 750
    assert len(curr_points) == 750

    num_inliers, rotation, translation, status = cv2.recoverPose(essential, curr_points, prev_points,
        focal=718.8560, pp=(607.1928, 185.2157))
    status = status.reshape(status.shape[0])
    prev_points, curr_points = prev_points[status>0], curr_points[status>0]
    assert len(prev_points) == 567
    assert len(curr_points) == 567

def test_matcher():
    images = [cv2.imread(FIXTURE_DIR+'/%06d.png'%i, cv2.IMREAD_COLOR) for i in range(2)]
    grays = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]

    detector = cv2.FastFeatureDetector_create(threshold=60, nonmaxSuppression=True)
    keypoints = [detector.detect(gray, None) for gray in grays]

    prev_points = np.array([kp.pt for kp in keypoints[0]], dtype=np.float32)
    curr_points, status, err = cv2.calcOpticalFlowPyrLK(
        grays[0], grays[1], prev_points, None,
        winSize=(21, 21), maxLevel=3, minEigThreshold=0.001,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )
    status = status.reshape(status.shape[0])

    FLANN_INDEX_KDTREE=0
    FLANN_INDEX_LSH=6
    matcher = cv2.FlannBasedMatcher({'algorithm':FLANN_INDEX_KDTREE, 'trees':5}, {'checks':50})
    target_points = np.array([kp.pt for kp in keypoints[1]], dtype=np.float32)
    matches = matcher.knnMatch(queryDescriptors=curr_points, trainDescriptors=target_points, k=1)
    assert len(curr_points) == 817
    assert len(target_points) == 845
    assert len(matches) == 817
    assert isinstance(matches[0][0], cv2.DMatch) == True
    assert matches[0][0].queryIdx == 0
    assert matches[1][0].queryIdx == 1
    assert matches[0][0].trainIdx == 34
    assert matches[0][0].trainIdx == 34
    assert matches[0][0].imgIdx == 0
    assert matches[0][0].distance == 55.99501419067383
    
    matches = matcher.radiusMatch(curr_points, target_points, maxDistance=3.0)
    status = np.array([0 if len(m) == 0 else 1 for m in matches])
    assert len(curr_points) == 817
    assert len(curr_points[status>0]) == 611

def test_descriptor():
    images = [cv2.imread(FIXTURE_DIR+'/%06d.png'%i, cv2.IMREAD_COLOR) for i in range(2)]
    grays = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]

    detector = cv2.FastFeatureDetector_create(threshold=60, nonmaxSuppression=True)
    keypoints = [detector.detect(gray, None) for gray in grays]
    assert len(keypoints[0]) == 817
    assert len(keypoints[1]) == 845

    descriptor = cv2.ORB_create()
    keypoints, descriptions = list(zip(
        *[descriptor.compute(gray, keypoint) for gray, keypoint in zip(grays, keypoints)]
    ))
    assert len(keypoints[0]) == 767
    assert len(keypoints[1]) == 784
    assert descriptions[0].shape == (767, 32)
    assert descriptions[1].shape == (784, 32)

    matcher = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=False)
    matches = matcher.knnMatch(descriptions[1], descriptions[0], k=2)

    status = [1 if match[0].distance/match[1].distance<0.7 else 0 for match in matches]
    status = np.array(status)
    assert len(status) == 784
    assert len(status[status>0]) == 458

    matcher = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.knnMatch(descriptions[1], descriptions[0], k=1)

    status = [1 if len(match)>0 else 0 for match in matches]
    status = np.array(status)
    assert len(status) == 784
    assert len(status[status>0]) == 591

