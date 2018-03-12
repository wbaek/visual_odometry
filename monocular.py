import sys
import logging
import time

import numpy as np
import cv2

import modules

def main(args, configs):
    logging.info('\nargs=%s\nconfig=%s', args, configs)

    preprocessor = modules.Preprocessor(configs['preprocessing'])
    detector = modules.Detector(configs['detector'])
    tracker = modules.Tracker(configs['tracker'])
    elapsed = modules.Elapsed()

    histories = []
    for frame in range(0, 1000):
        elapsed.clear()

        filename = '%s/image_2/%06d.png'%(args.path, frame)
        original = cv2.imread(filename, cv2.IMREAD_COLOR)
        image = preprocessor.process(original)

        histories.append(modules.History(original=original, image=image))
        elapsed.tic('load')

        tracked_pair = tracker.track(histories)
        histories[-1].tracked_pair = tracked_pair
        elapsed.tic('tracking')

        keypoints = detector.detect(image)
        histories[-1].keypoints = keypoints
        elapsed.tic('detection')

        points = modules.Utils.kp2np(keypoints)
        histories[-1].points = modules.Utils.nonmax_supression(modules.Utils.append(tracked_pair[1], points))
        elapsed.tic('appending')

        if len(tracked_pair[0]) > 8:
            focal = configs['pose_estimator']['focal_length']
            pp = tuple(configs['pose_estimator']['principle_point'])
            essential, status = cv2.findEssentialMat(tracked_pair[1], tracked_pair[0], focal=focal, pp=pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            inlier_pair = modules.Utils.remove_outlier(tracked_pair[0], tracked_pair[1], status)
            _, rotation, translation, status = cv2.recoverPose(essential, inlier_pair[1], inlier_pair[0], focal=focal, pp=pp)
            inlier_pair = modules.Utils.remove_outlier(inlier_pair[0], inlier_pair[1], status)

            histories[-1].inlier_pair = inlier_pair
            elapsed.tic('pose_estimation')

        logging.info(filename)
        logging.info(histories[-1])
        logging.info(elapsed)

        modules.Utils.draw(original, histories[-5:])
        cv2.imshow('image', original)
        cv2.waitKey(1)

if __name__ == '__main__':
    import argparse
    import ujson as json
    parser = argparse.ArgumentParser(description='visual odometry monocular')
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='config json file')
    parser.add_argument('-p', '--path', type=str, default='/data/private/storage/dataset/kitti/odometry/visual_odometry/dataset/sequences/00/')

    parser.add_argument('--log-filename',   type=str, default='')
    args = parser.parse_args()

    log_format = '[%(asctime)s %(levelname)s] %(message)s'
    if not args.log_filename:
        logging.basicConfig(level=logging.INFO, format=log_format, stream=sys.stderr)
    else:
        logging.basicConfig(level=logging.INFO, format=log_format, filename=args.log_filename)
    logging.getLogger("requests").setLevel(logging.WARNING)

    with open(args.config, 'r') as f:
        configs = json.load(f)
    main(args, configs)

