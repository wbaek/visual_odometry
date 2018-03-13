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
    estimator = modules.PoseEstimator(configs['pose_estimator'])
    elapsed = modules.Elapsed()

    histories = []
    for frame in range(0, 10000):
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

        try:
            R, t, inlier_pair = estimator.estimate(tracked_pair[0], tracked_pair[1])
            histories[-1].pose = [R, t]
            histories[-1].inlier_pair = inlier_pair
        except ValueError as e:
            logging.warning(e)
        elapsed.tic('pose_estimation')

        modules.Utils.draw(original, histories[-5:])
        elapsed.tic('draw_result')

        logging.info(filename)
        logging.info(histories[-1])
        logging.info(elapsed)

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

