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

    trajectory = np.zeros((600, 800, 3), dtype=np.uint8)

    histories = []
    global_pose = [np.eye(3), np.zeros((3, 1))]
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
        global_pose[1] = global_pose[1] + global_pose[0].dot( histories[-1].pose[1] )
        global_pose[0] = histories[-1].pose[0].dot( global_pose[0] ) 
        elapsed.tic('pose_estimation')

        cv2.circle(trajectory, (global_pose[1][0]+400, -global_pose[1][2]+500), radius=2, thickness=1, color=(0, 255, 0)) 
        modules.Utils.draw(original, histories[-5:])
        elapsed.tic('draw_result')

        logging.debug(filename)
        logging.info(histories[-1])
        logging.info('frame:%05d pose:(%s)', frame, ', '.join(['%.2f'%v for v in global_pose[1].T[0]]))
        logging.info(elapsed)
        
        if args.view:# and frame%100==0:
            cv2.imshow('image', original)
            cv2.imshow('trajectory', trajectory)
            cv2.waitKey(1)

if __name__ == '__main__':
    import argparse
    import ujson as json
    parser = argparse.ArgumentParser(description='visual odometry monocular')
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='config json file')
    parser.add_argument('-p', '--path', type=str, default='/data/private/storage/dataset/kitti/odometry/visual_odometry/dataset/sequences/00/')
    parser.add_argument('--view', action='store_true')

    parser.add_argument('--log-filename',   type=str, default='')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    level = logging.DEBUG if args.debug else logging.INFO
    log_format = '[%(asctime)s %(levelname)s] %(message)s'
    if not args.log_filename:
        logging.basicConfig(level=level, format=log_format, stream=sys.stderr)
    else:
        logging.basicConfig(level=level, format=log_format, filename=args.log_filename)
    logging.getLogger("requests").setLevel(logging.WARNING)

    with open(args.config, 'r') as f:
        configs = json.load(f)
    main(args, configs)

