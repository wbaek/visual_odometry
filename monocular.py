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

        if len(histories) >= 2:
            prev_points = histories[-2].points
            curr_points, status = tracker.track(histories[-2].image, histories[-1].image, prev_points)
            keypoints = tracker.update(histories[-2].keypoints, curr_points)
            descriptions = histories[-2].descriptions
            histories[-1].set_matches(keypoints, curr_points, descriptions, status)
            prev_points, curr_points = modules.Utils.remove_outlier(prev_points, curr_points, status)
            elapsed.tic('tracking')

            try:
                essential, status = estimator.essential(curr_points, prev_points)
                histories[-1].remove_outlier(status)
                prev_points, curr_points = modules.Utils.remove_outlier(prev_points, curr_points, status)

                R, t, status = estimator.pose(curr_points, prev_points, essential)
                histories[-1].remove_outlier(status)
                prev_points, curr_points = modules.Utils.remove_outlier(prev_points, curr_points, status)

                histories[-1].pose = [R, t]
            except ValueError as e:
                logging.warning(e)

            global_pose[1] = global_pose[1] + global_pose[0].dot( histories[-1].pose[1] )
            global_pose[0] = histories[-1].pose[0].dot( global_pose[0] ) 
            elapsed.tic('pose_estimation')

        keypoints = detector.detect(image)
        keypoints, descriptions = detector.compute(image, keypoints)
        elapsed.tic('detection')

        histories[-1].add(keypoints, descriptions)
        elapsed.tic('appending')

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

