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
    NUMBER_OF_FEATURES = configs['detector']['number_of_features']

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

        if len(histories) > 1:
            points, status = tracker.track(histories[-2].image, histories[-1].image, histories[-2].points)
            elapsed.tic('tracking')

            histories[-1].keypoints = tracker.update(histories[-2].keypoints, points, status)
            histories[-1].points = points[status>0]
            histories[-1].descriptions = histories[-2].descriptions[status>0]
            histories[-1].matches = np.array([[j, i] for i, j in enumerate(np.nonzero(status)[0].tolist())], dtype=np.int32)
            elapsed.tic('copying')

        if len(histories[-1].keypoints) < NUMBER_OF_FEATURES:
            keypoints = detector.detect(image)
            remain_counts = NUMBER_OF_FEATURES - len(histories[-1].keypoints)
            keypoints = sorted(keypoints, key=lambda x: x.response, reverse=True)[:remain_counts]

            keypoints, descriptions = detector.compute(image, keypoints)
            elapsed.tic('detection')

            histories[-1].add(keypoints, descriptions)
            elapsed.tic('appending')

        if len(histories) > 1:
            try:
                matches = histories[-1].matches
                prev_points = histories[-2].points[matches[:,0]]
                curr_points = histories[-1].points[matches[:,1]]
                essential, status = estimator.essential(prev_points, curr_points)
                histories[-1].matches = histories[-1].matches[status>0]

                matches = histories[-1].matches
                prev_points = histories[-2].points[matches[:,0]]
                curr_points = histories[-1].points[matches[:,1]]
                R, t, status = estimator.pose(prev_points, curr_points, essential)

                histories[-1].pose = [R, t]
                #histories[-1].matches = histories[-1].matches[status>0]
            except Exception as e:
                logging.warning(e)
        global_pose[1] = global_pose[1] + global_pose[0].dot( histories[-1].pose[1] )
        global_pose[0] = histories[-1].pose[0].dot( global_pose[0] ) 
        elapsed.tic('pose_estimation')

        if args.view:
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

