from __future__ import absolute_import
import logging

import numpy as np
import cv2

class Utils(object):
    @staticmethod
    def remove_outlier(pt1, pt2, status):
        status = status.reshape(status.shape[0])
        return [pt1[status>0], pt2[status>0]]

    @staticmethod
    def kp2np(keypoints):
        return np.array([kp.pt for kp in keypoints], dtype=np.float32)

    @staticmethod
    def draw(image, histories):
        num_histories = len(histories)-1
        for i, history in enumerate(histories[1:]):

            ratio = 1.0 * i / num_histories
            prev_points = Utils.kp2np([histories[i].keypoints[m[0]] for m in history.matches])
            curr_points = Utils.kp2np([     history.keypoints[m[1]] for m in history.matches])

            for pt1, pt2 in zip(prev_points, curr_points):
                cv2.line(image, tuple([int(v) for v in pt1]), tuple([int(v) for v in pt2]),
                    color=(0, 255*(ratio), 255*(1-ratio)), thickness=2)

        for kp in histories[-1].keypoints:
            pt = kp.pt
            cv2.circle(image, tuple([int(v) for v in pt]), radius=3,
                color=(255, 255, 255), thickness=-1)
            cv2.circle(image, tuple([int(v) for v in pt]), radius=2,
                color=(0, 0, 0), thickness=-1)
        for kp in [histories[-1].keypoints[m[1]] for m in histories[-1].matches]:
            pt = kp.pt
            cv2.circle(image, tuple([int(v) for v in pt]), radius=3,
                color=(0, 255, 0), thickness=-1)
            cv2.circle(image, tuple([int(v) for v in pt]), radius=2,
                color=(0, 0, 0), thickness=-1)


