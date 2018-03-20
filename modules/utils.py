from __future__ import absolute_import

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
    def nonmax_supression(points, scale=5.0):
        hashed = {hash( tuple([int(p/scale+0.5) for p in pt]) ):i for i, pt in enumerate(points)}
        return np.array([points[i] for k, i in hashed.items()])

    @staticmethod
    def append(points1, points2):
        if len(points1) == 0:
            return points2
        if len(points2) == 0:
            return points1
        return np.concatenate((points1, points2), axis=0)

    @staticmethod
    def draw(image, histories):
        num_histories = len(histories)
        for i, history in enumerate(histories[1:]):
            ratio = 1.0 * i / num_histories
            matches = history.matches
            for pt1, pt2 in zip(histories[i].points[matches[:,0]], history.points[matches[:,1]]):
                cv2.line(image, tuple([int(v) for v in pt1]), tuple([int(v) for v in pt2]),
                    color=(0, 255*(ratio), 255*(1-ratio)), thickness=2)
                
        for pt in histories[-1].points:
            cv2.circle(image, tuple([int(v) for v in pt]), radius=3,
                color=(255, 255, 255), thickness=-1)
            cv2.circle(image, tuple([int(v) for v in pt]), radius=2,
                color=(0, 0, 0), thickness=-1)
        if len(histories[-1].matches) == 0:
            return
        for pt in histories[-1].points[histories[-1].matches[:,1]]:
            cv2.circle(image, tuple([int(v) for v in pt]), radius=3,
                color=(0, 255, 0), thickness=-1)
            cv2.circle(image, tuple([int(v) for v in pt]), radius=2,
                color=(0, 0, 0), thickness=-1)

