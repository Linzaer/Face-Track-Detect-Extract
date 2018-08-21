"""
@author: Mahmoud I.Zidan
"""
'''
its purpose is to get the ground truth detection positions per frame.
specifically for Oxford TownCentre dataset
(http://www.robots.ox.ac.uk/~lav/Research/Projects/2009bbenfold_headpose/project.html)

Data format:
personNumber, frameNumber, headValid, bodyValid, headLeft, headTop, headRight, headBottom, bodyLeft, bodyTop, bodyRight, bodyBottom

Note: we ignore using/tracking head detection data
'''

import numpy as np


class GroundTruthDetections:

    def __init__(self, fname='TownCentre-groundtruth.top'):
        self.all_dets = np.loadtxt(fname, delimiter=',')  # load detections
        self._frames = int(self.all_dets[:, 1].max()) + 1  # 0 to 4500 inclusive

    '''as in practical realtime MOT, the detector doesn't run on every single frame'''

    def _do_detection(self, detect_prob=.4):

        return int(np.random.choice(2, 1, p=[1 - detect_prob, detect_prob]))

    '''returns the detected items positions or [] if no detection'''

    def get_detected_items(self, frame):

        if frame % 1 == 0:
            # if self._do_detection() or frame == 0:
            return self.all_dets[self.all_dets[:, 1] == frame, 8:]

        else:
            return []

        # return self.all_dets[self.all_dets[:, 1] == frame, 8:]

    def get_total_frames(self):
        return self._frames
