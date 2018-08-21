import numpy as np


def judge_side_face(facial_landmarks):
    wide_dist = np.linalg.norm(facial_landmarks[0] - facial_landmarks[1])
    high_dist = np.linalg.norm(facial_landmarks[0] - facial_landmarks[3])
    dist_rate = high_dist / wide_dist

    # cal std
    vec_A = facial_landmarks[0] - facial_landmarks[2]
    vec_B = facial_landmarks[1] - facial_landmarks[2]
    vec_C = facial_landmarks[3] - facial_landmarks[2]
    vec_D = facial_landmarks[4] - facial_landmarks[2]
    dist_A = np.linalg.norm(vec_A)
    dist_B = np.linalg.norm(vec_B)
    dist_C = np.linalg.norm(vec_C)
    dist_D = np.linalg.norm(vec_D)

    # cal rate
    high_rate = dist_A / dist_C
    width_rate = dist_C / dist_D
    high_ratio_variance = np.fabs(high_rate - 1.1)  # smaller is better
    width_ratio_variance = np.fabs(width_rate - 1)

    return dist_rate, high_ratio_variance, width_ratio_variance
