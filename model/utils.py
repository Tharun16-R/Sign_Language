import numpy as np


def hand_landmarks_to_features(landmarks):
    """
    landmarks: list of 21 dicts each with x,y,z (normalized)
    Returns 1D feature vector of length 63 (21*3). If landmarks is None, returns zeros.
    """
    F = 63
    if landmarks is None or len(landmarks) != 21:
        return np.zeros((F,), dtype=np.float32)
    arr = []
    for lm in landmarks:
        arr.extend([lm.get('x', 0.0), lm.get('y', 0.0), lm.get('z', 0.0)])
    return np.array(arr, dtype=np.float32)


def sequence_buffer_append(buf, feat, max_len):
    buf.append(feat)
    if len(buf) > max_len:
        buf.pop(0)
    return buf
