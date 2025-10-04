import os
import time
import numpy as np
import cv2
import mediapipe as mp

from model.utils import hand_landmarks_to_features, sequence_buffer_append
from model.infer import load_model, predict_sequence

SEQ_LEN = int(os.environ.get("SEQ_LEN", 30))
NUM_FEATURES = int(os.environ.get("NUM_FEATURES", 63))
LABELS = os.environ.get("LABELS", "HELLO,THANKS,ILOVEYOU").split(",")
MODEL_WEIGHTS = os.environ.get("MODEL_WEIGHTS", "model/weights/best_model.pth")


def main():
    model = load_model(NUM_FEATURES, len(LABELS), MODEL_WEIGHTS)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError('Cannot open webcam')

    buf = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            landmarks = None
            if res.multi_hand_landmarks:
                hand = res.multi_hand_landmarks[0]
                landmarks = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in hand.landmark]
                mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            feat = hand_landmarks_to_features(landmarks)
            buf = sequence_buffer_append(buf, feat, SEQ_LEN)

            if len(buf) == SEQ_LEN:
                arr = np.stack(buf, axis=0).astype(np.float32)
                idx, probs = predict_sequence(model, arr)
                pred = LABELS[int(idx)]
                cv2.putText(frame, f"{pred}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            cv2.imshow('Realtime Inference', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()


if __name__ == '__main__':
    main()
