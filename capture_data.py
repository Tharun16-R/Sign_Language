import os
import argparse
import time
import numpy as np
import cv2
import mediapipe as mp

from model.utils import hand_landmarks_to_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', required=True, help='Label/class name')
    parser.add_argument('--sequences', type=int, default=30)
    parser.add_argument('--frames', type=int, default=30)
    parser.add_argument('--out_dir', default='data/processed')
    args = parser.parse_args()

    save_dir = os.path.join(args.out_dir, args.label)
    os.makedirs(save_dir, exist_ok=True)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError('Cannot open webcam')

    try:
        for seq_idx in range(args.sequences):
            buf = []
            print(f"Recording sequence {seq_idx+1}/{args.sequences} for label '{args.label}'...")
            time.sleep(1.0)
            for f in range(args.frames):
                ret, frame = cap.read()
                if not ret:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = hands.process(rgb)

                landmarks = None
                if res.multi_hand_landmarks:
                    # use first hand
                    hand = res.multi_hand_landmarks[0]
                    landmarks = [
                        {"x": lm.x, "y": lm.y, "z": lm.z} for lm in hand.landmark
                    ]
                    # draw
                    mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                feat = hand_landmarks_to_features(landmarks)
                buf.append(feat)

                cv2.putText(frame, f"{args.label} seq {seq_idx+1}/{args.sequences} frame {f+1}/{args.frames}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.imshow('Capture', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            arr = np.stack(buf, axis=0).astype(np.float32)
            out_path = os.path.join(save_dir, f"{seq_idx:04d}.npz")
            np.savez_compressed(out_path, arr)
            print(f"Saved {out_path}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()


if __name__ == '__main__':
    main()
