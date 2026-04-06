import cv2
import os
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

# ================= НАСТРОЙКИ =================

video_path = "lesson3.mp4"
output_frames_dir = "output/pose_debug"

os.makedirs(output_frames_dir, exist_ok=True)

FRAME_SKIP = 3
PROCESS_SECONDS_LIMIT = 35   # 🔥 Обрабатываем только первые 30 секунд
SAVE_SECONDS_LIMIT = 35      # Сохраняем кадры только первые 30 секунд

# =============================================

model = YOLO("yolov8m-pose.pt")

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

frame_count = 0

# 🔥 Ограничение по кадрам (до 30 секунд)
max_frames = int(fps * PROCESS_SECONDS_LIMIT)

with tqdm(total=min(total_frames, max_frames), desc="Processing video") as pbar:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 🔥 ОСТАНОВКА ПОСЛЕ 30 СЕКУНД
        if frame_count > max_frames:
            break

        if frame_count % FRAME_SKIP != 0:
            frame_count += 1
            pbar.update(1)
            continue

        results = model(frame, conf=0.4, imgsz=960, verbose=False)[0]

        debug_frame = frame.copy()
        total_people = 0
        raised_hands = 0

        if results.keypoints is not None:

            keypoints = results.keypoints.xy.cpu().numpy()

            for person_kp in keypoints:

                total_people += 1

                left_shoulder = person_kp[5]
                right_shoulder = person_kp[6]
                left_elbow = person_kp[7]
                right_elbow = person_kp[8]
                left_wrist = person_kp[9]
                right_wrist = person_kp[10]

                # Высота человека
                min_y = np.min(person_kp[:, 1])
                max_y = np.max(person_kp[:, 1])
                person_height = max_y - min_y

                threshold = 20

                left_raised = False
                right_raised = False

                # ---- Левая рука ----
                if (
                    left_wrist[1] < left_shoulder[1] - threshold and
                    left_elbow[1] < left_shoulder[1] and
                    np.linalg.norm(left_wrist - left_shoulder) > 0.4 * person_height
                ):
                    left_raised = True

                # ---- Правая рука ----
                if (
                    right_wrist[1] < right_shoulder[1] - threshold and
                    right_elbow[1] < right_shoulder[1] and
                    np.linalg.norm(right_wrist - right_shoulder) > 0.4 * person_height
                ):
                    right_raised = True

                if left_raised or right_raised:
                    raised_hands += 1

                    # Подсветка поднятой руки (красным)
                    cv2.circle(debug_frame, (int(left_wrist[0]), int(left_wrist[1])), 8, (0, 0, 255), -1)
                    cv2.circle(debug_frame, (int(right_wrist[0]), int(right_wrist[1])), 8, (0, 0, 255), -1)

                # Рисуем ключевые точки
                for x, y in person_kp:
                    cv2.circle(debug_frame, (int(x), int(y)), 3, (0, 255, 0), -1)

        # ==== Текст сверху ====
        overlay_text1 = f"People: {total_people}"
        overlay_text2 = f"Raised hands: {raised_hands}"

        cv2.rectangle(debug_frame, (10, 10), (450, 90), (0, 0, 0), -1)
        cv2.putText(debug_frame, overlay_text1, (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(debug_frame, overlay_text2, (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        current_second = int(frame_count / fps)

        # Сохраняем только первые 30 секунд
        if current_second < SAVE_SECONDS_LIMIT:
            frame_name = f"sec_{current_second}_frame_{frame_count}.jpg"
            cv2.imwrite(os.path.join(output_frames_dir, frame_name), debug_frame)

        frame_count += 1
        pbar.update(1)

cap.release()

print("Pose-анализ завершен.")