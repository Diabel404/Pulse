import cv2
import json
import os
from ultralytics import YOLO
from tqdm import tqdm
import numpy as np

video_path = "lesson3.mp4"
output_json = "output/people_metrics.json"
output_frames_dir = "output/debug_frames"

os.makedirs(output_frames_dir, exist_ok=True)

model = YOLO("yolov8m.pt")

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

results_per_second = {}
FRAME_SKIP = 3
SAVE_SECONDS_LIMIT = 30  # сохраняем только первые 30 секунд
frame_count = 0

with tqdm(total=total_frames, desc="Processing video") as pbar:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % FRAME_SKIP != 0:
            frame_count += 1
            pbar.update(1)
            continue

        detections = model(
            frame,
            conf=0.45,
            iou=0.5,
            imgsz=960,
            verbose=False
        )[0]

        people_count = 0
        debug_frame = frame.copy()

        for box in detections.boxes:
            cls = int(box.cls[0])
            if cls == 0:
                people_count += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(debug_frame, f"{conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        current_second = int(frame_count / fps)

        # Отображение общего количества
        overlay_text = f"Total people: {people_count}"
        cv2.rectangle(debug_frame, (10, 10), (320, 60), (0, 0, 0), -1)
        cv2.putText(debug_frame, overlay_text, (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Сохраняем статистику
        if current_second not in results_per_second:
            results_per_second[current_second] = []

        results_per_second[current_second].append(people_count)

        # 🔥 Сохраняем кадры только первые 30 секунд
        if current_second < SAVE_SECONDS_LIMIT:
            frame_name = f"sec_{current_second}_frame_{frame_count}.jpg"
            cv2.imwrite(os.path.join(output_frames_dir, frame_name), debug_frame)

        frame_count += 1
        pbar.update(1)

cap.release()

# Усреднение по секундам
final_results = {}

for sec, counts in results_per_second.items():
    final_results[sec] = {
        "avg_people": int(np.mean(counts)),
        "max_people": int(np.max(counts)),
        "min_people": int(np.min(counts))
    }

with open(output_json, "w", encoding="utf-8") as f:
    json.dump(final_results, f, indent=2)

print(f"Метрики сохранены в {output_json}")
print(f"Кадры сохранены только для первых {SAVE_SECONDS_LIMIT} секунд")