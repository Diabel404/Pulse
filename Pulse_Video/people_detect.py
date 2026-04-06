import cv2
import json
from ultralytics import YOLO
from tqdm import tqdm
import numpy as np

video_path = "lesson.mp4"
output_json = "output/people_metrics.json"

# Более точная модель
model = YOLO("yolov8m.pt")

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

results_per_second = {}
FRAME_SKIP = 3
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

        # Детекция
        detections = model(
            frame,
            conf=0.45,      # жестче фильтр
            iou=0.5,        # лучшее подавление дублей
            imgsz=960,      # лучше для плотных сцен
            verbose=False
        )[0]

        people_count = 0

        for box in detections.boxes:
            cls = int(box.cls[0])
            if cls == 0:  # 0 = person
                people_count += 1

        current_second = int(frame_count / fps)

        if current_second not in results_per_second:
            results_per_second[current_second] = []

        results_per_second[current_second].append(people_count)

        frame_count += 1
        pbar.update(1)

cap.release()

# Усредняем по секундам
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