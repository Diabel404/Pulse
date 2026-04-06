import cv2
from ultralytics import YOLO
import os
import numpy as np

video_path = "lesson.mp4"
output_dir = "output/second_19_unique_people"
os.makedirs(output_dir, exist_ok=True)

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

target_second = 19
start_frame = int(fps * target_second)
end_frame = int(fps * (target_second + 1))

frame_idx = 0

# Функция подсчета уникальных людей через IoU
def count_unique_people(boxes, iou_thresh=0.4):
    # Оставляем только людей
    boxes = [b.xyxy[0].cpu().numpy() for b in boxes if int(b.cls) == 0]
    if len(boxes) == 0:
        return 0, []

    keep = []
    unique_boxes = []

    for i, box in enumerate(boxes):
        overlap = False
        for k in keep:
            xA = max(box[0], boxes[k][0])
            yA = max(box[1], boxes[k][1])
            xB = min(box[2], boxes[k][2])
            yB = min(box[3], boxes[k][3])
            interArea = max(0, xB - xA) * max(0, yB - yA)
            boxAArea = (box[2]-box[0])*(box[3]-box[1])
            boxBArea = (boxes[k][2]-boxes[k][0])*(boxes[k][3]-boxes[k][1])
            iou = interArea / float(boxAArea + boxBArea - interArea)
            if iou > iou_thresh:
                overlap = True
                break
        if not overlap:
            keep.append(i)
            unique_boxes.append(box)
    return len(unique_boxes), unique_boxes

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx < start_frame:
        frame_idx += 1
        continue
    if frame_idx >= end_frame:
        break

    results = model(frame, conf=0.3, iou=0.45)[0]

    people_count, unique_boxes = count_unique_people(results.boxes, iou_thresh=0.4)

    annotated_frame = frame.copy()
    for idx, box in enumerate(unique_boxes):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Person {idx+1}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Счётчик людей в углу
    cv2.putText(annotated_frame, f"People: {people_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    frame_filename = os.path.join(output_dir, f"frame_{frame_idx}.jpg")
    cv2.imwrite(frame_filename, annotated_frame)
    print(f"Сохранён кадр: {frame_filename} (уникальных людей: {people_count})")

    frame_idx += 1

cap.release()
print(f"Готово! Кадры из 19-й секунды с уникальными людьми сохранены в {output_dir}")
