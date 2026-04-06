import cv2
import numpy as np
import json
import os
from tqdm import tqdm

VIDEO_PATH = "lesson.mp4"
OUTPUT_PATH = "output/metrics.json"


def compute_energy(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Ошибка открытия видео")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = int(total_frames / fps)
    FRAME_SKIP = 3  # считать каждый 3-й кадр


    print(f"FPS: {fps}")
    print(f"Frames: {total_frames}")
    print(f"Duration (sec): {duration_sec}")

    ret, first_frame = cap.read()
    if not ret:
        print("Ошибка чтения первого кадра")
        return None

    prev_gray_full = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    h, w = prev_gray_full.shape

    # Берем нижнюю половину кадра (где ученики)
    prev_gray = prev_gray_full[int(h * 0.5):, :]

    energy_per_second = {}
    frame_count = 1

    for _ in tqdm(range(0, total_frames - 1, FRAME_SKIP)):
        ret, frame = cap.read()
        if not ret:
            break
        for _ in range(FRAME_SKIP - 1):
            cap.read()


        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = gray_full[int(h * 0.5):, :]

        # ===== Optical Flow =====
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            gray,
            None,
            0.5,   # pyr_scale
            3,     # levels
            15,    # winsize
            3,     # iterations
            5,     # poly_n
            1.2,   # poly_sigma
            0
        )

        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Средняя величина движения
        motion_mask = magnitude > 0.2
        if np.any(motion_mask):
            energy = np.mean(magnitude[motion_mask])
        else:
            energy = 0

        current_second = int(cap.get(cv2.CAP_PROP_POS_FRAMES) / fps)

        if current_second not in energy_per_second:
            energy_per_second[current_second] = []

        energy_per_second[current_second].append(float(energy))

        prev_gray = gray
        frame_count += 1

    cap.release()

    # ===== Усреднение по секундам =====
    raw_values = []

    for sec in sorted(energy_per_second.keys()):
        avg_energy = np.mean(energy_per_second[sec])
        raw_values.append(avg_energy)

    min_val = min(raw_values)
    max_val = max(raw_values)

    print(f"\nRaw energy min: {min_val}")
    print(f"Raw energy max: {max_val}")

    # ===== Нормализация =====
    normalized_values = []

    for val in raw_values:
        if max_val - min_val == 0:
            normalized = 0
        else:
            normalized = ((val - min_val) / (max_val - min_val)) * 100

        normalized_values.append(normalized)

    # ===== Сглаживание =====
    smoothed = []
    window_size = 5

    for i in range(len(normalized_values)):
        start = max(0, i - window_size // 2)
        end = min(len(normalized_values), i + window_size // 2 + 1)
        smoothed.append(np.mean(normalized_values[start:end]))

    result = []
    for sec, value in enumerate(smoothed):
        result.append({
            "second": sec,
            "energy": round(float(value), 2)
        })

    return result


def detect_low_activity(data, threshold=8, min_duration=120):
    low_periods = []
    start = None

    for point in data:
        if point["energy"] < threshold:
            if start is None:
                start = point["second"]
        else:
            if start is not None:
                duration = point["second"] - start
                if duration >= min_duration:
                    low_periods.append((start, point["second"]))
                start = None

    if start is not None:
        duration = data[-1]["second"] - start
        if duration >= min_duration:
            low_periods.append((start, data[-1]["second"]))

    return low_periods


def generate_recommendations(avg_energy, low_periods):
    recommendations = []

    if avg_energy < 10:
        recommendations.append(
            "Средняя активность низкая — попробуйте чаще вовлекать учеников вопросами."
        )

    if len(low_periods) > 0:
        recommendations.append(
            "Обнаружены длительные провалы внимания — рекомендуется смена формата активности."
        )

    if avg_energy > 70:
        recommendations.append(
            "Высокая активность — урок проходит динамично."
        )

    if not recommendations:
        recommendations.append(
            "Активность находится в нормальном диапазоне."
        )

    return recommendations


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def print_summary(data, low_periods, recommendations):
    energies = [d["energy"] for d in data]
    avg_energy = np.mean(energies)

    print("\n===== СВОДКА =====")
    print(f"Средняя энергия: {round(avg_energy, 2)}")
    print(f"Максимальная энергия: {round(max(energies), 2)}")
    print(f"Минимальная энергия: {round(min(energies), 2)}")

    print("\nПровалы активности (>120 сек):")
    if low_periods:
        for start, end in low_periods:
            print(f"С {start} по {end} сек")
    else:
        print("Не обнаружено")

    print("\nРЕКОМЕНДАЦИИ:")
    for r in recommendations:
        print("-", r)


if __name__ == "__main__":
    metrics = compute_energy(VIDEO_PATH)

    if metrics:
        save_json(metrics, OUTPUT_PATH)

        low_periods = detect_low_activity(metrics)
        avg_energy = np.mean([d["energy"] for d in metrics])
        recommendations = generate_recommendations(avg_energy, low_periods)

        print_summary(metrics, low_periods, recommendations)

        print(f"\nМетрики сохранены в {OUTPUT_PATH}")
