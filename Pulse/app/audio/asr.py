from faster_whisper import WhisperModel
import os

# Устройство: CPU
device = "cpu"

# Пути
DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data/input")
AUDIO_FILE = os.path.join(DATA_DIR, "test_audio.mp3")

# Инициализация модели
print("Загрузка модели Whisper...")
model = WhisperModel("small", device=device)  # small модель достаточно быстрая для CPU

def transcribe(audio_path):
    segments_generator, info = model.transcribe(audio_path)

    segments = []
    for segment in segments_generator:
        segments.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip()
        })

    return segments


if __name__ == "__main__":
    text = transcribe(AUDIO_FILE)
    print("\n=== Распознанный текст ===\n")
    print(text)