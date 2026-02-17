from app.audio import asr
from app.utils import stats

if __name__ == "__main__":
    audio_path = "data/input/test_audio.mp3"

    print("=== Распознаем речь ===")
    segments = asr.transcribe(audio_path)
    
    print("\n=== Текст ===\n", segments)

    print("\n=== Статистика ===")
    print("Вопросов:", stats.count_questions(segments))
    print("Всего слов:", stats.count_words(segments))
    print("TTR:", stats.estimate_ttr(segments))