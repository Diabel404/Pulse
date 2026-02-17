def count_questions(segments):
    """
    Считает количество вопросов в аудио по символу '?'
    """
    if not segments:
        return 0

    total = 0
    for seg in segments:
        text = seg.get("text", "")
        total += text.count("?")
    return total


def count_words(segments):
    """
    Общее количество слов
    """
    if not segments:
        return 0

    total = 0
    for seg in segments:
        text = seg.get("text", "")
        total += len(text.split())
    return total


def estimate_ttr(segments, teacher_segments=None):
    """
    Грубая оценка Teacher Talk Ratio (TTR)
    """
    if not teacher_segments:
        return None

    teacher_words = count_words(teacher_segments)
    total_words = count_words(segments)

    if total_words == 0:
        return 0

    return round((teacher_words / total_words) * 100, 2)