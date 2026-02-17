# app/audio/speaker.py
import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path

encoder = VoiceEncoder()

def get_speaker_embedding(audio_path: str):
    """
    Возвращает эмбеддинг голоса (вектор numpy) для аудиофайла.
    """
    wav = preprocess_wav(Path(audio_path))
    embedding = encoder.embed_utterance(wav)
    return embedding
