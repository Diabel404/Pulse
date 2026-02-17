from pyannote.audio import Pipeline

class DiarizationProcessor:
    def __init__(self, hf_token=None):
        """
        hf_token: токен HuggingFace, если требуется для загрузки модели
        """
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization",
            use_auth_token=hf_token
        )

    def diarize(self, audio_path):
        """
        Возвращает список сегментов с указанием спикера
        """
        diarization = self.pipeline(audio_path)
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker
            })
        return segments
