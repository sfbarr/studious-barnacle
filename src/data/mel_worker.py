import os
import numpy as np
import librosa

SR = 22050
DURATION = 30.0
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
N_FRAMES = int(np.ceil(DURATION * SR / HOP_LENGTH))  # 1292


def process_track(args: tuple):
    """Load one MP3, return (track_id, log_mel (1, n_mels, T)) or None."""
    tid, audio_dir = args
    tid_str = f"{tid:06d}"
    path = os.path.join(audio_dir, tid_str[:3], f"{tid_str}.mp3")
    if not os.path.exists(path):
        return None
    try:
        y, _ = librosa.load(path, sr=SR, duration=DURATION, mono=True)
        target_samples = int(SR * DURATION)
        if len(y) < target_samples:
            y = np.pad(y, (0, target_samples - len(y)))
        mel = librosa.feature.melspectrogram(
            y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)
        if log_mel.shape[1] > N_FRAMES:
            log_mel = log_mel[:, :N_FRAMES]
        elif log_mel.shape[1] < N_FRAMES:
            log_mel = np.pad(log_mel, ((0, 0), (0, N_FRAMES - log_mel.shape[1])))
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)
        return tid, log_mel[np.newaxis].astype(np.float32)
    except Exception:
        return None
