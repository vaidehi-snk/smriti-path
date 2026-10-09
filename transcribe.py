import sys
import subprocess
import numpy as np
import imageio_ffmpeg
from faster_whisper import WhisperModel

def load_audio(path):
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-i", path,
           "-f", "f32le", "-ac", "1", "-ar", "16000", "-"]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print(result.stderr.decode(errors="ignore")[-800:])
        sys.exit("ffmpeg could not read the file (check the filename).")
    return np.frombuffer(result.stdout, dtype=np.float32)

audio_path = sys.argv[1] if len(sys.argv) > 1 else "clip.m4a"
audio = load_audio(audio_path)

model = WhisperModel("small", device="cpu", compute_type="int8")
segments, info = model.transcribe(audio, vad_filter=True)
print("Detected:", info.language, round(info.language_probability, 2))

text = " ".join(s.text.strip() for s in segments)
open("transcript.txt", "w", encoding="utf-8").write(text)
print(text)