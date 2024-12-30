import time
from faster_whisper import WhisperModel
import torch


start = time.time()

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model
# model_size = "tiny.en"
model_size = "large-v3"

# Run on GPU with FP16
model = WhisperModel(model_size, device=device, compute_type="float16")

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")

audio_path = "./audio/GodVideo.wav"
video_path = "./audio/french_song.webm"
segments, info = model.transcribe(video_path, beam_size=5)

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

end = time.time()
print("Translation time: ", end - start)
