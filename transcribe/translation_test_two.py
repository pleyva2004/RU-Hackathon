from faster_whisper import WhisperModel
from transformers import MarianMTModel, MarianTokenizer
import torch
import time

# Load the transcription model
device = "cuda" if torch.cuda.is_available() else "cpu"
model_size = "large-v3"
whisper_model = WhisperModel(model_size, device=device, compute_type="float16")


# Load the translation model (French to English)
translation_model_name = "Helsinki-NLP/opus-mt-fr-en"
translator = MarianMTModel.from_pretrained(translation_model_name)
tokenizer = MarianTokenizer.from_pretrained(translation_model_name)

# Transcribe the audio (removed .en from model to support multiple languages)
audio_path = "./audio/french_song.webm"
segments, info = whisper_model.transcribe(audio_path, task="transcribe")

# Translate each transcribed segment to English
print("Transcription and Translation to English:")
start = time.time()
for segment in segments:
    # Get the transcribed text in source language
    source_text = segment.text

    # Translate text to English
    inputs = tokenizer(source_text, return_tensors="pt", padding=True)
    translated_tokens = translator.generate(**inputs)
    english_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

    # Print timestamp and translated text
    print(f"[{segment.start:.2f}s - {segment.end:.2f}s]: {english_text}")
end = time.time()
print("Translation time: ", end - start)
