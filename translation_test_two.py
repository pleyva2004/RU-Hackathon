from faster_whisper import WhisperModel
from transformers import MarianMTModel, MarianTokenizer
import torch

# Load the transcription model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
whisper_model = WhisperModel("medium.en", device=device, compute_type="float16")


# Load the translation model (English to Spanish)
translation_model_name = "Helsinki-NLP/opus-mt-en-es"
translator = MarianMTModel.from_pretrained(translation_model_name)
tokenizer = MarianTokenizer.from_pretrained(translation_model_name)

# Transcribe the English audio
audio_path = "./audio/chunk.wav"
segments, info = whisper_model.transcribe(audio_path, task="transcribe")

# Translate each transcribed segment from English to Spanish
print("Transcription and Translation to Spanish:")
for segment in segments:
    # Get the transcribed text in English
    english_text = segment.text

    # Translate text to Spanish
    inputs = tokenizer(english_text, return_tensors="pt", padding=True)
    translated_tokens = translator.generate(**inputs)
    spanish_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

    # Print timestamp and translated text
    print(f"[{segment.start:.2f}s - {segment.end:.2f}s]: {spanish_text}")
