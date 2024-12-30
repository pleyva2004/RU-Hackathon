import sounddevice as sd
import soundfile as sf
import numpy as np
import time

from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from threading import Thread

from faster_whisper import WhisperModel
import torch

import riva.client


# Use tiny model for fastest processing
model_size = "medium"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = WhisperModel(model_size, device=device, compute_type="float16")

# Load the translation model (supports many-to-many translation)
translation_model_name = "facebook/m2m100_1.2B"  # or "facebook/m2m100_1.2B" for better quality
translator = M2M100ForConditionalGeneration.from_pretrained(translation_model_name)
tokenizer = M2M100Tokenizer.from_pretrained(translation_model_name)
target_lang = "en"
list_translated_text = []
translated_text = ""

# Initialize Riva client
auth = riva.client.Auth(uri='localhost:50051')
riva_tts = riva.client.SpeechSynthesisService(auth)

 #Setting up TTS Request    
sample_rate_hz = 44100
language_code = "en-US"
voice_name = "English-US.Male-Happy"

# Settings for recording
samplerate = 16000
duration = 3
channels = 4
filename = f'./audio/live.wav'
recording_queue = []


# ANSI escape codes for colors
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

# Dictionary of supported M2M100 language codes
LANGUAGE_CODES = {
    'af': 'Afrikaans',
    'am': 'Amharic',
    'ar': 'Arabic',
    'ast': 'Asturian',
    'az': 'Azerbaijani',
    'ba': 'Bashkir',
    'be': 'Belarusian',
    'bg': 'Bulgarian',
    'bn': 'Bengali',
    'br': 'Breton',
    'bs': 'Bosnian',
    'ca': 'Catalan',
    'ceb': 'Cebuano',
    'cs': 'Czech',
    'cy': 'Welsh',
    'da': 'Danish',
    'de': 'German',
    'el': 'Greek',
    'en': 'English',
    'es': 'Spanish',
    'et': 'Estonian',
    'fa': 'Persian',
    'ff': 'Fulah',
    'fi': 'Finnish',
    'fr': 'French',
    'fy': 'Western Frisian',
    'ga': 'Irish',
    'gd': 'Scottish Gaelic',
    'gl': 'Galician',
    'gu': 'Gujarati',
    'ha': 'Hausa',
    'he': 'Hebrew',
    'hi': 'Hindi',
    'hr': 'Croatian',
    'ht': 'Haitian',
    'hu': 'Hungarian',
    'hy': 'Armenian',
    'id': 'Indonesian',
    'ig': 'Igbo',
    'ilo': 'Iloko',
    'is': 'Icelandic',
    'it': 'Italian',
    'ja': 'Japanese',
    'jv': 'Javanese',
    'ka': 'Georgian',
    'kk': 'Kazakh',
    'km': 'Khmer',
    'kn': 'Kannada',
    'ko': 'Korean',
    'lb': 'Luxembourgish',
    'lg': 'Ganda',
    'ln': 'Lingala',
    'lo': 'Lao',
    'lt': 'Lithuanian',
    'lv': 'Latvian',
    'mg': 'Malagasy',
    'mk': 'Macedonian',
    'ml': 'Malayalam',
    'mn': 'Mongolian',
    'mr': 'Marathi',
    'ms': 'Malay',
    'my': 'Burmese',
    'ne': 'Nepali',
    'nl': 'Dutch',
    'no': 'Norwegian',
    'ns': 'Northern Sotho',
    'oc': 'Occitan',
    'or': 'Oriya',
    'pa': 'Punjabi',
    'pl': 'Polish',
    'ps': 'Pashto',
    'pt': 'Portuguese',
    'ro': 'Romanian',
    'ru': 'Russian',
    'sd': 'Sindhi',
    'si': 'Sinhala',
    'sk': 'Slovak',
    'sl': 'Slovenian',
    'so': 'Somali',
    'sq': 'Albanian',
    'sr': 'Serbian',
    'ss': 'Swati',
    'su': 'Sundanese',
    'sv': 'Swedish',
    'sw': 'Swahili',
    'ta': 'Tamil',
    'th': 'Thai',
    'tl': 'Tagalog',
    'tn': 'Tswana',
    'tr': 'Turkish',
    'uk': 'Ukrainian',
    'ur': 'Urdu',
    'uz': 'Uzbek',
    'vi': 'Vietnamese',
    'wo': 'Wolof',
    'xh': 'Xhosa',
    'yi': 'Yiddish',
    'yo': 'Yoruba',
    'zh': 'Chinese',
    'zu': 'Zulu'
}


def audio_preprocessing(audio_data):
    #print("Audio preprocessing...")


    audio_amplitude = np.max(np.abs(audio_data))
    #print(f"Audio data shape: {audio_data.shape}, max value: {audio_amplitude}")
    
    # Skip processing if audio is too quiet
    if audio_amplitude < 0.01:  # Adjust this threshold as needed
        #print(f"{YELLOW}Audio too quiet, skipping transcription{RESET_COLOR}")
        return False
    # Normalize audio data
    audio_data = audio_data / audio_amplitude
    
    # Save to file
    sf.write(filename, audio_data, samplerate)

    return True

def process_audio():

    #print("Processing audio...")

    try:
        segments, info = model.transcribe(filename, 
                                          beam_size=1,
                                          vad_filter=True,
                                          vad_parameters=dict(min_silence_duration_ms=250))

        for segment in segments:
            if segment.text.strip():

                # Start translation in parallel with typing
                translation = Thread(target=translate_audio, args=(segment.text,))
                translation.start()

                # Create typing animation effect
                for char in segment.text:
                    print(f"{CYAN}{char}{RESET_COLOR}", end='', flush=True)
                    time.sleep(0.05)  # Adjust this value to control typing speed
                print(" ", end='', flush=True)  # Add space between segments

                # Wait for translation to complete
                translation.join()

                # Create typing animation effect for translated text
                global translated_text
                translated_text = list_translated_text.pop()


                #speaking = Thread(target=speak_translation, args=(translated_text,))
                #speaking.start()

                for char in translated_text:
                    print(f"{NEON_GREEN}{char}{RESET_COLOR}", end='', flush=True)
                    time.sleep(0.05)
                print()  # New line after translation

                #speaking.join()

        print()  # New line after all segments

    except Exception as e:
        print(f"{YELLOW}Error during transcription: {e}{RESET_COLOR}")
        return 

def translate_audio(text):
    try:
        
        # Skip translation for empty or very short text
        if not text or len(text.strip()) < 2:
            return None

        # Optimize tokenizer settings
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128  # Limit max length for faster processing
        )
        
        # Optimize generation parameters
        translated_tokens = translator.generate(
            **inputs,
            forced_bos_token_id=tokenizer.get_lang_id(target_lang),
            num_beams=2,          # Reduce beam size (default is usually 4 or 5)
            max_length=128,       # Limit output length
            early_stopping=True,  # Stop when possible
            no_repeat_ngram_size=2
        )
        
        translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    
        list_translated_text.append(translated_text)

    except Exception as e:
        print(f"{YELLOW}Translation error: {e}{RESET_COLOR}")
        return None

def speak_translation(text):
    try:
        req = { 
            "language_code"  : language_code,
            "encoding"       : riva.client.AudioEncoding.LINEAR_PCM ,   # LINEAR_PCM and OGGOPUS encodings are supported
            "sample_rate_hz" : sample_rate_hz,                          # Generate 44.1KHz audio
            "voice_name"     : voice_name                                # The name of the voice to generate
        }
       
        req["text"] = text
        resp = riva_tts.synthesize(**req)
        audio_samples = np.frombuffer(resp.audio, dtype=np.int16)
        #ipd.Audio(audio_samples, rate=sample_rate_hz)
    
        # Play audio
        sd.play(audio_samples, sample_rate_hz)
        sd.wait()  # Wait until audio finishes playing

    except Exception as e:
        print(f"{YELLOW}TTS error: {e}{RESET_COLOR}")

def list_audio_devices():
    devices = sd.query_devices()
    print("\nAvailable audio devices:")
    for i, device in enumerate(devices):
        print(f"{i}: {device['name']}")
        print(f"    Input channels: {device['max_input_channels']}")
        print(f"    Output channels: {device['max_output_channels']}")
        print(f"    Default samplerate: {device['default_samplerate']}")
        print()
    
    while True:
        try:
            device_id = int(input("Select audio device by number: "))
            if 0 <= device_id < len(devices):
                print(f"Selected device: {devices[device_id]['name']}")
                return device_id
            else:
                print("Invalid device number. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

def record_audio():
    mydata = sd.rec(int(samplerate * duration),
                        samplerate=samplerate,
                        channels=channels,
                        blocking=True)
    recording_queue.append(mydata)

def main():
    try:
        print(f"{PINK}Starting continuous transcription... Press Ctrl+C to stop.{RESET_COLOR}")
        # Get first recording directly
        record_audio()
      
        while True:
            
            # Start next recording immediately
            recording_thread = Thread(target=record_audio)
            recording_thread.start()

            # Process the previous recording while the new one is happening
            audio = recording_queue.pop(0)  # Remove and get the recording
            if audio_preprocessing(audio):   # Process it, discard if too quiet
                process_audio()

            recording_thread.join()

    except KeyboardInterrupt:
        print("\nTranscription stopped by user")


if __name__ == "__main__":
    main()