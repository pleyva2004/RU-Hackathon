import sounddevice as sd
import soundfile as sf
import numpy as np
import time as t
from queue import Queue
import os

from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from threading import Thread

from faster_whisper import WhisperModel
import torch

import riva.client


# Use tiny model for fastest processing
torch.backends.cudnn.benchmark = True  # Enable CUDA optimization
model_size = "medium"
model = WhisperModel(model_size, 
                    device="cuda" ,
                    compute_type="float16",  # Use float16 for faster GPU processing
                    download_root=None)      # Specify model cache directory

# Load the translation model (supports many-to-many translation)
translation_model_name = "facebook/m2m100_418M"   # or "facebook/m2m100_1.2B" for better quality
translator = M2M100ForConditionalGeneration.from_pretrained(translation_model_name).half().cuda()  # .half() converts to float16
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
SILENCE_THRESHOLD = 0.0005859375232830644 # silence threshold to decide when to stop recording
SAMPLE_RATE = 16000  # Sampling rate in Hz
CHUNK_SIZE = 64    # Size of each audio chunk (in samples)
CHANNELS = 4
#FILENAME = f'./audio/live.wav'


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


def audio_preprocessing(audio_data, FILENAME):
    print("Audio preprocessing...")


    audio_amplitude = np.max(np.abs(audio_data))
    #print(f"Audio data shape: {audio_data.shape}, max value: {audio_amplitude}")
    
    # Skip processing if audio is too quiet
    #if audio_amplitude < 0.01:  # Adjust this threshold as needed
        #print(f"{YELLOW}Audio too quiet, skipping transcription{RESET_COLOR}")
        #return False
    # Normalize audio data
    audio_data = audio_data / audio_amplitude
    
    # Save to file
    sf.write(FILENAME, audio_data, SAMPLE_RATE)

    return True

def process_audio(file_name):

    #print("Processing audio...")

    try:
        segments, info = model.transcribe(file_name, 
                                          beam_size=1,
                                          vad_filter=True)

        for segment in segments:
            if segment.text.strip():

                # Start translation in parallel with typing
                translation = Thread(target=translate_audio, args=(segment.text,))
                translation.start()

                # Create typing animation effect
                for char in segment.text:
                    print(f"{CYAN}{char}{RESET_COLOR}", end='', flush=True)
                    t.sleep(0.025)  # Adjust this value to control typing speed
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
                    t.sleep(0.025)
                print()  # New line after translation
                # Add pause after translation before next segment
                t.sleep(0.5)

                #speaking.join()

        print()  # New line after all segments

    except Exception as e:
        print(f"{YELLOW}Error during transcription: {e}{RESET_COLOR}")
        print("file_name: ", file_name)
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
            max_length=128  # Limit max length
        ).to("cuda")  
        
        # Optimize generation parameters
        translated_tokens = translator.generate(
            **inputs,
            forced_bos_token_id=tokenizer.get_lang_id(target_lang),
            num_beams=2,          # Reduce beam size
            max_length=128,       # Limit output length
            early_stopping=True,  # Stop when possible
            no_repeat_ngram_size=2
        )
        
        # Move tokens back to CPU before decoding
        translated_text = tokenizer.batch_decode(translated_tokens.cpu(), skip_special_tokens=True)[0]
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

def record_audio(i):
    if i == 20:
        i = 0
 
    file_name = f"./audio/live{i}.wav"
    q = Queue()

    def callback(indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:  # Check for audio input errors
            print(f"{YELLOW}Audio input error: {status}{RESET_COLOR}")
            return
        
        if indata is None or len(indata) == 0:
            print(f"{YELLOW}Empty audio input received{RESET_COLOR}")
            return
        
        # Compute the maximum amplitude of the current audio chunk
        volume = np.max(np.abs(indata))
        
        if volume < SILENCE_THRESHOLD:
            #print("silence detected")
            # Store the current time when silence is first detected
            if not hasattr(callback, 'silence_start'):
                callback.silence_start = t.time()
            elif t.time() - callback.silence_start >= 1.0:
                #print("About to raise CallbackStop...")
                q.put(None) 
                raise sd.CallbackStop()
        else:
            #print("sound detected")
            # Reset silence detection if we detect sound
            if hasattr(callback, 'silence_start'):
                del callback.silence_start
        
        q.put(indata.copy())

    # Make sure the file is opened before recording anything:
    with sf.SoundFile(file_name, mode='w', samplerate=SAMPLE_RATE, channels=4) as file:
        with sd.InputStream(samplerate=SAMPLE_RATE,
                          channels=4, callback=callback):
            while True:
                data = q.get()
                if data is None:  # Check for exit signal
                    # Check if file size is bigger than 300KB
                    if os.path.getsize(file_name) > 400 * 1024:
                        process_thread = Thread(target=process_audio, args=(file_name,))
                        process_thread.start()
                        i += 1
                        record_audio(i)
                        process_thread.join()
                    else:
                        i += 1
                        record_audio(i)
                else:
                    file.write(data)             
          
def main2():
    try: 
        while True:
            i = 0

            if i == 6:
                i = 0
        
            FILENAME = f"./audio/live{i}.wav"
            q = Queue()

            def callback(indata, frames, time, status):
                """This is called (from a separate thread) for each audio block."""
                # Compute the maximum amplitude of the current audio chunk
                volume = np.max(np.abs(indata))
                
                if volume < SILENCE_THRESHOLD:
                    print("silence detected")
                    # Store the current time when silence is first detected
                    if not hasattr(callback, 'silence_start'):
                        callback.silence_start = t.time()
                    elif t.time() - callback.silence_start >= 1.5:
                        print("About to raise CallbackStop...")
                        q.put(None) 
                        raise sd.CallbackStop()
                else:
                    print("sound detected")
                    # Reset silence detection if we detect sound
                    if hasattr(callback, 'silence_start'):
                        del callback.silence_start
                
                q.put(indata.copy())

            # Make sure the file is opened before recording anything:
            with sf.SoundFile(FILENAME, mode='w', samplerate=SAMPLE_RATE, channels=4) as file:
                with sd.InputStream(samplerate=SAMPLE_RATE,
                                channels=4, callback=callback):
                    print('Recording...')
                    while True:
                        data = q.get()
                        if data is None:  # Check for exit signal
                            print('\nRecording finished due to silence detection')
                            print('Recording saved as: ' + repr(FILENAME))
                            i += 1
                            break
                        else:
                            file.write(data)
        
    except KeyboardInterrupt:
        print("\nTranscription stopped by user")
        return 0

def main():
    try:
        print(f"{PINK}Starting continuous transcription... Press Ctrl+C to stop.{RESET_COLOR}")
        record_audio(0) 
    except KeyboardInterrupt:
        print('\nRecording finished')
        return 0

if __name__ == "__main__":
    main()

# Lower the time between silence detection to 1 second
