import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel
import torch
import threading
from queue import Queue
import numpy as np

# Use tiny model for fastest processing
model_size = "tiny"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = WhisperModel(model_size, device=device, compute_type="float16")

# Reduce duration for faster processing
samplerate = 16000
duration = 3
channels = 1
filename = './audio/output_two.wav'

# Create a queue for audio chunks
audio_queue = Queue()

# Add new queue for playback
playback_queue = Queue()

def process_audio():
    while True:
        try:
            audio_data = audio_queue.get()
            if audio_data is None:  # Sentinel value to stop the thread
                break
            
            # Add audio preprocessing
            audio_amplitude = np.max(np.abs(audio_data))
            #print(f"Audio data shape: {audio_data.shape}, max value: {audio_amplitude}")
            
            # Skip processing if audio is too quiet
            if audio_amplitude < 0.01:  # Adjust this threshold as needed
                #print("Audio too quiet, skipping transcription")
                continue
                
            # Normalize audio data
            audio_data = audio_data / audio_amplitude
            
            # Save to file
            sf.write(filename, audio_data, samplerate)
            
            try:
                # Transcribe with adjusted parameters
                segments, info = model.transcribe(filename, 
                                              beam_size=1,
                                              vad_filter=True,
                                              vad_parameters=dict(min_silence_duration_ms=250))  # Reduced silence threshold
                
                # Convert segments to list and check content
                segments_list = list(segments)
                print(f"Number of segments detected: {len(segments_list)}")
                
                if segments_list:  # Only process if segments are not empty
                    for segment in segments_list:
                        print(segment.text)
                else:
                    print("No speech detected in this chunk")
                    
            except Exception as e:
                print(f"Error during transcription: {e}")
                continue
            
        except Exception as e:
            print(f"Error in processing: {e}")
            continue

def play_audio():
    while True:
        try:
            audio_data = playback_queue.get()
            if audio_data is None:  # Sentinel value to stop the thread
                break
            
              # Add audio preprocessing
            audio_amplitude = np.max(np.abs(audio_data))
            #print(f"Audio data shape: {audio_data.shape}, max value: {audio_amplitude}")
            
            # Skip processing if audio is too quiet
            if audio_amplitude < 0.01:  # Adjust this threshold as needed
                #print("Audio too quiet, skipping transcription")
                continue
                
            # Normalize audio data
            audio_data = audio_data / audio_amplitude
            
            # Play the audio data
            sd.play(audio_data, samplerate)
            sd.wait()  # Wait until playback is finished
            
        except Exception as e:
            print(f"Error in playback: {e}")
            continue

# Update thread creation
process_thread = threading.Thread(target=process_audio)
process_thread.start()
playback_thread = threading.Thread(target=play_audio)
playback_thread.start()

try:
    print("Starting continuous transcription... Press Ctrl+C to stop.")
    while True:
        # Record audio (this blocks for 'duration' seconds)
        mydata = sd.rec(int(samplerate * duration),
                    samplerate=samplerate,
                    channels=channels,
                    blocking=True)
        
        # Add the recorded data to both queues
        audio_queue.put(mydata)
        playback_queue.put(mydata)

except KeyboardInterrupt:
    print("\nTranscription stopped by user")
    # Signal both threads to stop
    audio_queue.put(None)
    playback_queue.put(None)
    process_thread.join()
    playback_thread.join()
