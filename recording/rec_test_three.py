import os
import sounddevice as sd
import soundfile as sf
import numpy as np
from queue import Queue
import time as t

# Settings
SAMPLE_RATE = 16000  # Sampling rate in Hz
CHUNK_SIZE = 64    # Size of each audio chunk (in samples)


# Function to determine silence threshold based on background noise
def get_silence_threshold(duration=1, multiplier=1.2):
    print("Recording background noise...")
    noise_sample = sd.rec(int(SAMPLE_RATE * duration), samplerate=SAMPLE_RATE, channels=4, dtype='float32')
    sd.wait()
    noise_max_amplitude = np.max(np.abs(noise_sample))
    threshold = noise_max_amplitude * multiplier
    print(f"Silence Threshold: {threshold}")
    return threshold

# Set the silence threshold
SILENCE_THRESHOLD = 0.0005859375232830644


def record_audio(i):
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
        
        # Apply some initial processing to the audio chunk
        processed_data = indata.copy()
        # Apply pre-emphasis to enhance high frequencies (makes voice clearer)
        #processed_data[1:] = processed_data[1:] - 0.97 * processed_data[:-1]
        
        q.put(processed_data)

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
                    record_audio(i)
                else:
                    file.write(data)

def test_recording():
    try:
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
                        return 0
                    file.write(data)  
       
    except KeyboardInterrupt:
        print('\nRecording finished: ' + repr(FILENAME))
        return 0

def main():
    if os.path.getsize("./audio/live6.wav") < 250 * 1024:
        print("File is too small")
        return 0
    try:
        record_audio(2) 
    except KeyboardInterrupt:
        print('\nRecording finished')
        return 0

if __name__ == "__main__":
    main()



