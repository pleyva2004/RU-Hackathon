import sounddevice as sd
import soundfile as sf

# Reduce duration for faster processing
samplerate = 16000
duration = 3
channels = 1
filename = './audio/output_two.wav'


try:
    print("Starting continuous transcription... Press Ctrl+C to stop.")
    while True:
        # Record audio (this blocks for 'duration' seconds)
        mydata = sd.rec(int(samplerate * duration),
                    samplerate=samplerate,
                    channels=channels,
                    blocking=True)
        
except KeyboardInterrupt:
    print("\nTranscription stopped by user")