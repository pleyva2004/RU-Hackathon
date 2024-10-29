import sounddevice as sd
import soundfile as sf

samplerate = 44100  # Hertz
duration = 5  # seconds
channels = 1  # mono
filename = 'output_two.wav'


print("start recording")
mydata = sd.rec(int(samplerate * duration),
                samplerate=samplerate,
                channels=channels,
                blocking=True)
print("end recording")

sf.write(filename, mydata, samplerate)