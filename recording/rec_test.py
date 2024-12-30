import pyaudio
import wave


def record_audio(file_path, chunk_length, chunk_size, format, channels, rate):
    p = pyaudio.PyAudio()

    stream = p.open(format=format,
                    channels=channels,
                    rate=rate,
                    input=True,
                    frames_per_buffer=chunk_size)

    print("Recording...")

    frames = []

    for _ in range(0, int(rate / chunk_size * chunk_length)):
        data = stream.read(chunk_size)
        frames.append(data)

    print("Finished recording.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(file_path, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(format))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()


def main():
    chunk_length = 2  # seconds to record
    chunk_size = 1024  # number of frames per buffer
    format = pyaudio.paInt16  # audio format
    channels = 1  # mono
    rate = 16000  # Hertz

    record_audio("./audio/output.wav", chunk_length, chunk_size, format, channels, rate)


if __name__ == "__main__":
    main()
