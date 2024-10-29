import pyaudio
import wave
from faster_whisper import WhisperModel


def record_audio(file_path, p, stream, rate, chunk_size, format, channels, chunk_length):
    frames = []

    for _ in range(0, int(rate / chunk_size * chunk_length)):
        data = stream.read(chunk_size)
        frames.append(data)

    wf = wave.open(file_path, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(format))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()


def transcribe_audio(model, file_path):
    segments, info = model.transcribe(file_path, task='translate',
                                      beam_size=5,
                                      multilingual=True,
                                      target_language='en')

    # print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    for segment in segments:
        # print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        return segment.text


def main():
    # Set settings for recording
    chunk_length = 1  # seconds to record
    chunk_size = 1024  # number of frames per buffer
    format = pyaudio.paInt16  # audio format
    channels = 1  # mono
    rate = 48000  # Hertz

    p = pyaudio.PyAudio()

    stream = p.open(format=format,
                    channels=channels,
                    rate=rate,
                    input=True,
                    frames_per_buffer=chunk_size)

    # Set settings for model
    model_size = "tiny.en"
    model = WhisperModel(model_size, device="cuda", compute_type="float16")

    # Save Transcription
    accumulated_transcription = ""

    try:
        while True:
            print("Program Starting. Press Ctrl+C to stop.")
            chunk_file_path = "./audio/chunk.wav"
            record_audio(chunk_file_path, p, stream, rate, chunk_size, format, channels, chunk_length)

            transcription = transcribe_audio(model, "./audio/chunk.wav")
            print(transcription)

            # Add transcription to accumulated_transcription
            accumulated_transcription += transcription + ""

    except KeyboardInterrupt:
        print("Exiting...")
        with open("log.txt", "w") as f:
            f.write(accumulated_transcription)

    finally:
        print("LOG: ", accumulated_transcription)
        stream.stop_stream()
        stream.close()
        p.terminate()


if __name__ == "__main__":
    main()
