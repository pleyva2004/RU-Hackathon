import numpy as np
import sounddevice as sd
import riva.client

# Initialize Riva client
auth = riva.client.Auth(uri='localhost:50051')
riva_tts = riva.client.SpeechSynthesisService(auth)



def main():
    # Setting up TTS request
    sample_rate_hz = 44100
    language_code = "en-US"
    voice_name = "English-US.Male-Happy"

    req = { 
            "language_code"  : language_code,
            "encoding"       : riva.client.AudioEncoding.LINEAR_PCM ,   # LINEAR_PCM and OGGOPUS encodings are supported
            "sample_rate_hz" : sample_rate_hz,                          # Generate 44.1KHz audio
            "voice_name"     : voice_name                                # The name of the voice to generate
    }

    req["text"] = "Is it recognize speech or wreck a nice beach?"
    resp = riva_tts.synthesize(**req)
    audio_samples = np.frombuffer(resp.audio, dtype=np.int16)
    #ipd.Audio(audio_samples, rate=sample_rate_hz)

    
    # Play audio
    sd.play(audio_samples, sample_rate_hz)
    sd.wait()  # Wait until audio finishes playing


if __name__ == "__main__":
    main()

