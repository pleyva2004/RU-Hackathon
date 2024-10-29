import openai
from pathlib import Path
from openai import OpenAI
import pygame

# Set your API key
openai.api_key = ""
client = OpenAI(

    # base_url="http://172.30.64.1:1234/v1",
    # api_key="not-needed"
    api_key = openai.api_key
)

# model = "local-model"
model = "gpt-4"


def translate_text(input_text, target_language, source_language=None):
    # Adjust prompt based on whether source_language is provided
    if source_language:
        prompt = f"Translate the following text from {source_language} to {target_language}: \"{input_text}\""
    else:
        prompt = f"Translate the following text to {target_language}: \"{input_text}\""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )

        # Extract the translation from the response
        translation = response.choices[0].message
        translation = str(translation)
        start = translation.find('=') + 2
        end = translation.find('refusal') - 3
        translation = translation[start:end]
        return translation

    except Exception as e:
        return f"An error occurred: {str(e)}"

# Example usage with source language detection
# translated_text = translate_text("Bonjour, comment ça va?", "Hindu")
# print(translated_text)  # Output: "Hello, how are you?"


def talk(input_text):

    # change the name of the file with each iteration
    speech_file_path = Path(__file__).parent / "audio" / "speech5.mp3"

    response = openai.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=input_text
    )

    response.stream_to_file(speech_file_path)
# talk("Today is a wonderful day to build something people love!")


def listen():
    audio_file = open("./audio/spanish_audio_test.m4a", "rb")
    transcription = openai.audio.translations.create(
        model="whisper-1",
        file=audio_file,
        response_format="json"
    )
    return transcription


def audio_out(path):
    # Initialize the pygame mixer
    pygame.mixer.init()

    # Load the MP3 file
    pygame.mixer.music.load(path)

    # Play the file
    pygame.mixer.music.play()

    # Keep the program running to allow the music to play
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)


audio_out("./audio/spanish_audio_test.mp3")

transcript = listen()
mytrans = str(transcript)[18:-2]
print(mytrans)

translated_text = translate_text(mytrans, "Hindi")
print(translated_text)

talk(translated_text)

audio_out("./audio/speech5.mp3")
