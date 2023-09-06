import os
import pyaudio
import wave
import asyncio
from pydub import AudioSegment
import multiprocessing
import openai
import logging
import signal
from pprint import pprint  # For pretty printing

# Setting up the logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")
pydub_logger = logging.getLogger("pydub.converter")
pydub_logger.setLevel(logging.WARNING)

# Audio parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 3
OUTPUT_FOLDER = 'recordings'
FILE_PREFIX = 'recording'

# OpenAI API parameters
API_KEY = os.environ.get("OPENAI_API_KEY")
if not API_KEY:
    logging.error("No API Key found in environment variables. Exiting.")
    exit(1)

openai.api_key = API_KEY

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

def record(filename):
    class TimeoutException(Exception): 
        pass 

    def timeout_handler(signum, frame): 
        raise TimeoutException

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(20)  # Ten seconds for recording, 10 seconds for re-encoding

    try:
        logging.debug("Starting recording...")
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        frames = []
        for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        p.terminate()

        logging.debug(f"Saving recorded data to {filename}.wav")
        wf = wave.open(filename + '.wav', 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

        logging.debug(f"Converting {filename}.wav to MP3")
        sound = AudioSegment.from_wav(filename + '.wav')
        sound.export(filename + '.mp3', format="mp3")
        os.remove(filename + '.wav')

    except TimeoutException:
        logging.error("Conversion process took too long!")

async def record_and_save(filename):
    process = multiprocessing.Process(target=record, args=(filename,))
    process.start()
    process.join()
    logging.debug(f"Finished recording and saving {filename}.mp3")


def detect_language_via_gpt(text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a language detector. Only respond with the language detected in the text provided. Say nothing else."},
            {"role": "user", "content": text}
        ]
    )
    
    detected_language = response.choices[0].message.content
    return detected_language


# Static set of detected languages (You can change or expand this as needed)
detected_languages = set(['en', 'ko'])

async def transcribe_and_translate(filename):
    try:
        logging.debug(f"Starting transcribe for {filename}.mp3")
        with open(filename + ".mp3", 'rb') as file:
            response = openai.Audio.transcribe("whisper-1", file)
            original_text = response.get('text') 

            if original_text:
                translations = {'original': original_text}
                
                # Now, translate into all detected languages
                for lang in detected_languages:
                    translation_response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": f"You are a live translator. Repeat what is stated in the language: {lang}."},
                            {"role": "user", "content": original_text}
                        ]
                    )
                    translated_text = translation_response.choices[0].message.content
                    translations[lang] = translated_text

                for lang, text in translations.items():
                    if lang != 'original':
                        print(f"{lang.upper()}: {text}")
                print("\n")  # Two new lines after each set of translations
 


            else:
                logging.error("No transcription obtained from OpenAI.")

    except Exception as e:
        logging.error(f"Error while transcribing and translating: {e}")

async def main():
    index = 1
    while True:
        filename = os.path.join(OUTPUT_FOLDER, FILE_PREFIX + f"_{index}")
        logging.debug(f"Processing audio file: {filename}")
        await record_and_save(filename)
        task = asyncio.create_task(transcribe_and_translate(filename))
        await asyncio.sleep(0.1)  # short delay to allow the task to start
        logging.debug(f"Task created for transcribing and translating {filename}.mp3 {task}")
        index += 1

if __name__ == "__main__":
    asyncio.run(main())