import os
import pyaudio
import wave
import asyncio
from pydub import AudioSegment
import openai
import logging
import aioprocessing
import signal
from collections import deque
from datetime import datetime
from tqdm import tqdm

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

# Define a 'pending_translations' variable and a deque to hold history
pending_translations = 0
translation_queue_depth_history = deque(maxlen=10)

def record(filename, ended_recording_queue):
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
        ended_recording_queue.put(datetime.now().timestamp())

    except TimeoutException:
        logging.error("Conversion process took too long!")

async def record_and_save(filename):
    global pending_translations
    ended_recording_queue = aioprocessing.AioQueue()
    process = aioprocessing.AioProcess(target=record, args=(filename, ended_recording_queue))
    process.start()
    await process.coro_join()
    logging.debug(f"Finished recording and saving {filename}.mp3")
    pending_translations += 1
    translation_queue_depth_history.append(pending_translations)
    # Retrieve ended_recording value from queue
    ended_recording = datetime.fromtimestamp(ended_recording_queue.get())
    return ended_recording

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

async def transcribe_and_translate(filename, started_translation):
    global pending_translations
    lag = datetime.now() - started_translation
    logging.info(f'Lag between recording and start of translation: {lag.total_seconds()} seconds')
    # The rest of your transcribe_and_translate function goes here...
    pending_translations -= 1
    # Update the history
    translation_queue_depth_history.append(pending_translations)
    logging.info(f"Current translation queue depth: {pending_translations}")

async def main():
    global pending_translations
    index = 1
    while True:
        filename = os.path.join(OUTPUT_FOLDER, FILE_PREFIX + f"_{index}")
        logging.debug(f"Processing audio file: {filename}")
        ended_recording = await record_and_save(filename)
        task = asyncio.create_task(transcribe_and_translate(filename, ended_recording))
        await asyncio.sleep(0.1)
        logging.debug(f"Task created for transcribing and translating {filename}.mp3 {task}")
        logging.info(f"Current translation queue depth: {pending_translations}")
        # Visualize the history with progress bar
        print("Queue depth history (last 10 entries):", ' '.join(map(str, translation_queue_depth_history)))
        with tqdm(total=max(translation_queue_depth_history), bar_format='{l_bar}{bar}|') as pbar:
            for i in range(pending_translations):
                await asyncio.sleep(0.1)  # Non-blocking delay
                pbar.update(1)
        index += 1

if __name__ == "__main__":
    asyncio.run(main())