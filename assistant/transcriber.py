from transformers import pipeline
import sounddevice as sd
import pyaudio, webrtcvad
import torch
import numpy as np
import sys
import time

# Audio Config
FORMAT = pyaudio.paInt16
CHANNELS = 1
CHUNK_DURATION = 20
RATE = 16000
CHUNK_SIZE = int(RATE*CHUNK_DURATION / 1000)


# Initialization
vad = webrtcvad.Vad()
vad.set_mode(2)  # 0: Aggressive filtering, 3: Less aggressive

device =  torch.accelerator.current_accelerator().type if torch.accelerator.is_available else "cpu"

# Wake word detector
classifier = pipeline(
    "audio-classification", model="MIT/ast-finetuned-speech-commands-v2", device="cpu"
)

#print(classifier.model.config.id2label[27])

# speech transcriber
transcriber = pipeline(
    "automatic-speech-recognition", model="openai/whisper-small.en", device=device
)

# implementations
def detect_wake(wake_word="marvin", prob_threshold=0.70, chunk_length=2.0, stream_chunk=0.25, debug=False):
    if wake_word not in classifier.model.config.label2id.keys():
        raise ValueError(
            f"Wake word {wake_word} not in set of valid class labels, pick a wake word in the set {classifier.model.config.label2id.keys()}."
        )

    sampling_rate = classifier.feature_extractor.sampling_rate

    p = pyaudio.PyAudio()

    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE
    )

    print("Listening for wake word...")
    frames = []
    counter = 0
    while True:
        seconds = 0.1
        for _ in range(int(RATE / CHUNK_SIZE * seconds)):
            frame = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            frames.append(frame)

        # Trim frames to prevent excessive growth (e.g., keep last 2 seconds)
        max_frames = int(RATE / CHUNK_SIZE * 2)  # Store 2 seconds of audio
        if len(frames) > max_frames:
            frames = frames[-max_frames:]
    
        try:
            audio_bytes = b"".join(frames)
            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16) 
            audio = audio_int16.astype(np.float32) / 32768.0
            prediction = classifier(audio)
            prediction = prediction[0]
            if debug:
                print(prediction)
            if prediction["label"] == wake_word and prediction["score"] > prob_threshold:
                return (frames, stream)
        except IOError as e:
                if e.errno == pyaudio.paInputOverflowed:
                    print("Warning: Input buffer overflowed. Skipping frame.")
                    continue  # Skip the current frame and continue reading
                else:
                    raise
        except Exception as e:
            print(e)
            stream.stop_stream()
            stream.close()
                

def transcribe(prev_audio=None, stream=None):
    if prev_audio:
        if stream:
            frames = record_audio(prev_audio=prev_audio, stream=stream)
        else:
            frames = record_audio(prev_audio=prev_audio)
    else:
        if stream:
            frames = record_audio(stream=stream)
        else:
            frames = record_audio()
    return transcriber(frames, generate_kwargs={"max_new_tokens": 128})["text"]


# helper
def is_speech(frame, sample_rate): 
    return vad.is_speech(frame, sample_rate)

def record_audio(prev_audio=None, stream=None):
    # FIXME: clean up this code module
    # Open stream

    p = pyaudio.PyAudio()

    if not stream:

        stream = p.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK_SIZE)

    frames = prev_audio if prev_audio else []
    counter = 0

    print("Listening for speech...")
    while True:
        ambient_noise = False
        frame = stream.read(CHUNK_SIZE)
        frame_data = np.frombuffer(frame, dtype=np.int16)

        #print(np.abs(frame_data).mean())
        if np.abs(frame_data).mean() < 180:  # Adjust threshold based on
            ambient_noise = True
            #frame = np.zeros_like(frame_data).tobytes()

        if is_speech(frame, RATE):
            if not ambient_noise:
                counter = 0

            frames.append(frame)
        else:
            counter += 1
            if counter >= 85:
                print("Silence detected, stopping recording.")
                break

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    audio_bytes = b''.join(frames)
    audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
    audio_float32 = audio_int16.astype(np.float32) / 32768.0
    return audio_float32


if __name__ == "__main__":
    #from model import display_and_speak
    #display_and_speak(transcribe())
    #print(f"final: {transcribe()}")
    #prev_audio, stream = detect_wake(debug=True)
    #query = transcribe(prev_audio=prev_audio, stream=stream)
    query = transcribe()