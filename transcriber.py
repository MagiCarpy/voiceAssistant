from transformers import pipeline
from transformers.pipelines.audio_utils import ffmpeg_microphone_live
import sounddevice as sd
import pyaudio, webrtcvad
import torch
import numpy as np
import sys

# Audio Config
FORMAT = pyaudio.paInt16
CHANNELS = 1
CHUNK_DURATION = 20
RATE = 16000
CHUNK_SIZE = int(RATE*CHUNK_DURATION / 1000)


# Initialization
vad = webrtcvad.Vad()
vad.set_mode(2)  # 0: Aggressive filtering, 3: Less aggressive

# device = "gpu" if torch.cuda.is_available() else "cpu"
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Wake word detector
classifier = pipeline(
    "audio-classification", model="MIT/ast-finetuned-speech-commands-v2", device="cpu"
)

print(classifier.model.config.id2label[27])

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
    while True:
        if not stream:
            stream = p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE
            )
            
        seconds = 0.10
        for _ in range(0, int(RATE/CHUNK_SIZE*seconds)):
            data = stream.read(CHUNK_SIZE)
            int16_array = np.frombuffer(data, dtype=np.int16)
            frames.append(int16_array)

        if len(frames) > int(RATE/CHUNK_SIZE*seconds) * 20:
            frames = frames[5*int(RATE/CHUNK_SIZE*seconds):]

    # mic = ffmpeg_microphone_live(
    #          sampling_rate=sampling_rate,
    #          chunk_length_s=chunk_length,
    #          stream_chunk_s=stream_chunk,
    #      )

    
        try:
            audio = np.concatenate(frames).astype(np.float32) / 32768.0
            prediction = classifier(audio)
            prediction = prediction[0]
            if debug:
                print(prediction)
            if prediction["label"] == wake_word:
                if prediction["score"] > prob_threshold:
                    stream.stop_stream()
                    stream.close()
                    return True
        except Exception as e:
            print(e)
            stream.stop_stream()
            stream.close()
                

def transcribe():
    frames = record_audio()
    return transcriber(frames, generate_kwargs={"max_new_tokens": 128})["text"]


# helper
def is_speech(frame, sample_rate): 
    return vad.is_speech(frame, sample_rate)

def record_audio():

    # FIXME: make audio record when sound spikes
    # Makes the user not have to wait for model 
    # to start listening.

    # Open stream
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK_SIZE)

    frames = []
    recording = False
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

            if not recording:
                print("Recording started.")
                recording = True
            frames.append(frame)
        else:
            counter += 1
            if recording and counter >= 85:
                print("Silence detected, stopping recording.")
                break

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

    audio_bytes = b''.join(frames)
    audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
    audio_float32 = audio_int16.astype(np.float32) / 32768.0
    return audio_float32


if __name__ == "__main__":
    detect_wake(debug=True)
    #print(f"final: {transcribe()}")