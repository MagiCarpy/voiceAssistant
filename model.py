
from ollama import chat
from piper import PiperVoice, SynthesisConfig
import sounddevice as sd
import soundfile as sf
import numpy as np
import pyaudio
import wave
import os
import time
import torch
import re
from multiprocessing import Queue, Process

device = "cuda" if torch.cuda.is_available() else "cpu"

model = "voiceAssistant"
voice_model = os.getcwd() + "/en_GB-alan-medium.onnx"
voice = PiperVoice.load(voice_model)

syn_config = SynthesisConfig(
    volume=0.5,  # half as loud
    length_scale=0.5,  # twice as slow
    noise_scale=1.0,  # more audio variation
    noise_w_scale=1.0,  # more speaking variation
    normalize_audio=False, # use raw audio from voice
)

def display_and_speak(prompt, model_type=model):
    stream = chat(
        model=model_type,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    full_text = ""
    buffer = ""
    for chunk in stream:
        if "message" in chunk and "content" in chunk["message"]:
            chunk_text = chunk["message"]["content"]
            print(chunk_text, end="", flush=True)
            if chunk_text:
                full_text += chunk_text
                buffer += chunk_text
                if bool(re.search(r'[.!?]', chunk_text)):
                    playback(buffer)
                    buffer = ""


#---Text to Speech---#
def playback(text):
    with wave.open("test.wav", "wb") as wav_file:
        voice.synthesize_wav(text, wav_file, syn_config=syn_config)

    data, fs = sf.read("test.wav", dtype='float32')

    # Play the audio data
    sd.play(data, fs)
    sd.wait()

    os.remove("test.wav")


#---Voice Assistant Text---#
def get_model_response(prompt, model_type=model):
    stream = chat(
        model=model_type,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    return stream

def display_response(stream):
    for chunk in stream:    
        print(chunk["message"]["content"], end="", flush=True)
    print("\n")
    
if __name__ == "__main__":
    display_and_speak("Give me 3 sentences about sterile technique.")

