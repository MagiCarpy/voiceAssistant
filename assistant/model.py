
from ollama import chat
from display import QApplication, OverlayWindow, overlay_display
from piper import PiperVoice, SynthesisConfig
import sounddevice as sd
import soundfile as sf
import numpy as np
import pyaudio
import wave
import os, sys
import time
import torch
import re
from multiprocessing import Process, Queue

device =  torch.accelerator.current_accelerator().type if torch.accelerator.is_available else "cpu"

model = "voiceAssist2"
voice_model = os.getcwd() + "/en_GB-alan-medium.onnx"
voice = PiperVoice.load(voice_model)

def overlay_worker(text):
    app = QApplication(sys.argv)
    overlay = OverlayWindow()
    overlay.updateText(text)
    overlay.show()
    sys.exit(app.exec_())

syn_config = SynthesisConfig(
    volume=0.5,  # half as loud
    length_scale=0.5,  # twice as slow
    noise_scale=1.0,  # more audio variation
    noise_w_scale=1.0,  # more speaking variation
    normalize_audio=False, # use raw audio from voice
)

conversation_history = []

def display_and_speak(prompt, model_type=model, overlay_queue=None):
    conversation_history.append({"role": "user", "content": prompt})

    response = chat(
        model=model_type,
        messages=conversation_history,
        stream=True,
    )

    full_text = ""
    buffer = ""
    for chunk in response:
        if "message" in chunk and "content" in chunk["message"]:
            chunk_text = chunk["message"]["content"]
            print(chunk_text, end="", flush=True)
            if chunk_text:
                full_text += chunk_text
                if overlay_queue:
                    overlay_queue.put(full_text)
                buffer += chunk_text
                if bool(re.search(r'[.!?]', chunk_text)):
                    playback(buffer)
                    buffer = ""

    conversation_history.append({"role": "assistant", "content": full_text})
    print(conversation_history)

    #time.sleep(5)
    #p.terminate()
    #p.join()

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
    response = chat(
        model=model_type,
        messages=conversation_history,
        stream=True,
    )

    return response

def display_response(stream):
    for chunk in stream:    
        print(chunk["message"]["content"], end="", flush=True)
    print("\n")

    
if __name__ == "__main__":
    display_and_speak("Give me 3 sentences about sterile technique.")

