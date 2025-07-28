from ollama import chat, generate
from piper import PiperVoice, SynthesisConfig
import sounddevice as sd
import soundfile as sf
import wave
import os
import torch
import re

# FIXME: check if this actually does anything (maybe should move to the class?)
device =  torch.accelerator.current_accelerator().type if torch.accelerator.is_available else "cpu"
default_assistant_model = "voiceAssist2"
default_voice_model = os.path.dirname(os.path.abspath(__file__)) + "/../en_GB-alan-medium.onnx" 

class VoiceAssistant:
    def __init__(self, assistant_model=default_assistant_model, voice_model=default_voice_model):
        self.assistant_model = assistant_model
        self.voice_model = voice_model
        self.conv_history = []
        # FIXME: add voice tweaking parameters for synthesisConfig (ex. speed)

        # Preload the models at initialization
        self.voice = PiperVoice.load(self.voice_model)
        self.syn_config = SynthesisConfig(
            volume=0.5,  # half as loud
            length_scale=0.5,  # twice as slow
            noise_scale=1.0,  # more audio variation
            noise_w_scale=1.0,  # more speaking variation
            normalize_audio=False, # use raw audio from voice
        )

        print("LOADED ASSISTANT")
        generate(model=self.assistant_model, prompt="")

    def display_and_speak(self, prompt, model_type=default_assistant_model, overlay_queue=None):
        self.conv_history.append({"role": "user", "content": prompt})
        max_history = 20

        response = chat(
            model=model_type,
            messages=self.conv_history,
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
                        self._playback(buffer)
                        buffer = ""

        self.conv_history.append({"role": "assistant", "content": full_text})
        if len(self.conv_history) > max_history:
            self.conv_history[max_history/2:]
        print(self.conv_history)

    def _playback(self, text):
        with wave.open("test.wav", "wb") as wav_file:
            self.voice.synthesize_wav(text, wav_file, syn_config=self.syn_config)

        data, fs = sf.read("test.wav", dtype='float32')

        # Play the audio data
        sd.play(data, fs)
        sd.wait()

        os.remove("test.wav")

    
if __name__ == "__main__":
    assistant = VoiceAssistant()
    assistant.display_and_speak("What is the square root of 2?")