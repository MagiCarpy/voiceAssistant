from ollama import chat
import pyaudio
import wave
import numpy as np
import os
from piper import PiperVoice, SynthesisConfig
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import sounddevice as sd
import soundfile as sf
import time
import torch
import re
from multiprocessing import Queue, Event, Process

model = "voiceAssistant"
model_path = "/Users/carpp/github/voiceAssistant/en_GB-alan-medium.onnx"
voice = PiperVoice.load("/Users/carpp/github/voiceAssistant/en_GB-alan-medium.onnx")

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

    stop_event = Event()

    speaking = False

    # playback_process = Process(target=dequeue_and_play, args=(active_audios, stop_event))
    # playback_process.start()

    processes = []
    sentence_num = 1
    full_text = ""
    buffer = ""
    for chunk in stream:
        if "message" in chunk and "content" in chunk["message"]:
            chunk_text = chunk["message"]["content"]
            if chunk_text:
                full_text += chunk_text
                buffer += chunk_text
                print(chunk_text, end="")
                if bool(re.search(r'[.!?]', chunk_text)):
                    # audio_process = Process(target=synthesize_and_queue, args=(buffer, speaker_embeddings, active_audios, sentence_num))
                    playback_process = Process(piper_speak(buffer))
                    playback_process.start()
                    # audio_process.start()
                    # sentence_num += 1
                    # processes.append(audio_process)
                    buffer = ""

    # for process in reversed(processes):
    #      process.join()

    # stop_event.set()
    # playback_process.join()


#---Text to Speech---#
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts", local_files_only=True)
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts", local_files_only=True).to(device)
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan", local_files_only=True).to(device)

# embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
# print("loaded")

# speaker_embeddings = torch.tensor(embeddings_dataset[5205]["xvector"]).unsqueeze(0)

# i = 0
# for index, line in enumerate(embeddings_dataset):
#     if i % 600 + 25 == 0:
#         print(f"{index}: {line['filename']}")
#     i += 1

# torch.save(speaker_embeddings, "male_xvectors5205.pt")
speaker_embeddings = torch.load("./TTSModels/male_xvectors5205.pt")

def synthesize(text, speaker_embeddings = speaker_embeddings):
    inputs = processor(text=text, return_tensors="pt")
    speech = model.generate_speech(
        inputs["input_ids"].to(device), speaker_embeddings.to(device), vocoder=vocoder
    )
    return speech.cpu().numpy()

# attempt to synchronize thread execution
def synthesize_and_queue(text, speaker_embeddings, queue, index):
    audio = synthesize(text, speaker_embeddings)
    queue.put((index, audio))

# FIXME: Use BST tree or some other data structure to make
# fetching the audio in the correct order more efficient
def dequeue_and_play(queue, stop_event):
    ordered_audios = []
    current_audio = 1
    while not stop_event.is_set() or not queue.empty() or not len(ordered_audios) == 0:
        print(current_audio)
        if not queue.empty() or not len(ordered_audios) == 0:
            if not queue.empty():
                index, audio = queue.get()
                if (index == current_audio):
                    sd.play(0.75 * audio, 16000)
                    sd.wait()
                    current_audio += 1
                else:
                    ordered_audios.sort(key=lambda x: x[0])
                    ordered_audios.append((index, audio))
                    if (ordered_audios[0][0] == current_audio):
                        _, audio = ordered_audios.pop(0)
                        sd.play(0.75 * audio, 16000)
                        sd.wait()
                        current_audio += 1
            else:
                ordered_audios.sort(key=lambda x: x[0])
                if (ordered_audios[0][0] == current_audio):
                    _, audio = ordered_audios.pop(0)
                    sd.play(0.75 * audio, 16000)
                    sd.wait()
                    current_audio += 1
        else:
            time.sleep(0.1)

def play_audio(audio):
    sd.play(0.75*audio, 16000)
    sd.wait()


#---Voice Assistant---#
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

def piper_speak(text):
    print("speak")
    with wave.open("test.wav", "wb") as wav_file:
        voice.synthesize_wav(text, wav_file, syn_config=syn_config)

    data, fs = sf.read("test.wav", dtype='float32')

    # Play the audio data
    sd.play(data, fs)
    sd.wait()

    os.remove("test.wav")
    
if __name__ == "__main__":
    

    #piper_speak("hello test sentence 1 2 3")
    display_and_speak("Give me 5 sentences, numbered, about penguins.")

