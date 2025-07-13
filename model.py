from ollama import chat
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import sounddevice as sd
import torch
import re
from queue import Queue
import multiprocessing

#---Voice Assistant---#
model = "voiceAssistant" 

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

def display_and_speak(prompt, model_type=model):
    stream = chat(
        model=model_type,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    active_audios = Queue()
    talking_signal = False
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
                    if active_audios.qsize() >= 2:
                        dequeue_and_play(active_audios, talking_signal)
                    audio_process = multiprocessing.Process(synthesize_and_queue(buffer, speaker_embeddings, active_audios))
                    audio_process.start()
                    print("started")

                    buffer = ""
    while active_audios.qsize() != 0:
        dequeue_and_play(active_audios, talking_signal)


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
    return speech.cpu()

# attempt to synchronize thread execution
def synthesize_and_queue(text, speaker_embeddings, queue):
    audio = synthesize(text, speaker_embeddings).numpy()
    queue.put(audio)

def dequeue_and_play(queue, speaking_signal):
    speaking_signal = True
    audio = queue.get()
    sd.play(0.75 * audio, 16000)
    sd.wait()
    speaking_signal = False

def play_audio(audio):
    sd.play(0.75*audio, 16000)
    sd.wait()


if __name__ == "__main__":
    #audio = synthesize("Hello brandon this is a test. I want to see if this scottish voice is any good.")
    #print("aaa")
    #sd.play(0.5*audio, samplerate=16000)
    #sd.wait()

    display_and_speak("What is psilocybin and can it be made into a concentrated serum for concentrated dosage?")

