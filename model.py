from ollama import chat
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import sounddevice as sd
import threading
import torch
import re

#---Voice Assistant---#
model = "voiceAssist" 

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
    
    full_text = ""
    buffer = ""
    thread_active = True
    for chunk in stream:
        if "message" in chunk and "content" in chunk["message"]:
            chunk_text = chunk["message"]["content"]
            if chunk_text:
                print(chunk_text, end="")  # Real-time display
                full_text += chunk_text
                buffer += chunk_text
                if bool(re.search(r'[.!?]', chunk_text)):
                    audio = synthesize(buffer, speaker_embeddings).numpy()
                    thread_active = True
                    audio_thread = threading.Thread(target=sd.play,args=(0.5*audio,16000,))
                    # sd.play(0.5 * audio, samplerate=16000)
                    audio_thread.start()
                    audio_thread.join()
                    sd.wait()

                    buffer = ""
                    

    # audio = synthesize(full_text, speaker_embeddings).numpy()
    # sd.play(0.5 * audio, samplerate=16000)
    # sd.wait()


#---Text to Speech---#
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)

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

def synthesize(text, speaker_embeddings):
    inputs = processor(text=text, return_tensors="pt")
    speech = model.generate_speech(
        inputs["input_ids"].to(device), speaker_embeddings.to(device), vocoder=vocoder
    )
    return speech.cpu()

if __name__ == "__main__":
    audio = synthesize("Hello brandon this is a test. I want to see if this scottish voice is any good.")
    print("aaa")
    sd.play(0.5*audio, samplerate=16000)
    sd.wait()