import torch
import sounddevice as sd
import numpy as np
import kenlm
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from pyctcdecode import build_ctcdecoder
from datasets import load_dataset
import noisereduce as nr
import queue
import sys

# Configuration
MODEL_ID = "facebook/wav2vec2-base-960h"
SAMPLE_RATE = 16000
CHUNK_DURATION = 2.0
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" 

# Initialize model and processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
model.eval()  # Set to evaluation mode

kenlm_model = 'C:\\Users\\brand\\github\\voiceAssistant\\models\\lm_csr_64k_vp_2gram.arpa'

# Vocab txt
with open("vocab.txt", "r", encoding="utf-8") as file:
    lines = file.readlines()
    vocab = [line.strip() for line in lines]

vocab_dict = processor.tokenizer.get_vocab()
sorted_vocab = sorted(vocab_dict.items(), key=lambda x: x[1])
labels = [label for label, _ in sorted_vocab]

decoder = build_ctcdecoder(labels=labels,
                           kenlm_model_path=kenlm_model)

q = queue.Queue()

def callback(indata, frames, time, status):
    """Callback to capture audio chunks."""
    if status:
        print(status, file=sys.stderr)
    q.put(indata.copy())

def preprocess_audio(audio):
    """Preprocess audio to match model requirements."""
    # Ensure mono and correct sampling rate
    audio = np.squeeze(audio).astype(np.float32)

    max_val = np.max(np.abs(audio))
    
    if max_val > 0:
        audio = audio / max_val

    audio_clean = nr.reduce_noise(y=audio, sr=SAMPLE_RATE, prop_decrease=0.8, stationary=False)
    energy = np.sum(audio_clean**2) / len(audio_clean)
    print(energy)
    if energy < 1e-3:
        return np.zeros_like(audio_clean)
    return audio_clean

def transcribe_audio(audio):
    """Transcribe audio using wav2vec 2.0."""
    # Preprocess for model
    inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
    input_values = {k: v.to(model.device) for k, v in inputs.items()} 

    # Perform inference
    with torch.no_grad():
        logits = model(**input_values).logits

    # Decode to text
    logits = logits.detach().cpu().numpy()
    transcription = [decoder.decode(logit, beam_width=20) for logit in logits]
    return transcription[0]

def main():
    try:
        # Set up audio stream
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                            dtype='float32',
                            blocksize=CHUNK_SIZE, callback=callback):
            print("Listening... Press Ctrl+C to stop.")
            buffer = []
            while True:
                # Get audio chunk
                audio_chunk = q.get()
                audio_chunk = audio_chunk.flatten()


                # Preprocess audio
                audio_processed = preprocess_audio(audio_chunk)

                # Transcribe
                transcription = transcribe_audio(audio_processed)
                if transcription.strip():
                    print("Transcription:", transcription)

    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()