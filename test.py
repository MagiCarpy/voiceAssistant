import torch
import sounddevice as sd
import numpy as np
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline

# Device config
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load Whisper model
model_id = "openai/whisper-base"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True
).to(device)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

# Audio settings
SAMPLING_RATE = 16000  # Whisper expects 16kHz
DURATION = 1  # seconds of audio per chunk

print("Listening... Press Ctrl+C to stop")

try:
    while True:
        print("\nRecording chunk...")
        audio = sd.rec(int(DURATION * SAMPLING_RATE), samplerate=SAMPLING_RATE, channels=1, dtype='float32')
        sd.wait()
        audio = np.squeeze(audio)

        # Process with Whisper
        result = pipe({"array": audio, "sampling_rate": SAMPLING_RATE})
        print("Transcript:", result["text"])

except KeyboardInterrupt:
    print("\nStopped.")
