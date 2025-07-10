from transcriber import detect_wake, transcribe
from model import get_model_response, display_response, synthesize, speaker_embeddings, display_and_speak
import sounddevice as sd
import torch

torch.cuda.empty_cache()


if __name__ == "__main__":
    while True:
        detect_wake(debug=True)
        print("\n<-----------VOICE ASSISTANT READY----------->")
        query = transcribe()
        print("\n<----------------PROCESSING---------------->")
        print(f"Query: {query}")
        display_and_speak(query)
        print("\n<-----------------COMPLETE----------------->")
        #display_response(get_model_response(query))

