from transcriber import detect_wake, transcribe
from model import display_and_speak
import torch

#torch.cuda.empty_cache()

assistantName = "Marvin"

if __name__ == "__main__":
    while True:
        prev_audio, stream = detect_wake(debug=True)
        print("\n<-----------VOICE ASSISTANT READY-----------> \n")
        query = transcribe(prev_audio=prev_audio, stream=stream)
        print("\n<----------------PROCESSING----------------> \n")
        query = query[query.index(assistantName):]
        print(f"Query: {query} \n")

        print("Response: ", end="")
        display_and_speak(query)
        print("\n")
        print("\n<-----------------COMPLETE----------------->")
        #display_response(get_model_response(query))

