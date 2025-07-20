from assistant.transcriber import detect_wake, transcribe
from assistant.model import display_and_speak
from display import overlay_display
import torch
from multiprocessing import Process, Queue

#torch.cuda.empty_cache()

assistantName = "Marvin"

if __name__ == "__main__":
    q = Queue()
    p = Process(target=overlay_display, args=(q,))
    p.start()
    while True:
        prev_audio, stream = detect_wake(debug=True)
        print("\n<-----------VOICE ASSISTANT READY-----------> \n")
        if not p.is_alive():
            p = Process(target=overlay_display, args=(q,))
            p.start()
        query = transcribe(prev_audio=prev_audio, stream=stream)
        print("\n<----------------PROCESSING----------------> \n")
        try:
            query = query[query.index(assistantName):]
        except ValueError:
            print("\n<----------------Try Again----------------> \n") 
            continue
        q.put("Generating Response")
        print(f"Query: {query} \n")
        print("Response: ", end="")
        display_and_speak(query, overlay_queue=q)
        print("\n")
        print("\n<-----------------COMPLETE----------------->")
        #display_response(get_model_response(query))

