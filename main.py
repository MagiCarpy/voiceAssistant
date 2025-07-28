from models.transcriber import Transcriber
from models.model import VoiceAssistant
from processors.display import overlay_display
import torch
from multiprocessing import Process, Queue

#torch.cuda.empty_cache()

assistantName = "Marvin"

if __name__ == "__main__":
    assistant = VoiceAssistant()
    transcriber = Transcriber()

    stop_signal = False
    q = Queue()
    q.put("")
    p = Process(target=overlay_display, args=(q,))
    p.start()
    while True:

        prev_audio, stream = transcriber.detect_wake(debug=True)
        print("\n<-----------VOICE ASSISTANT READY-----------> \n")
        if not p or not p.is_alive():
            print("Overlay starting")
            p = Process(target=overlay_display, args=(q,))
            p.start()
        query = transcriber.record_audio(prev_audio=prev_audio, stream=stream)
        print("\n<----------------PROCESSING----------------> \n")
        try:
            query = query[query.index(assistantName):]
            stream.close()
        except ValueError:
            stream.close()
            print("\n<----------------Try Again----------------> \n") 
            continue
        q.put("...")
        print(f"Query: {query} \n")
        print("Response: ", end="")
        assistant.display_and_speak(query, overlay_queue=q)
        print("\n")
        print("\n<-----------------COMPLETE----------------->")
        #display_response(get_model_response(query))