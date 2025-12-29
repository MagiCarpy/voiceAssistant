from models.transcriber import Transcriber
from models.model import VoiceAssistant
from processors.display import overlay_display
import torch
from multiprocessing import Process, Queue, set_start_method

assistantName = "Marvin"

def main():
    query_q = Queue()  # Use multiprocessing.Queue instead of queue.Queue
    overlay_q = Queue()
    overlay_q.put("")
    transcriber = Transcriber(query_q)
    assistant = VoiceAssistant()

    p = Process(target=overlay_display, args=(overlay_q,))
    p.start()

    try:
        while True:
            prev_audio, stream = transcriber.detect_wake(debug=True)
            print("\n<-----------VOICE ASSISTANT READY-----------> \n")
            if not p.is_alive():
                print("Overlay starting")
                p = Process(target=overlay_display, args=(overlay_q,))
                p.start()
            
            query = transcriber.record_audio(prev_audio=prev_audio, stream=stream)
            print("\n<----------------PROCESSING----------------> \n") 
            overlay_q.put("...")
            name_index = query.index(assistantName)
            print(f"Query: {query[name_index + 1:]} \n")
            assistant.display_and_speak(prompt_queue=query_q, overlay_queue=overlay_q)
            print("\n")
            print("\n<-----------------COMPLETE----------------->")

    finally:
        if p.is_alive():
            p.terminate()
            p.join()
        if transcriber.preprocessing.is_alive():
            transcriber.preprocessing.terminate()
            transcriber.preprocessing.join()

def overlay_and_speak_worker():
    pass

if __name__ == "__main__":
    torch.cuda.empty_cache()
    set_start_method('spawn', force=True)
    main()