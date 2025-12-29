from models.transcriber import Transcriber
from models.model import VoiceAssistant
from processors.display import overlay_display
import torch
from multiprocessing import Process, Queue, set_start_method
from queue import Queue as normalQueue
assistantName = "Marvin"

def main():
    query_q = normalQueue()
    overlay_q = Queue()
    overlay_q.put("")

    transcriber = Transcriber(query_q)
    assistant = VoiceAssistant()

    p = Process(target=overlay_display, args=(overlay_q,))
    p.start()

    while True:
        prev_audio, stream = transcriber.detect_wake(debug=True)
        print("\n<-----------VOICE ASSISTANT READY-----------> \n")
        if not p or not p.is_alive():
            print("Overlay starting")
            p = Process(target=overlay_display, args=(overlay_q,))
            p.start()
        
        query = transcriber.record_audio(prev_audio=prev_audio, stream=stream)

        print("\n<----------------PROCESSING----------------> \n") 
        overlay_q.put("...")
        print(f"Query: {query} \n")
        assistant.display_and_speak(prompt_queue=query_q, overlay_queue=overlay_q)
        print("\n")
        print("\n<-----------------COMPLETE----------------->")

        # query = transcriber.record_audio(prev_audio=prev_audio, stream=stream)
        # print("\n<----------------PROCESSING----------------> \n")
        # try:
        #     query = query[query.index(assistantName):]
        #     stream.close()
        # except ValueError:
        #     pass
        # except AttributeError:
        #     stream.close()
        #     print("\n<----------------Try Again----------------> \n") 
        #     continue

        # overlay_q.put("...")
        # print(f"Query: {query} \n")
        # print("Response: ", end="")
        # assistant.display_and_speak(query, overlay_queue=overlay_q)
        # print("\n")
        # print("\n<-----------------COMPLETE----------------->")

def overlay_and_speak_worker():#assistant=VoiceAssistant(), transcriber=Transcriber()):
    while True:
        #check if wake word detected.
            #interrupt and prompt model

        #prev_audio, stream = transcriber.detect_wake(debug=True)
        pass

if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()