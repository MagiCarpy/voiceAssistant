from models.transcriber import Transcriber
from models.model import VoiceAssistant
from processors.display import overlay_display
import torch
from multiprocessing import Process, Queue, set_start_method

assistantName = "Marvin"

def main():
    assistant = VoiceAssistant()
    transcriber = Transcriber()

    overlay_queue = Queue()
    overlay_queue.put("")
    p = Process(target=overlay_display, args=(overlay_queue,))
    p.start()

    while True:
        prev_audio, stream = transcriber.detect_wake(debug=True)
        print("\n<-----------VOICE ASSISTANT READY-----------> \n")
        if not p or not p.is_alive():
            print("Overlay starting")
            p = Process(target=overlay_display, args=(overlay_queue,))
            p.start()
        query = transcriber.record_audio(prev_audio=prev_audio, stream=stream)
        print("\n<----------------PROCESSING----------------> \n")
        try:
            query = query[query.index(assistantName):]
            stream.close()
        except ValueError:
            pass
        except AttributeError:
            stream.close()
            print("\n<----------------Try Again----------------> \n") 
            continue

        overlay_queue.put("...")
        print(f"Query: {query} \n")
        print("Response: ", end="")
        assistant.display_and_speak(query, overlay_queue=overlay_queue)
        print("\n")
        print("\n<-----------------COMPLETE----------------->")

def overlay_and_speak_worker():#assistant=VoiceAssistant(), transcriber=Transcriber()):
    while True:
        #check if wake word detected.
            #interrupt and prompt model

        #prev_audio, stream = transcriber.detect_wake(debug=True)
        pass

if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()