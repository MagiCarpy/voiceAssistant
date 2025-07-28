from transformers import pipeline
import pyaudio, webrtcvad
import torch
import numpy as np
from multiprocessing import Queue, Process

# Audio Config
FORMAT = pyaudio.paInt16
CHANNELS = 1
CHUNK_DURATION = 20
RATE = 16000
CHUNK_SIZE = int(RATE*CHUNK_DURATION / 1000)

device =  torch.accelerator.current_accelerator().type if torch.accelerator.is_available else "cpu"

default_wake_model = "./MIT/ast-finetuned-speech-commands-v2" 

default_transcriber_model = "./openai/whisper-small.en"

default_wake_word = "marvin"

vad = webrtcvad.Vad()
vad.set_mode(2) # filtering level

class Transcriber:
    def __init__(self, wake_word=default_wake_word, wake_model=default_wake_model, transcriber_model=default_transcriber_model):
        self.classifier = pipeline(
    "audio-classification", model=default_wake_model, device=-1)
        self.transcriber = pipeline(
    "automatic-speech-recognition", model=default_transcriber_model, device=-1
    )
        # preprocessing = Process(target=self.transcriber_worker, args=(frames, transcriptions_q))
        # preprocessing.start()
        
    def detect_wake(self, wake_word=default_wake_word, prob_threshold=0.70, chunk_length=2.0, stream_chunk=0.25, debug=False):
        if wake_word not in self.classifier.model.config.label2id.keys():
            raise ValueError(
                f"Wake word {wake_word} not in set of valid class labels, pick a wake word in the set {self.classifier.model.config.label2id.keys()}."
            )

        sampling_rate = self.classifier.feature_extractor.sampling_rate

        p = pyaudio.PyAudio()

        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE
        )

        print("Listening for wake word...")
        frames = []
        counter = 0
        while True:
            seconds = 0.1
            for _ in range(int(RATE / CHUNK_SIZE * seconds)):
                frame = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                frames.append(frame)

            # Trim frames to prevent excessive growth (e.g., keep last 2 seconds)
            max_frames = int(RATE / CHUNK_SIZE * 2)  # Store 2 seconds of audio
            if len(frames) > max_frames:
                frames = frames[-max_frames:]
        
            try:
                audio_bytes = b"".join(frames)
                audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16) 
                audio = audio_int16.astype(np.float32) / 32768.0
                prediction = self.classifier(audio)
                prediction = prediction[0]
                if debug:
                    print(prediction)
                if prediction["label"] == wake_word and prediction["score"] > prob_threshold:
                    return (frames, stream)
            except IOError as e:
                    if e.errno == pyaudio.paInputOverflowed:
                        print("Warning: Input buffer overflowed. Skipping frame.")
                        continue  # Skip the current frame and continue reading
                    else:
                        raise
            except Exception as e:
                print(e)
                stream.stop_stream()
                stream.close()
            
    def transcribe(self, prev_audio=None, stream=None):
        if prev_audio:
            if stream:
                frames = self.record_audio(prev_audio=prev_audio, stream=stream)
            else:
                frames = self.record_audio(prev_audio=prev_audio)
        else:
            if stream:
                frames = self.record_audio(stream=stream)
            else:
                frames = self.record_audio()
        return self.transcriber(frames, generate_kwargs={"max_new_tokens": 128})["text"]
    
    def _is_speech(self, frame, sample_rate): 
        return vad.is_speech(frame, sample_rate)

    def record_audio(self, prev_audio=None, stream=None):
        # FIXME: clean up this code module
        # Open stream

        p = pyaudio.PyAudio()

        if not stream:

            stream = p.open(format=FORMAT,
                                channels=CHANNELS,
                                rate=RATE,
                                input=True,
                                frames_per_buffer=CHUNK_SIZE)

        frames = prev_audio if prev_audio else []
        transcriptions_q = Queue()
        preprocessing = None
        counter = 0
        print("Listening for speech...")
        while True:
            ambient_noise = False
            frame = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            frame_data = np.frombuffer(frame, dtype=np.int16)

            #print(np.abs(frame_data).mean())
            if np.abs(frame_data).mean() < 180:  # Adjust threshold based on
                ambient_noise = True
                #frame = np.zeros_like(frame_data).tobytes()

            if self._is_speech(frame, RATE):
                if not ambient_noise:
                    counter = 0
                frames.append(frame)
            else:
                print(counter)
                counter += 1
                if counter == 30:
                    if preprocessing is not None and preprocessing.is_alive():
                        preprocessing.terminate()
                        preprocessing.join()

                    print("Sending")
                    preprocessing = Process(target=self.transcriber_worker, args=(frames, transcriptions_q))
                    preprocessing.start()
                    #query = transcriber(processed_frames, generate_kwargs={"max_new_tokens": 128})["text"]
                if counter >= 60:
                    print("Silence detected, stopping recording.")
                    break

        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        p.terminate()

        while preprocessing.is_alive() or not transcriptions_q.empty():
            if not transcriptions_q.empty():
                query = transcriptions_q.get()
        
        return query

    def _process_frames(self, frames):
        audio_bytes = b''.join(frames)
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        return audio_float32

    def transcriber_worker(self, frames, queue):
        processed_frames = self._process_frames(frames)
        query = self.transcriber(processed_frames, generate_kwargs={"max_new_tokens": 128})["text"]
        queue.put(query)
        


# Initialization
# vad = webrtcvad.Vad()
# vad.set_mode(2)  # 0: Aggressive filtering, 3: Less aggressive

# Wake word detector

# classifier = pipeline(
#     "audio-classification", model="./MIT/ast-finetuned-speech-commands-v2", device="cpu"
# )

# speech transcriber

# transcriber = pipeline(
#     "automatic-speech-recognition", model="./openai/whisper-small.en", device=device
# )

# implementations
# def detect_wake(wake_word="marvin", prob_threshold=0.70, chunk_length=2.0, stream_chunk=0.25, debug=False):
#     if wake_word not in classifier.model.config.label2id.keys():
#         raise ValueError(
#             f"Wake word {wake_word} not in set of valid class labels, pick a wake word in the set {classifier.model.config.label2id.keys()}."
#         )

#     sampling_rate = classifier.feature_extractor.sampling_rate

#     p = pyaudio.PyAudio()

#     stream = p.open(
#         format=FORMAT,
#         channels=CHANNELS,
#         rate=RATE,
#         input=True,
#         frames_per_buffer=CHUNK_SIZE
#     )

#     print("Listening for wake word...")
#     frames = []
#     counter = 0
#     while True:
#         seconds = 0.1
#         for _ in range(int(RATE / CHUNK_SIZE * seconds)):
#             frame = stream.read(CHUNK_SIZE, exception_on_overflow=False)
#             frames.append(frame)

#         # Trim frames to prevent excessive growth (e.g., keep last 2 seconds)
#         max_frames = int(RATE / CHUNK_SIZE * 2)  # Store 2 seconds of audio
#         if len(frames) > max_frames:
#             frames = frames[-max_frames:]
    
#         try:
#             audio_bytes = b"".join(frames)
#             audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16) 
#             audio = audio_int16.astype(np.float32) / 32768.0
#             prediction = classifier(audio)
#             prediction = prediction[0]
#             if debug:
#                 print(prediction)
#             if prediction["label"] == wake_word and prediction["score"] > prob_threshold:
#                 return (frames, stream)
#         except IOError as e:
#                 if e.errno == pyaudio.paInputOverflowed:
#                     print("Warning: Input buffer overflowed. Skipping frame.")
#                     continue  # Skip the current frame and continue reading
#                 else:
#                     raise
#         except Exception as e:
#             print(e)
#             stream.stop_stream()
#             stream.close()
                

# def transcribe(prev_audio=None, stream=None):
#     if prev_audio:
#         if stream:
#             frames = record_audio(prev_audio=prev_audio, stream=stream)
#         else:
#             frames = record_audio(prev_audio=prev_audio)
#     else:
#         if stream:
#             frames = record_audio(stream=stream)
#         else:
#             frames = record_audio()
#     return transcriber(frames, generate_kwargs={"max_new_tokens": 128})["text"]


# helper
# def is_speech(frame, sample_rate): 
#     return vad.is_speech(frame, sample_rate)

# def record_audio(prev_audio=None, stream=None):
#     # FIXME: clean up this code module
#     # Open stream

#     p = pyaudio.PyAudio()

#     if not stream:

#         stream = p.open(format=FORMAT,
#                             channels=CHANNELS,
#                             rate=RATE,
#                             input=True,
#                             frames_per_buffer=CHUNK_SIZE)

#     frames = prev_audio if prev_audio else []
#     transcriptions_q = Queue()
#     preprocessing = None
#     counter = 0
#     print("Listening for speech...")
#     while True:
#         ambient_noise = False
#         frame = stream.read(CHUNK_SIZE, exception_on_overflow=False)
#         frame_data = np.frombuffer(frame, dtype=np.int16)

#         #print(np.abs(frame_data).mean())
#         if np.abs(frame_data).mean() < 180:  # Adjust threshold based on
#             ambient_noise = True
#             #frame = np.zeros_like(frame_data).tobytes()

#         if is_speech(frame, RATE):
#             if not ambient_noise:
#                 counter = 0
#             frames.append(frame)
#         else:
#             print(counter)
#             counter += 1
#             if counter == 30:
#                 if preprocessing is not None and preprocessing.is_alive():
#                     preprocessing.terminate()
#                     preprocessing.join()

#                 print("Sending")
#                 preprocessing = Process(target=transcriber_worker, args=(frames, transcriptions_q))
#                 preprocessing.start()
#                 #query = transcriber(processed_frames, generate_kwargs={"max_new_tokens": 128})["text"]
#             if counter >= 60:
#                 print("Silence detected, stopping recording.")
#                 break

#     # Stop and close the stream
#     stream.stop_stream()
#     stream.close()
#     p.terminate()

#     while preprocessing.is_alive() or not transcriptions_q.empty():
#         if not transcriptions_q.empty():
#             query = transcriptions_q.get()
    
#     return query

# def process_frames(frames):
#     audio_bytes = b''.join(frames)
#     audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
#     audio_float32 = audio_int16.astype(np.float32) / 32768.0
#     return audio_float32

# def transcriber_worker(frames, queue):
#     processed_frames = process_frames(frames)
#     query = transcriber(processed_frames, generate_kwargs={"max_new_tokens": 128})["text"]
#     queue.put(query)

if __name__ == "__main__":
    pass