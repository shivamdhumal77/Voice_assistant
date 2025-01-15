import time
import threading
import numpy as np
import whisper
from queue import Queue
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import torch
from kokoro import generate
from models import build_model
import sounddevice as sd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import io

# Initialize models and services
stt = whisper.load_model("base.en")
llm = ChatOpenAI(
    api_key="ollama",  
    base_url="https://sunny-gerri-finsocialdigitalsystem-d9b385fa.koyeb.app/v1",
    model="athene-v2"  
)

# Define prompt template
template = """
You are a helpful and friendly AI assistant. You are polite, respectful, don't use emojis, and aim to provide concise responses.
The conversation transcript is as follows:
{history}

And here is the user's follow-up: {input}

Your response:
"""
PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)

chain = ConversationChain(
    prompt=PROMPT,
    verbose=False,
    memory=ConversationBufferMemory(ai_prefix="Assistant:"),
    llm=llm,
)

# FastAPI app initialization
app = FastAPI(title="Audio Assistant API")

class ResponseModel(BaseModel):
    transcription: str
    audio_url: str

# Utility Functions
def record_audio_with_timeout(silence_threshold: float = 0.01, silence_limit: float = 3.0) -> np.ndarray:
    """
    Records audio until silence is detected for a specified time.
    """
    data_queue = Queue()
    stop_event = threading.Event()

    def callback(indata, frames, time, status):
        if status:
            print(f"Recording error: {status}")
        data_queue.put(bytes(indata))

    def recorder():
        try:
            with sd.RawInputStream(
                samplerate=16000, dtype="int16", channels=1, callback=callback
            ):
                while not stop_event.is_set():
                    time.sleep(0.1)
        except Exception as e:
            stop_event.set()
            raise RuntimeError(f"Error in audio recording: {e}")

    recording_thread = threading.Thread(target=recorder)
    recording_thread.start()

    audio_data = []
    silent_frames = 0
    silence_limit_samples = int(16000 * silence_limit)

    try:
        while not stop_event.is_set():
            while not data_queue.empty():
                frame = data_queue.get()
                audio_chunk = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
                audio_data.append(audio_chunk)

                if np.abs(audio_chunk).mean() < silence_threshold:
                    silent_frames += len(audio_chunk)
                else:
                    silent_frames = 0

                if silent_frames > silence_limit_samples:
                    stop_event.set()
    except Exception as e:
        raise RuntimeError(f"Error processing audio: {e}")
    finally:
        recording_thread.join()

    return np.concatenate(audio_data) if audio_data else np.array([])

def transcribe(audio_np: np.ndarray) -> str:
    """
    Transcribes the given audio data using the Whisper model.
    """
    try:
        result = stt.transcribe(audio_np, fp16=False)
        return result["text"].strip()
    except Exception as e:
        return f"Error during transcription: {e}"

def get_llm_response(text: str) -> str:
    """
    Generates a response using the language model.
    """
    try:
        response = chain.predict(input=text)
        if response.startswith("Assistant:"):
            response = response[len("Assistant:"):].strip()
        return response
    except Exception as e:
        return "Error in generating the response."

def play_audio_with_kokoro(text: str) -> np.ndarray:
    """
    Generate and return speech audio using Kokoro from the input text.
    """
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = build_model('kokoro-v0_19.pth', device)
        voice_name = 'af'
        voicepack = torch.load(f'voices/{voice_name}.pt', weights_only=True).to(device)

        audio, _ = generate(model, text, voicepack, lang=voice_name[0])
        return audio
    except Exception as e:
        raise RuntimeError(f"Error generating or playing audio: {e}")

# API endpoint to process the audio input and provide output
@app.post("/process-audio", response_model=ResponseModel)
def process_audio():
    """
    Process input audio from the microphone, transcribe it to text, and return synthesized audio and transcription.
    """
    try:
        # Record audio
        audio_np = record_audio_with_timeout()

        if audio_np.size == 0:
            raise HTTPException(status_code=400, detail="No audio input detected.")

        # Transcribe audio to text
        transcription = transcribe(audio_np)

        # Generate a response from the language model
        response_text = get_llm_response(transcription)

        # Synthesize audio response
        response_audio = play_audio_with_kokoro(response_text)

        # Save the audio as a file in memory
        audio_io = io.BytesIO()
        sd.write(audio_io, response_audio, 24000, format='WAV')
        audio_io.seek(0)

        # Return the transcription and an audio URL (file is served as static content, for example)
        return JSONResponse(content={
            "transcription": transcription,
            "audio_url": "data:audio/wav;base64," + base64.b64encode(audio_io.getvalue()).decode("utf-8")
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
