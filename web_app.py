from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import os
import sys
import time
from threading import Lock
from enhanced_space_assistant import EnhancedSpaceScienceAssistant
import base64

# Import voice functionality
try:
    import openai
    from elevenlabs.client import ElevenLabs
    VOICE_ENABLED = True
except ImportError:
    VOICE_ENABLED = False
    print("⚠️  Voice libraries not available. Install with: pip install openai elevenlabs")

app = FastAPI(title="Space Science Assistant API")

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Rate limiting for TTS requests
tts_lock = Lock()
tts_last_request = {}
TTS_COOLDOWN = 2  # seconds between requests per session

# Initialize the space assistant
space_assistant = None

# Pydantic models for request validation
class QuestionRequest(BaseModel):
    question: str

class TextToSpeechRequest(BaseModel):
    text: str

def initialize_assistant():
    global space_assistant
    try:
        space_assistant = EnhancedSpaceScienceAssistant()
        space_assistant.initialize()
        print("Space Science Assistant initialized successfully!")
        return True
    except Exception as e:
        print(f"Error initializing assistant: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize the assistant on startup."""
    if not initialize_assistant():
        print("❌ Failed to initialize the assistant. Please check your configuration.")

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        question = request.question.strip()
        
        if not question:
            raise HTTPException(status_code=400, detail="Please provide a question")
        
        if not space_assistant:
            raise HTTPException(status_code=500, detail="Assistant not initialized")
        
        # Get response from the assistant
        response = space_assistant.ask_question(question)
        
        return {
            'response': response.get('response', ''),
            'sources': response.get('sources', []),
            'topics': response.get('topics', [])
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/topics")
async def get_topics():
    try:
        if not space_assistant:
            raise HTTPException(status_code=500, detail="Assistant not initialized")
        
        topics = space_assistant.get_available_topics()
        return {'topics': topics}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/speech_to_text")
async def speech_to_text(request: Request, audio: UploadFile = File(...)):
    """Convert uploaded audio to text using OpenAI Whisper."""
    try:
        if not VOICE_ENABLED:
            raise HTTPException(status_code=400, detail="Voice features not available")
        
        if not audio.filename:
            raise HTTPException(status_code=400, detail="No audio file selected")
        
        # Use OpenAI Whisper for speech-to-text
        openai_key = os.getenv('OPENAI_API_KEY')
        if not openai_key:
            raise HTTPException(status_code=400, detail="OpenAI API key not configured")
            
        client = openai.OpenAI(api_key=openai_key)
        
        # Read audio file content
        audio_content = await audio.read()
        
        # Convert to format OpenAI can handle
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=(audio.filename or "audio.wav", audio_content, "audio/wav"),
            language="en"
        )
        
        return {
            'text': transcript.text.strip(),
            'success': True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Speech-to-text error: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Speech-to-text error: {str(e)}")

@app.post("/text_to_speech")
async def text_to_speech(request: Request, tts_request: TextToSpeechRequest):
    """Convert text to speech using ElevenLabs."""
    try:
        if not VOICE_ENABLED:
            raise HTTPException(status_code=400, detail="Voice features not available")
        
        # Simple rate limiting based on IP
        client_ip = request.client.host
        current_time = time.time()
        
        with tts_lock:
            if client_ip in tts_last_request:
                time_since_last = current_time - tts_last_request[client_ip]
                if time_since_last < TTS_COOLDOWN:
                    return JSONResponse(
                        status_code=429,
                        content={
                            'error': f'Please wait {TTS_COOLDOWN - time_since_last:.1f} seconds before making another request.',
                            'rate_limited': True
                        }
                    )
            
            tts_last_request[client_ip] = current_time
        
        text = tts_request.text.strip()
        
        if not text:
            raise HTTPException(status_code=400, detail="No text provided")
        
        # Setup ElevenLabs API key
        elevenlabs_key = os.getenv('ELEVENLABS_API_KEY')
        if not elevenlabs_key:
            raise HTTPException(status_code=400, detail="ElevenLabs API key not configured")
        
        # Initialize ElevenLabs client
        client = ElevenLabs(api_key=elevenlabs_key)
        
        # Generate audio using ElevenLabs
        voice_id = "21m00Tcm4TlvDq8ikWAM"  # Default voice
        audio = client.text_to_speech.convert(
            voice_id=voice_id,
            text=text,
            model_id="eleven_monolingual_v1"
        )
        
        # Convert audio to base64 for web transmission
        # ElevenLabs returns an iterator of audio chunks
        audio_bytes = b''.join(audio)
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        return {
            'audio': audio_base64,
            'success': True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        # Handle ElevenLabs rate limiting
        if '429' in error_msg or 'too_many_concurrent_requests' in error_msg:
            return JSONResponse(
                status_code=429,
                content={
                    'error': 'ElevenLabs rate limit exceeded. Please wait a moment and try again.',
                    'rate_limited': True
                }
            )
        raise HTTPException(status_code=500, detail=f"Text-to-speech error: {error_msg}")

@app.get("/voice_status")
async def voice_status():
    """Check if voice features are available."""
    return {
        'voice_enabled': VOICE_ENABLED,
        'openai_key': bool(os.getenv('OPENAI_API_KEY')),
        'elevenlabs_key': bool(os.getenv('ELEVENLABS_API_KEY'))
    }

@app.get("/history")
async def get_history():
    try:
        if not space_assistant:
            raise HTTPException(status_code=500, detail="Assistant not initialized")
        
        history = space_assistant.get_conversation_history()
        return {'history': history}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/clear_history")
async def clear_history():
    try:
        if not space_assistant:
            raise HTTPException(status_code=500, detail="Assistant not initialized")
        
        space_assistant.clear_conversation_history()
        return {'message': 'Conversation history cleared'}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/rebuild_knowledge")
async def rebuild_knowledge():
    """Rebuild the knowledge base with updated information."""
    try:
        if not space_assistant:
            raise HTTPException(status_code=500, detail="Assistant not initialized")
        
        space_assistant.rebuild_knowledge_base()
        return {'success': True, 'message': 'Knowledge base rebuilt successfully'}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    
    print("Starting Space Science Assistant Web Interface...")
    print("\n🚀 Space Science Assistant Web UI is ready!")
    print("🌟 Open your browser and navigate to: http://localhost:5000")
    print("🌙 Explore the cosmos with our AI assistant!\n")
    uvicorn.run(app, host="0.0.0.0", port=10000)
