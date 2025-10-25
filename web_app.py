from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import os
import time
from threading import Lock
import base64
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import voice functionality
try:
    import openai
    from elevenlabs.client import ElevenLabs
    VOICE_ENABLED = True
    logger.info("✅ Voice libraries loaded successfully")
except ImportError as e:
    VOICE_ENABLED = False
    logger.warning(f"⚠️  Voice libraries not available: {e}")

# Try to import assistant - MAKE IT COMPLETELY OPTIONAL
space_assistant = None
ASSISTANT_AVAILABLE = False

try:
    from enhanced_space_assistant import EnhancedSpaceScienceAssistant
    ASSISTANT_AVAILABLE = True
    logger.info("✅ Assistant module found")
except ImportError as e:
    logger.warning(f"⚠️  Assistant module not available: {e}")
    logger.warning("   App will run in limited mode without AI assistant")

# Initialize FastAPI app
app = FastAPI(
    title="Space Science Assistant API",
    description="AI-powered space science assistant with voice capabilities",
    version="1.0.0"
)

# Mount static files and templates (optional)
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
    templates = Jinja2Templates(directory="templates")
    logger.info("✅ Static files and templates configured")
except Exception as e:
    logger.warning(f"⚠️  Static files or templates not found: {e}")
    templates = None

# Rate limiting for TTS requests
tts_lock = Lock()
tts_last_request = {}
TTS_COOLDOWN = 2

# Pydantic models
class QuestionRequest(BaseModel):
    question: str

class TextToSpeechRequest(BaseModel):
    text: str

def initialize_assistant():
    """Initialize the space science assistant."""
    global space_assistant
    
    if not ASSISTANT_AVAILABLE:
        logger.warning("⚠️  Assistant module not available, skipping initialization")
        return False
    
    try:
        logger.info("🚀 Initializing Space Science Assistant...")
        space_assistant = EnhancedSpaceScienceAssistant()
        
        # Try to initialize
        if hasattr(space_assistant, 'initialize'):
            space_assistant.initialize()
        
        logger.info("✅ Space Science Assistant initialized successfully!")
        return True
    except Exception as e:
        logger.error(f"❌ Error initializing assistant: {e}")
        import traceback
        traceback.print_exc()
        space_assistant = None
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize the assistant on startup."""
    logger.info("=" * 60)
    logger.info("🌟 Starting Space Science Assistant Web Service")
    logger.info("=" * 60)
    
    # Check environment variables
    logger.info("🔍 Checking environment configuration...")
    openai_key = os.getenv('OPENAI_API_KEY')
    elevenlabs_key = os.getenv('ELEVENLABS_API_KEY')
    
    if openai_key:
        logger.info("✅ OpenAI API key found")
    else:
        logger.warning("⚠️  OpenAI API key not set")
    
    if elevenlabs_key:
        logger.info("✅ ElevenLabs API key found")
    else:
        logger.warning("⚠️  ElevenLabs API key not set")
    
    # Initialize assistant (optional)
    if ASSISTANT_AVAILABLE:
        if not initialize_assistant():
            logger.warning("⚠️  Assistant initialization failed. Service will continue without AI features.")
    else:
        logger.warning("⚠️  Assistant module not available. Service will run in limited mode.")
    
    logger.info("=" * 60)
    logger.info("✅ Service startup complete!")
    logger.info("=" * 60)

@app.get("/")
async def index(request: Request):
    """Serve the main web interface."""
    if templates:
        return templates.TemplateResponse("index.html", {"request": request})
    else:
        return JSONResponse(
            content={
                "message": "Space Science Assistant API is running",
                "status": "healthy",
                "assistant_available": ASSISTANT_AVAILABLE,
                "assistant_initialized": space_assistant is not None,
                "endpoints": {
                    "health": "/health",
                    "ask": "/ask",
                    "topics": "/topics",
                    "history": "/history",
                    "voice_status": "/voice_status"
                }
            }
        )

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "assistant_available": ASSISTANT_AVAILABLE,
        "assistant_initialized": space_assistant is not None,
        "voice_enabled": VOICE_ENABLED,
        "timestamp": time.time(),
        "port": os.getenv('PORT', '10000')
    }

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """Ask a question to the space science assistant."""
    try:
        question = request.question.strip()
        
        if not question:
            raise HTTPException(status_code=400, detail="Please provide a question")
        
        if not space_assistant:
            # Return a friendly message instead of error
            return {
                'response': f"I received your question: '{question}'. However, the AI assistant is not currently initialized. Please check the server logs or try again later.",
                'sources': [],
                'topics': [],
                'status': 'assistant_not_available'
            }
        
        logger.info(f"📝 Question received: {question[:100]}...")
        
        # Get response from the assistant
        response = space_assistant.ask_question(question)
        
        logger.info(f"✅ Response generated successfully")
        
        return {
            'response': response.get('response', ''),
            'sources': response.get('sources', []),
            'topics': response.get('topics', []),
            'status': 'success'
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error processing question: {e}")
        return {
            'response': f"An error occurred while processing your question. Error: {str(e)}",
            'sources': [],
            'topics': [],
            'status': 'error'
        }

@app.get("/topics")
async def get_topics():
    """Get available topics from the assistant."""
    try:
        if not space_assistant:
            return {'topics': [], 'message': 'Assistant not initialized'}
        
        topics = space_assistant.get_available_topics()
        return {'topics': topics}
        
    except Exception as e:
        logger.error(f"❌ Error fetching topics: {e}")
        return {'topics': [], 'error': str(e)}

@app.post("/speech_to_text")
async def speech_to_text(request: Request, audio: UploadFile = File(...)):
    """Convert uploaded audio to text using OpenAI Whisper."""
    try:
        if not VOICE_ENABLED:
            raise HTTPException(
                status_code=400, 
                detail="Voice features not available. Please install required libraries."
            )
        
        if not audio.filename:
            raise HTTPException(status_code=400, detail="No audio file selected")
        
        openai_key = os.getenv('OPENAI_API_KEY')
        if not openai_key:
            raise HTTPException(
                status_code=400, 
                detail="OpenAI API key not configured."
            )
        
        logger.info(f"🎤 Processing audio file: {audio.filename}")
        
        client = openai.OpenAI(api_key=openai_key)
        audio_content = await audio.read()
        
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=(audio.filename or "audio.wav", audio_content, "audio/wav"),
            language="en"
        )
        
        logger.info(f"✅ Transcription successful")
        
        return {
            'text': transcript.text.strip(),
            'success': True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Speech-to-text error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Speech-to-text error: {str(e)}")

@app.post("/text_to_speech")
async def text_to_speech(request: Request, tts_request: TextToSpeechRequest):
    """Convert text to speech using ElevenLabs."""
    try:
        if not VOICE_ENABLED:
            raise HTTPException(
                status_code=400, 
                detail="Voice features not available."
            )
        
        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()
        
        with tts_lock:
            if client_ip in tts_last_request:
                time_since_last = current_time - tts_last_request[client_ip]
                if time_since_last < TTS_COOLDOWN:
                    wait_time = TTS_COOLDOWN - time_since_last
                    return JSONResponse(
                        status_code=429,
                        content={
                            'error': f'Please wait {wait_time:.1f} seconds.',
                            'rate_limited': True
                        }
                    )
            tts_last_request[client_ip] = current_time
        
        text = tts_request.text.strip()
        
        if not text:
            raise HTTPException(status_code=400, detail="No text provided")
        
        if len(text) > 5000:
            raise HTTPException(status_code=400, detail="Text too long. Maximum 5000 characters.")
        
        elevenlabs_key = os.getenv('ELEVENLABS_API_KEY')
        if not elevenlabs_key:
            raise HTTPException(status_code=400, detail="ElevenLabs API key not configured.")
        
        logger.info(f"🔊 Generating speech ({len(text)} chars)...")
        
        client = ElevenLabs(api_key=elevenlabs_key)
        voice_id = "21m00Tcm4TlvDq8ikWAM"
        
        audio = client.text_to_speech.convert(
            voice_id=voice_id,
            text=text,
            model_id="eleven_monolingual_v1"
        )
        
        audio_bytes = b''.join(audio)
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        logger.info(f"✅ Speech generated successfully")
        
        return {
            'audio': audio_base64,
            'success': True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Text-to-speech error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/voice_status")
async def voice_status():
    """Check if voice features are available."""
    return {
        'voice_enabled': VOICE_ENABLED,
        'openai_configured': bool(os.getenv('OPENAI_API_KEY')),
        'elevenlabs_configured': bool(os.getenv('ELEVENLABS_API_KEY')),
        'speech_to_text_available': VOICE_ENABLED and bool(os.getenv('OPENAI_API_KEY')),
        'text_to_speech_available': VOICE_ENABLED and bool(os.getenv('ELEVENLABS_API_KEY'))
    }

@app.get("/history")
async def get_history():
    """Get conversation history."""
    try:
        if not space_assistant:
            return {'history': [], 'message': 'Assistant not initialized'}
        
        history = space_assistant.get_conversation_history()
        return {'history': history}
        
    except Exception as e:
        logger.error(f"❌ Error fetching history: {e}")
        return {'history': [], 'error': str(e)}

@app.post("/clear_history")
async def clear_history():
    """Clear conversation history."""
    try:
        if not space_assistant:
            return {'message': 'Assistant not initialized', 'success': False}
        
        space_assistant.clear_conversation_history()
        logger.info("🗑️  Conversation history cleared")
        return {'message': 'Conversation history cleared', 'success': True}
        
    except Exception as e:
        logger.error(f"❌ Error clearing history: {e}")
        return {'message': str(e), 'success': False}

@app.post("/rebuild_knowledge")
async def rebuild_knowledge():
    """Rebuild the knowledge base."""
    try:
        if not space_assistant:
            raise HTTPException(status_code=503, detail="Assistant not initialized")
        
        logger.info("🔄 Rebuilding knowledge base...")
        space_assistant.rebuild_knowledge_base()
        logger.info("✅ Knowledge base rebuilt successfully")
        
        return {
            'success': True, 
            'message': 'Knowledge base rebuilt successfully'
        }
    except Exception as e:
        logger.error(f"❌ Error rebuilding knowledge base: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Handle 404 errors."""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint not found",
            "path": str(request.url.path)
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred."
        }
    )

if __name__ == '__main__':
    import uvicorn
    
    # CRITICAL: Get port from environment
    port = int(os.getenv('PORT', 10000))
    host = '0.0.0.0'  # Must be 0.0.0.0 for Render
    
    # Log configuration
    print("\n" + "=" * 60)
    print("🚀 Starting Space Science Assistant Web Interface")
    print("=" * 60)
    print(f"🌟 Host: {host}")
    print(f"🌟 Port: {port}")
    print(f"🌟 PORT env: {os.getenv('PORT', 'NOT SET - using default 10000')}")
    print(f"🌙 Environment: {'Production' if os.getenv('PORT') else 'Development'}")
    print(f"🌙 Voice Enabled: {VOICE_ENABLED}")
    print(f"🌙 Assistant Available: {ASSISTANT_AVAILABLE}")
    print("=" * 60)
    print(f"\n✨ Server starting on http://{host}:{port}")
    print(f"✨ Health check: http://localhost:{port}/health")
    print(f"✨ API docs: http://localhost:{port}/docs\n")
    
    # Run the application
    uvicorn.run(
        app, 
        host=host, 
        port=port, 
        log_level="info",
        access_log=True
    )
