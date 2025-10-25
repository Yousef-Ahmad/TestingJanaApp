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
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import voice functionality
try:
    import openai
    from elevenlabs.client import ElevenLabs
    VOICE_ENABLED = True
    logger.info("✅ Voice libraries loaded successfully")
except ImportError as e:
    VOICE_ENABLED = False
    logger.warning(f"⚠️  Voice libraries not available: {e}")
    logger.warning("Install with: pip install openai elevenlabs")

# Initialize FastAPI app
app = FastAPI(
    title="Space Science Assistant API",
    description="AI-powered space science assistant with voice capabilities",
    version="1.0.0"
)

# Mount static files and templates
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
TTS_COOLDOWN = 2  # seconds between requests per session

# Initialize the space assistant
space_assistant = None

# Pydantic models for request validation
class QuestionRequest(BaseModel):
    question: str

class TextToSpeechRequest(BaseModel):
    text: str

def initialize_assistant():
    """Initialize the space science assistant."""
    global space_assistant
    try:
        logger.info("🚀 Initializing Space Science Assistant...")
        space_assistant = EnhancedSpaceScienceAssistant()
        space_assistant.initialize()
        logger.info("✅ Space Science Assistant initialized successfully!")
        return True
    except Exception as e:
        logger.error(f"❌ Error initializing assistant: {e}")
        import traceback
        traceback.print_exc()
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
    
    # Initialize assistant
    if not initialize_assistant():
        logger.error("❌ Failed to initialize the assistant. Please check your configuration.")
        logger.warning("⚠️  Service will continue but assistant features may not work.")
    
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
                "endpoints": {
                    "ask": "/ask",
                    "topics": "/topics",
                    "history": "/history",
                    "voice_status": "/voice_status",
                    "health": "/health"
                }
            }
        )

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "assistant_initialized": space_assistant is not None,
        "voice_enabled": VOICE_ENABLED,
        "timestamp": time.time()
    }

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """Ask a question to the space science assistant."""
    try:
        question = request.question.strip()
        
        if not question:
            raise HTTPException(status_code=400, detail="Please provide a question")
        
        if not space_assistant:
            raise HTTPException(
                status_code=503, 
                detail="Assistant not initialized. Please try again in a moment."
            )
        
        logger.info(f"📝 Question received: {question[:100]}...")
        
        # Get response from the assistant
        response = space_assistant.ask_question(question)
        
        logger.info(f"✅ Response generated successfully")
        
        return {
            'response': response.get('response', ''),
            'sources': response.get('sources', []),
            'topics': response.get('topics', [])
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error processing question: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/topics")
async def get_topics():
    """Get available topics from the assistant."""
    try:
        if not space_assistant:
            raise HTTPException(
                status_code=503, 
                detail="Assistant not initialized"
            )
        
        topics = space_assistant.get_available_topics()
        return {'topics': topics}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error fetching topics: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

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
        
        # Use OpenAI Whisper for speech-to-text
        openai_key = os.getenv('OPENAI_API_KEY')
        if not openai_key:
            raise HTTPException(
                status_code=400, 
                detail="OpenAI API key not configured. Please set OPENAI_API_KEY environment variable."
            )
        
        logger.info(f"🎤 Processing audio file: {audio.filename}")
        
        client = openai.OpenAI(api_key=openai_key)
        
        # Read audio file content
        audio_content = await audio.read()
        logger.info(f"📊 Audio file size: {len(audio_content)} bytes")
        
        # Convert to format OpenAI can handle
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=(audio.filename or "audio.wav", audio_content, "audio/wav"),
            language="en"
        )
        
        logger.info(f"✅ Transcription successful: {transcript.text[:100]}...")
        
        return {
            'text': transcript.text.strip(),
            'success': True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Speech-to-text error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Speech-to-text error: {str(e)}")

@app.post("/text_to_speech")
async def text_to_speech(request: Request, tts_request: TextToSpeechRequest):
    """Convert text to speech using ElevenLabs."""
    try:
        if not VOICE_ENABLED:
            raise HTTPException(
                status_code=400, 
                detail="Voice features not available. Please install required libraries."
            )
        
        # Simple rate limiting based on IP
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
                            'error': f'Please wait {wait_time:.1f} seconds before making another request.',
                            'rate_limited': True
                        }
                    )
            
            tts_last_request[client_ip] = current_time
        
        text = tts_request.text.strip()
        
        if not text:
            raise HTTPException(status_code=400, detail="No text provided")
        
        if len(text) > 5000:
            raise HTTPException(
                status_code=400, 
                detail="Text too long. Maximum 5000 characters."
            )
        
        # Setup ElevenLabs API key
        elevenlabs_key = os.getenv('ELEVENLABS_API_KEY')
        if not elevenlabs_key:
            raise HTTPException(
                status_code=400, 
                detail="ElevenLabs API key not configured. Please set ELEVENLABS_API_KEY environment variable."
            )
        
        logger.info(f"🔊 Generating speech for text ({len(text)} chars)...")
        
        # Initialize ElevenLabs client
        client = ElevenLabs(api_key=elevenlabs_key)
        
        # Generate audio using ElevenLabs
        voice_id = "21m00Tcm4TlvDq8ikWAM"  # Default voice (Rachel)
        audio = client.text_to_speech.convert(
            voice_id=voice_id,
            text=text,
            model_id="eleven_monolingual_v1"
        )
        
        # Convert audio to base64 for web transmission
        # ElevenLabs returns an iterator of audio chunks
        audio_bytes = b''.join(audio)
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        logger.info(f"✅ Speech generated successfully ({len(audio_bytes)} bytes)")
        
        return {
            'audio': audio_base64,
            'success': True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ Text-to-speech error: {error_msg}")
        
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
    openai_available = bool(os.getenv('OPENAI_API_KEY'))
    elevenlabs_available = bool(os.getenv('ELEVENLABS_API_KEY'))
    
    return {
        'voice_enabled': VOICE_ENABLED,
        'openai_configured': openai_available,
        'elevenlabs_configured': elevenlabs_available,
        'speech_to_text_available': VOICE_ENABLED and openai_available,
        'text_to_speech_available': VOICE_ENABLED and elevenlabs_available
    }

@app.get("/history")
async def get_history():
    """Get conversation history."""
    try:
        if not space_assistant:
            raise HTTPException(
                status_code=503, 
                detail="Assistant not initialized"
            )
        
        history = space_assistant.get_conversation_history()
        return {'history': history}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error fetching history: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/clear_history")
async def clear_history():
    """Clear conversation history."""
    try:
        if not space_assistant:
            raise HTTPException(
                status_code=503, 
                detail="Assistant not initialized"
            )
        
        space_assistant.clear_conversation_history()
        logger.info("🗑️  Conversation history cleared")
        return {'message': 'Conversation history cleared', 'success': True}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error clearing history: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/rebuild_knowledge")
async def rebuild_knowledge():
    """Rebuild the knowledge base with updated information."""
    try:
        if not space_assistant:
            raise HTTPException(
                status_code=503, 
                detail="Assistant not initialized"
            )
        
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

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Handle 404 errors."""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint not found",
            "path": str(request.url.path),
            "message": "The requested resource does not exist"
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
            "message": "An unexpected error occurred. Please try again later."
        }
    )

if __name__ == '__main__':
    import uvicorn
    
    # Get port from environment variable (Render provides this)
    # Default to 10000 for local development
    port = int(os.getenv('PORT', 10000))
    host = os.getenv('HOST', '0.0.0.0')
    
    # Log configuration
    print("\n" + "=" * 60)
    print("🚀 Starting Space Science Assistant Web Interface")
    print("=" * 60)
    print(f"🌟 Host: {host}")
    print(f"🌟 Port: {port}")
    print(f"🌙 Environment: {'Production' if os.getenv('PORT') else 'Development'}")
    print(f"🌙 Voice Enabled: {VOICE_ENABLED}")
    print("=" * 60)
    print(f"\n✨ Server will be available at: http://{host}:{port}")
    print("✨ Health check endpoint: http://localhost:{port}/health")
    print("✨ API documentation: http://localhost:{port}/docs")
    print("\n🌌 Explore the cosmos with our AI assistant!\n")
    
    # Run the application
    uvicorn.run(
        app, 
        host=host, 
        port=port, 
        log_level="info",
        access_log=True
    )
