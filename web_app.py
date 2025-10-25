from flask import Flask, render_template, request, jsonify
import os
import sys
import io
import wave
import time
from threading import Lock
from enhanced_space_assistant import EnhancedSpaceScienceAssistant

# Import voice functionality
try:
    import openai
    from elevenlabs.client import ElevenLabs
    VOICE_ENABLED = True
except ImportError:
    VOICE_ENABLED = False
    print("⚠️  Voice libraries not available. Install with: pip install openai elevenlabs")

app = Flask(__name__)

# Rate limiting for TTS requests
tts_lock = Lock()
tts_last_request = {}
TTS_COOLDOWN = 2  # seconds between requests per session

# Initialize the space assistant
space_assistant = None

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({'error': 'Please provide a question'}), 400
        
        if not space_assistant:
            return jsonify({'error': 'Assistant not initialized'}), 500
        
        # Get response from the assistant
        response = space_assistant.ask_question(question)
        
        return jsonify({
            'response': response.get('response', ''),
            'sources': response.get('sources', []),
            'topics': response.get('topics', [])
        })
        
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/topics')
def get_topics():
    try:
        if not space_assistant:
            return jsonify({'error': 'Assistant not initialized'}), 500
        
        topics = space_assistant.get_available_topics()
        return jsonify({'topics': topics})
        
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/speech_to_text', methods=['POST'])
def speech_to_text():
    """Convert uploaded audio to text using OpenAI Whisper."""
    try:
        if not VOICE_ENABLED:
            return jsonify({'error': 'Voice features not available'}), 400
        
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'No audio file selected'}), 400
        
        # Use OpenAI Whisper for speech-to-text
        openai_key = os.getenv('OPENAI_API_KEY')
        if not openai_key:
            return jsonify({'error': 'OpenAI API key not configured'}), 400
            
        client = openai.OpenAI(api_key=openai_key)
        
        # Convert FileStorage to a file-like object that OpenAI can handle
        audio_file.seek(0)  # Reset file pointer to beginning
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=(audio_file.filename or "audio.wav", audio_file.read(), "audio/wav"),
            language="en"
        )
        
        return jsonify({
            'text': transcript.text.strip(),
            'success': True
        })
        
    except Exception as e:
        print(f"Speech-to-text error: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Speech-to-text error: {str(e)}'}), 500

@app.route('/text_to_speech', methods=['POST'])
def text_to_speech():
    """Convert text to speech using ElevenLabs."""
    try:
        if not VOICE_ENABLED:
            return jsonify({'error': 'Voice features not available'}), 400
        
        # Simple rate limiting based on IP
        client_ip = request.remote_addr
        current_time = time.time()
        
        with tts_lock:
            if client_ip in tts_last_request:
                time_since_last = current_time - tts_last_request[client_ip]
                if time_since_last < TTS_COOLDOWN:
                    return jsonify({
                        'error': f'Please wait {TTS_COOLDOWN - time_since_last:.1f} seconds before making another request.',
                        'rate_limited': True
                    }), 429
            
            tts_last_request[client_ip] = current_time
        
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Setup ElevenLabs API key
        elevenlabs_key = os.getenv('ELEVENLABS_API_KEY')
        if not elevenlabs_key:
            return jsonify({'error': 'ElevenLabs API key not configured'}), 400
        
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
        import base64
        # ElevenLabs returns an iterator of audio chunks
        audio_bytes = b''.join(audio)
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        return jsonify({
            'audio': audio_base64,
            'success': True
        })
        
    except Exception as e:
        error_msg = str(e)
        # Handle ElevenLabs rate limiting
        if '429' in error_msg or 'too_many_concurrent_requests' in error_msg:
            return jsonify({
                'error': 'ElevenLabs rate limit exceeded. Please wait a moment and try again.',
                'rate_limited': True
            }), 429
        return jsonify({'error': f'Text-to-speech error: {error_msg}'}), 500

@app.route('/voice_status')
def voice_status():
    """Check if voice features are available."""
    return jsonify({
        'voice_enabled': VOICE_ENABLED,
        'openai_key': bool(os.getenv('OPENAI_API_KEY')),
        'elevenlabs_key': bool(os.getenv('ELEVENLABS_API_KEY'))
    })

@app.route('/history')
def get_history():
    try:
        if not space_assistant:
            return jsonify({'error': 'Assistant not initialized'}), 500
        
        history = space_assistant.get_conversation_history()
        return jsonify({'history': history})
        
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/clear_history', methods=['POST'])
def clear_history():
    try:
        if not space_assistant:
            return jsonify({'error': 'Assistant not initialized'}), 500
        
        space_assistant.clear_conversation_history()
        return jsonify({'message': 'Conversation history cleared'})
        
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/rebuild_knowledge', methods=['POST'])
def rebuild_knowledge():
    """Rebuild the knowledge base with updated information."""
    try:
        if not space_assistant:
            return jsonify({'error': 'Assistant not initialized'}), 500
        
        space_assistant.rebuild_knowledge_base()
        return jsonify({'success': True, 'message': 'Knowledge base rebuilt successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
if __name__ == '__main__':
    print("Starting Space Science Assistant Web Interface...")
    
    if initialize_assistant():
        print("\n🚀 Space Science Assistant Web UI is ready!")
        print("🌟 Open your browser and navigate to: http://localhost:5000")
        print("🌙 Explore the cosmos with our AI assistant!\n")
        port = int(os.environ.get("PORT", 10000))  # ✅ CORRECT
        app.run(host="0.0.0.0", port=port)
    else:
        print("❌ Failed to initialize the assistant. Please check your configuration.")
        sys.exit(1)
