import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
HUGGING_FACE_TOKEN = os.getenv('HUGGING_FACE_TOKEN')
STT_METHOD = os.getenv('STT_METHOD', 'whisper_local')
SUMMARIZE_METHOD = os.getenv('SUMMARIZE_METHOD', 'local_model')

# Gemini 모델 설정
GEMINI_MODEL_FOR_SUMMARY = os.getenv('MODEL', 'gemini-2.5-flash')
GEMINI_MODEL_FOR_FINAL_SUMMARY = os.getenv('MODEL', 'gemini-2.5-flash')

# Claude 모델 설정
CLAUDE_MODEL = os.getenv('CLAUDE_MODEL', 'claude-opus-4-6')

SUPPORTED_AUDIO_FORMATS = ['.mp3', '.mp4', '.wav', '.m4a', '.flac', '.ogg']
MAX_FILE_SIZE_MB = 25  # OpenAI Whisper API limit
