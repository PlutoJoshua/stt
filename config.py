import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
HUGGING_FACE_TOKEN = os.getenv('HUGGING_FACE_TOKEN')
STT_METHOD = os.getenv('STT_METHOD', 'whisper_local')
SUMMARIZE_METHOD = os.getenv('SUMMARIZE_METHOD', 'local_model')

SUPPORTED_AUDIO_FORMATS = ['.mp3', '.mp4', '.wav', '.m4a', '.flac', '.ogg']
MAX_FILE_SIZE_MB = 25  # OpenAI Whisper API limit
