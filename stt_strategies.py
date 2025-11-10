"""
STT 처리를 위한 전략 패턴 구현입니다.
각 STT 방법(Whisper API, 로컬 모델 등)은 별도의 '전략' 클래스로 캡슐화됩니다.
"""
import os
import openai
import whisper
import torch
from pyannote.audio import Pipeline
from huggingface_hub import HfFolder
from datetime import timedelta
import platform
import config

# MLX-Whisper는 Apple Silicon에서만 사용 가능
mlx_whisper = None
if platform.system() == "Darwin":
    try:
        import mlx_whisper
    except ImportError:
        pass

# --- 1. 기본 전략 인터페이스 ---
class BaseSTTStrategy:
    """모든 STT 전략을 위한 기본 인터페이스"""
    def __init__(self):
        print(f"Initializing strategy: {self.__class__.__name__}")

    def transcribe(self, audio_file: str) -> str:
        """
        오디오 파일을 텍스트로 변환합니다.
        이 메소드는 모든 하위 클래스에서 구현되어야 합니다.
        """
        raise NotImplementedError("transcribe() 메소드가 구현되지 않았습니다.")

    def _format_transcript_with_timestamps(self, whisper_results: dict) -> str:
        """Whisper 결과에서 타임스탬프가 포함된 텍스트를 생성합니다."""
        final_transcript = ""
        if 'segments' not in whisper_results:
            return whisper_results.get('text', '')
            
        for segment in whisper_results['segments']:
            start_time = segment['start']
            start_str = str(timedelta(seconds=round(start_time))).split('.')[0]
            text = segment['text']
            final_transcript += f"[{start_str}] {text.strip()}\n"
        return final_transcript.strip()

# --- 2. 구체적인 전략 클래스들 ---

class WhisperAPIStrategy(BaseSTTStrategy):
    """OpenAI Whisper API를 사용하는 전략"""
    def __init__(self):
        super().__init__()
        if not config.OPENAI_API_KEY:
            raise ValueError("OpenAI API 키가 설정되지 않았습니다.")
        self.openai_client = openai.OpenAI(api_key=config.OPENAI_API_KEY)

    def transcribe(self, audio_file: str) -> str:
        try:
            with open(audio_file, 'rb') as file:
                transcript = self.openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=file,
                    language="ko",
                    response_format="verbose_json"
                )
            return self._format_transcript_with_timestamps(transcript)
        except Exception as e:
            raise RuntimeError(f"Whisper API 음성 인식 실패: {str(e)}")

class WhisperLocalStrategy(BaseSTTStrategy):
    """로컬 Whisper 모델을 사용하는 전략"""
    def __init__(self, model_name="large-v3"):
        super().__init__()
        print(f"로컬 Whisper 모델({model_name})을 로딩 중...")
        # Forcing CPU to avoid MPS sparse tensor bug
        device = "cpu"
        self.whisper_model = whisper.load_model(model_name, device=device)
        print(f"Whisper 모델이 {device}에 로드되었습니다.")

    def transcribe(self, audio_file: str) -> str:
        try:
            result = self.whisper_model.transcribe(audio_file, language="ko", fp16=torch.cuda.is_available(), word_timestamps=True)
            return self._format_transcript_with_timestamps(result)
        except Exception as e:
            raise RuntimeError(f"로컬 Whisper 음성 인식 실패: {str(e)}")

class MLXWhisperStrategy(BaseSTTStrategy):
    """MLX-Whisper (Apple Silicon)를 사용하는 전략"""
    def __init__(self, model_repo="mlx-community/whisper-large-v3-mlx"):
        super().__init__()
        if not mlx_whisper:
            raise RuntimeError("MLX Whisper를 사용할 수 없습니다. 'mlx-whisper'가 설치되었는지, 그리고 Apple Silicon 환경인지 확인해주세요.")
        self.model_repo = model_repo
        print(f"MLX Whisper 백엔드를 사용합니다. 모델({model_repo})은 실행 시 로드됩니다.")

    def transcribe(self, audio_file: str) -> str:
        try:
            result = mlx_whisper.transcribe(
                audio_file, 
                path_or_hf_repo=self.model_repo, 
                language="ko",
                word_timestamps=True
            )
            return self._format_transcript_with_timestamps(result)
        except Exception as e:
            raise RuntimeError(f"MLX Whisper 음성 인식 실패: {str(e)}")

class DiarizeWhisperStrategy(BaseSTTStrategy):
    """화자 분리(Diarization)와 로컬 Whisper를 함께 사용하는 전략"""
    def __init__(self, whisper_model_name="large-v3", diarize_pipeline_name="pyannote/speaker-diarization-3.1"):
        super().__init__()
        # Forcing CPU for stability
        device = torch.device("cpu")

        print(f"로컬 Whisper 모델({whisper_model_name})을 로딩 중...")
        self.whisper_model = whisper.load_model(whisper_model_name, device=device)
        print(f"Whisper 모델이 {device}에 로드되었습니다.")

        if not config.HUGGING_FACE_TOKEN:
            raise ValueError("Hugging Face 인증 토큰이 설정되지 않았습니다. .env 파일에 HUGGING_FACE_TOKEN을 추가해주세요.")
        
        HfFolder.save_token(config.HUGGING_FACE_TOKEN)
        
        print(f"Pyannote-audio 화자 분리 파이프라인({diarize_pipeline_name})을 로딩 중...")
        self.diarization_pipeline = Pipeline.from_pretrained(diarize_pipeline_name, use_auth_token=config.HUGGING_FACE_TOKEN)
        self.diarization_pipeline.to(device)
        print(f"화자 분리 파이프라인이 {device}에 로드되었습니다.")

    def transcribe(self, audio_file: str) -> str:
        print("1/3: 화자 분리 실행 중...")
        diarization = self.diarization_pipeline(audio_file)

        print("2/3: 음성 인식 실행 중 (단어 타임스탬프 포함)...")
        whisper_results = self.whisper_model.transcribe(audio_file, language="ko", word_timestamps=True, fp16=False)
        
        print("3/3: 결과 결합 중...")
        return self._combine_results(diarization, whisper_results)

    def _combine_results(self, diarization, whisper_results):
        """화자 분리 및 음성 인식 결과 결합"""
        word_segments = whisper_results['segments']
        word_speaker_mapping = []
        for segment in word_segments:
            for word in segment['words']:
                word_start, word_end = word['start', 'end']
                word_mid_time = word_start + (word_end - word_start) / 2
                
                speaker = "UNKNOWN"
                for turn, _, speaker_label in diarization.itertracks(yield_label=True):
                    if turn.start <= word_mid_time <= turn.end:
                        speaker = speaker_label
                        break
                word_speaker_mapping.append({'word': word['word'], 'start': word_start, 'end': word_end, 'speaker': speaker})

        final_transcript = ""
        current_speaker = None
        segment_start_time = 0
        segment_text = ""

        for word_info in word_speaker_mapping:
            speaker = word_info['speaker']
            word = word_info['word']
            start_time = word_info['start']

            if current_speaker is None:
                current_speaker = speaker
                segment_start_time = start_time
            
            if speaker != current_speaker:
                start_str = str(timedelta(seconds=round(segment_start_time))).split('.')[0]
                final_transcript += f"[{start_str}] **{current_speaker}**: {segment_text.strip()}\n\n"
                
                current_speaker = speaker
                segment_start_time = start_time
                segment_text = ""

            segment_text += word
        
        if segment_text:
            start_str = str(timedelta(seconds=round(segment_start_time))).split('.')[0]
            final_transcript += f"[{start_str}] **{current_speaker}**: {segment_text.strip()}\n"
            
        return final_transcript.strip()

# --- 3. 전략 팩토리 ---
def get_stt_strategy(method: str) -> BaseSTTStrategy:
    """
    주어진 메소드 이름에 해당하는 STT 전략 객체를 반환합니다.
    """
    if method == 'whisper_api':
        return WhisperAPIStrategy()
    elif method == 'whisper_local':
        return WhisperLocalStrategy()
    elif method == 'whisper_mlx':
        return MLXWhisperStrategy()
    elif method == 'whisper_local_diarize':
        return DiarizeWhisperStrategy()
    else:
        raise ValueError(f"지원하지 않는 STT 방법: {method}")

def get_available_stt_methods() -> list:
    """사용 가능한 STT 방법 목록을 반환합니다."""
    methods = []
    if mlx_whisper:
        methods.append('whisper_mlx')
    
    methods.append('whisper_local')

    if config.OPENAI_API_KEY:
        methods.append('whisper_api')
    if config.HUGGING_FACE_TOKEN:
        methods.append('whisper_local_diarize')
    return methods
