import os
import openai
import whisper
import torch
from pyannote.audio import Pipeline
from huggingface_hub import HfApi, HfFolder
from datetime import timedelta
import numpy as np
import config
import platform

# Try to import mlx_whisper
mlx_whisper = None
if platform.system() == "Darwin":
    try:
        import mlx_whisper
    except ImportError:
        print("MLX Whisper not found. To use it, run: pip install mlx-whisper")
        pass

class STTService:
    def __init__(self, method=None):
        if method:
            self.method = method
        else:
            self.method = config.STT_METHOD
        
        self.openai_client = None
        self.whisper_model = None
        self.diarization_pipeline = None

        print(f"STT 서비스 초기화 (방법: {self.method})")

        if self.method == 'whisper_api':
            if not config.OPENAI_API_KEY:
                raise ValueError("OpenAI API 키가 설정되지 않았습니다.")
            self.openai_client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
        
        elif self.method in ['whisper_local', 'whisper_local_diarize']:
            print("로컬 Whisper 모델(large-v3)을 로딩 중...")
            # Forcing CPU to avoid MPS sparse tensor bug
            device = "cpu"

            self.whisper_model = whisper.load_model("large-v3", device=device)
            print(f"Whisper 모델이 {device}에 로드되었습니다.")

            if self.method == 'whisper_local_diarize':
                if not config.HUGGING_FACE_TOKEN:
                    raise ValueError("Hugging Face 인증 토큰이 설정되지 않았습니다. .env 파일에 HUGGING_FACE_TOKEN을 추가해주세요.")
                
                HfFolder.save_token(config.HUGGING_FACE_TOKEN)
                
                print("Pyannote-audio 화자 분리 파이프라인을 로딩 중...")
                pipeline_name = "pyannote/speaker-diarization-3.1"
                self.diarization_pipeline = Pipeline.from_pretrained(pipeline_name, use_auth_token=config.HUGGING_FACE_TOKEN)
                # Forcing to CPU as well for stability
                self.diarization_pipeline.to(torch.device(device))
                print(f"화자 분리 파이프라인이 {device}에 로드되었습니다.")

        elif self.method == 'whisper_mlx':
            if not mlx_whisper:
                raise RuntimeError("MLX Whisper를 사용할 수 없습니다. 'mlx-whisper'가 설치되었는지, 그리고 Apple Silicon 환경인지 확인해주세요.")
            print("MLX Whisper 백엔드를 사용합니다. 모델은 실행 시 로드됩니다.")

    def transcribe_with_api(self, audio_file):
        """OpenAI Whisper API를 사용한 음성 인식"""
        try:
            with open(audio_file, 'rb') as file:
                transcript = self.openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=file,
                    language="ko"
                )
            return transcript.text
        except Exception as e:
            raise RuntimeError(f"Whisper API 음성 인식 실패: {str(e)}")

    def transcribe_with_local(self, audio_file):
        """로컬 Whisper 모델을 사용한 음성 인식"""
        try:
            if self.whisper_model is None:
                raise RuntimeError("로컬 Whisper 모델이 초기화되지 않았습니다.")
            print("음성 인식 중...")
            result = self.whisper_model.transcribe(audio_file, language="ko", fp16=torch.cuda.is_available())
            return result["text"]
        except Exception as e:
            raise RuntimeError(f"로컬 Whisper 음성 인식 실패: {str(e)}")

    def transcribe_with_mlx(self, audio_file):
        """MLX Whisper를 사용한 음성 인식"""
        try:
            if not mlx_whisper:
                raise RuntimeError("MLX Whisper가 초기화되지 않았습니다.")
            print("MLX Whisper로 음성 인식 중...")
            # Using large-v3 model for consistency with the local pytorch version
            result = mlx_whisper.transcribe(
                audio_file, 
                path_or_hf_repo="mlx-community/whisper-large-v3-mlx", 
                language="ko"
            )
            return result["text"]
        except Exception as e:
            raise RuntimeError(f"MLX Whisper 음성 인식 실패: {str(e)}")

    def transcribe_with_diarization(self, audio_file):
        """로컬 모델을 사용한 화자 분리 및 음성 인식"""
        if self.whisper_model is None or self.diarization_pipeline is None:
            raise RuntimeError("화자 분리 모델이 초기화되지 않았습니다.")

        print("1/3: 화자 분리 실행 중...")
        diarization = self.diarization_pipeline(audio_file)

        print("2/3: 음성 인식 실행 중 (단어 타임스탬프 포함)...")
        whisper_results = self.whisper_model.transcribe(audio_file, language="ko", word_timestamps=True, fp16=torch.cuda.is_available())
        
        print("3/3: 결과 결합 중...")
        return self._combine_results(diarization, whisper_results)

    def _combine_results(self, diarization, whisper_results):
        """화자 분리 및 음성 인식 결과 결합"""
        word_segments = whisper_results['segments']
        
        # 각 단어에 화자 할당
        word_speaker_mapping = []
        for segment in word_segments:
            for word in segment['words']:
                word_start, word_end = word['start'], word['end']
                # 단어의 중간 지점을 기준으로 화자 찾기
                word_mid_time = word_start + (word_end - word_start) / 2
                
                speaker = "UNKNOWN"
                for turn, _, speaker_label in diarization.itertracks(yield_label=True):
                    if turn.start <= word_mid_time <= turn.end:
                        speaker = speaker_label
                        break
                word_speaker_mapping.append({'word': word['word'], 'start': word_start, 'end': word_end, 'speaker': speaker})

        # 화자별로 텍스트 재구성
        final_transcript = ""
        current_speaker = None
        segment_start_time = None
        segment_text = ""

        for word_info in word_speaker_mapping:
            speaker = word_info['speaker']
            word = word_info['word']
            start_time = word_info['start']

            if current_speaker is None:
                current_speaker = speaker
                segment_start_time = start_time
            
            if speaker != current_speaker:
                # 이전 세그먼트 완료
                start_str = str(timedelta(seconds=segment_start_time)).split('.')[0]
                final_transcript += f"[{start_str}] **{current_speaker}**: {segment_text.strip()}\n\n"
                
                # 새 세그먼트 시작
                current_speaker = speaker
                segment_start_time = start_time
                segment_text = ""

            segment_text += word
        
        # 마지막 세그먼트 추가
        if segment_text:
            start_str = str(timedelta(seconds=segment_start_time)).split('.')[0]
            final_transcript += f"[{start_str}] **{current_speaker}**: {segment_text.strip()}\n"
            
        return final_transcript.strip()

    def transcribe(self, audio_file):
        """설정된 방법에 따라 음성을 텍스트로 변환"""
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"오디오 파일을 찾을 수 없습니다: {audio_file}")

        print(f"음성 인식 시작 ({self.method})...")

        if self.method == 'whisper_api':
            return self.transcribe_with_api(audio_file)
        elif self.method == 'whisper_mlx':
            return self.transcribe_with_mlx(audio_file)
        elif self.method == 'whisper_local':
            return self.transcribe_with_local(audio_file)
        elif self.method == 'whisper_local_diarize':
            return self.transcribe_with_diarization(audio_file)
        else:
            raise ValueError(f"지원하지 않는 STT 방법: {self.method}")

    def save_transcript(self, text, output_file):
        """변환된 텍스트를 파일로 저장"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"텍스트 저장 완료: {output_file}")
        except Exception as e:
            raise RuntimeError(f"텍스트 저장 실패: {str(e)}")

    def get_available_methods(self):
        """사용 가능한 STT 방법 반환"""
        methods = []
        # MLX is preferred on Mac
        if mlx_whisper:
            methods.append('whisper_mlx')
        
        methods.append('whisper_local')

        if config.OPENAI_API_KEY:
            methods.append('whisper_api')
        if config.HUGGING_FACE_TOKEN:
            methods.append('whisper_local_diarize')
        return methods