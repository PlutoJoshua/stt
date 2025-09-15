import os
import openai
import whisper
from pathlib import Path
import config

class STTService:
    def __init__(self, method=None):
        if method:
            self.method = method
        else:
            self.method = config.STT_METHOD
        
        self.openai_client = None
        self.whisper_model = None

        print(f"STT 서비스 초기화 (방법: {self.method})")

        if self.method == 'whisper_api':
            if not config.OPENAI_API_KEY:
                raise ValueError("OpenAI API 키가 설정되지 않았습니다.")
            self.openai_client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
        elif self.method == 'whisper_local':
            print("로컬 Whisper 모델(large-v3)을 로딩 중... (CPU)")
            self.whisper_model = whisper.load_model("large-v3")

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
            print("CPU로 음성 인식 중...")
            result = self.whisper_model.transcribe(audio_file, language="ko")
            return result["text"]
        except Exception as e:
            raise RuntimeError(f"로컬 Whisper 음성 인식 실패: {str(e)}")

    def transcribe(self, audio_file):
        """설정된 방법에 따라 음성을 텍스트로 변환"""
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"오디오 파일을 찾을 수 없습니다: {audio_file}")

        print(f"음성 인식 시작 ({self.method})...")

        if self.method == 'whisper_api':
            return self.transcribe_with_api(audio_file)
        elif self.method == 'whisper_local':
            return self.transcribe_with_local(audio_file)
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
        methods = ['whisper_local']
        if config.OPENAI_API_KEY:
            methods.append('whisper_api')
        return methods