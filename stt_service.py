"""
이 모듈은 STTService 클래스를 정의하며, 전략 패턴을 사용하여
실제 음성-텍스트 변환 로직을 동적으로 설정합니다.
"""

from stt_strategies import BaseSTTStrategy, get_available_stt_methods

class STTService:
    """
    STTService는 특정 STT 전략을 주입받아 음성-텍스트 변환을 수행합니다.
    """
    def __init__(self, strategy: BaseSTTStrategy):
        """
        STT 전략 객체로 서비스를 초기화합니다.

        :param strategy: BaseSTTStrategy를 상속받은 STT 전략 객체
        """
        self.strategy = strategy
        self.method = strategy.__class__.__name__ # 전략의 클래스 이름으로 메소드 설정

    def transcribe(self, audio_file: str) -> str:
        """
        주입된 전략을 사용하여 오디오 파일을 텍스트로 변환합니다.

        :param audio_file: 변환할 오디오 파일의 경로
        :return: 변환된 텍스트
        """
        print(f"STTService가 '{self.method}' 전략을 사용하여 변환을 시작합니다...")
        return self.strategy.transcribe(audio_file)

    @staticmethod
    def get_available_methods() -> list:
        """
        사용 가능한 모든 STT 방법의 목록을 반환합니다.
        실제 로직은 stt_strategies 모듈에 위임합니다.
        """
        return get_available_stt_methods()