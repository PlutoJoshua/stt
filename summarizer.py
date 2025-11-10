"""
이 모듈은 TextSummarizer 클래스를 정의하며, 전략 패턴을 사용하여
실제 텍스트 요약 로직을 동적으로 설정합니다.
"""

from summarizer_strategies import BaseSummarizeStrategy, get_available_summarize_methods

class TextSummarizer:
    """
    TextSummarizer는 특정 요약 전략을 주입받아 텍스트 요약을 수행합니다.
    """
    def __init__(self, strategy: BaseSummarizeStrategy):
        """
        요약 전략 객체로 서비스를 초기화합니다.
        """
        self.strategy = strategy
        self.method = strategy.__class__.__name__

    def summarize(self, text: str, summary_type="general", context=None, include_timestamps=False) -> str:
        """
        주입된 전략을 사용하여 텍스트를 요약합니다.
        """
        if not text or len(text.strip()) < 50:
            return "요약하기에는 텍스트가 너무 짧습니다."
        
        print(f"TextSummarizer가 '{self.method}' 전략을 사용하여 요약을 시작합니다...")
        return self.strategy.summarize(text, summary_type, context, include_timestamps)

    def create_bullet_points(self, text: str, context=None, include_timestamps=False) -> str:
        """
        주입된 전략을 사용하여 텍스트를 불릿 포인트로 요약합니다.
        """
        if not text or len(text.strip()) < 50:
            return "요약하기에는 텍스트가 너무 짧습니다."

        print(f"TextSummarizer가 '{self.method}' 전략을 사용하여 불릿 포인트 요약을 시작합니다...")
        return self.strategy.create_bullet_points(text, context, include_timestamps)

    @staticmethod
    def get_available_methods() -> list:
        """
        사용 가능한 모든 요약 방법의 목록을 반환합니다.
        """
        return get_available_summarize_methods()

    @staticmethod
    def save_summary(summary: str, output_file: str):
        """요약을 파일로 저장"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(summary)
            print(f"요약 저장 완료: {output_file}")
        except Exception as e:
            # 실제 프로덕션에서는 로깅하는 것이 좋습니다.
            print(f"요약 저장 실패: {str(e)}")
