from stt_service import STTService
from summarizer import TextSummarizer
from stt_strategies import get_stt_strategy
from summarizer_strategies import get_summarize_strategy

# Private cache for model instances
_stt_services = {}
_summarizers = {}

def get_stt_service(method: str) -> STTService:
    """
    요청된 메소드에 대한 STTService 인스턴스를 캐시하고 반환합니다.
    """
    if method not in _stt_services:
        print(f"Creating STT service for method: {method}...")
        strategy = get_stt_strategy(method)
        _stt_services[method] = STTService(strategy)
        print(f"STT service for {method} created and cached.")
    return _stt_services[method]

def get_summarizer(method: str) -> TextSummarizer:
    """
    요청된 메소드에 대한 TextSummarizer 인스턴스를 캐시하고 반환합니다.
    """
    if method not in _summarizers:
        print(f"Creating Summarizer service for method: {method}...")
        strategy = get_summarize_strategy(method)
        _summarizers[method] = TextSummarizer(strategy)
        print(f"Summarizer service for {method} created and cached.")
    return _summarizers[method]
