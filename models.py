from stt_service import STTService
from summarizer import TextSummarizer

# Private cache for model instances
_stt_services = {}
_summarizer = None

def get_stt_service(method):
    """
    Gets a cached STTService instance for the given method.
    Loads the model on first request.
    """
    if method not in _stt_services:
        print(f"Loading model for STT method: {method}...")
        _stt_services[method] = STTService(method=method)
        print(f"Model for {method} loaded.")
    return _stt_services[method]

def get_summarizer():
    """
    Gets a cached TextSummarizer instance.
    The internal models are lazy-loaded by the class itself.
    """
    global _summarizer
    if _summarizer is None:
        print("Initializing TextSummarizer...")
        _summarizer = TextSummarizer()
    return _summarizer
