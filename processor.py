import os
from pathlib import Path
from datetime import datetime

from audio_processor import AudioProcessor
from stt_service import STTService
from summarizer import TextSummarizer
import config

def process_file(audio_file, output_dir, stt_method, summarize_method, summary_type, 
                 context_file, no_summary, bullet_points, status_callback=None):
    """
    음성 파일을 처리하는 핵심 로직입니다.
    CLI와 웹 UI에서 모두 호출할 수 있도록 분리되었습니다.
    status_callback: 진행 상황을 전달받을 콜백 함수.
    """
    def log(message):
        """상태 콜백이 있으면 호출하고, 없으면 print합니다."""
        if status_callback:
            status_callback(message)
        else:
            print(message)

    try:
        os.makedirs(output_dir, exist_ok=True)
        audio_path = Path(audio_file)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{audio_path.stem}_{timestamp}"
        transcript_file = Path(output_dir) / f"{base_name}_transcript.txt"
        summary_file = Path(output_dir) / f"{base_name}_summary.md"
        
        log(f"🎵 오디오 파일 처리 시작: {audio_file}")
        
        audio_processor = AudioProcessor()
        audio_info = audio_processor.get_audio_info(audio_file)
        log(f"📊 파일 정보: {audio_info['duration_formatted']}, "
            f"{audio_info['file_size_mb']:.1f}MB, "
            f"{audio_info['channels']}채널, {audio_info['frame_rate']}Hz")
        
        log("🔊 오디오를 WAV 형식으로 변환 중...")
        final_stt_method = stt_method if stt_method else config.STT_METHOD
        converted_wav_file = audio_processor.convert_to_wav(audio_file, stt_method=final_stt_method)
        
        log(f"✍️ 음성-텍스트 변환 시작 (방법: {final_stt_method})...")
        stt_service = STTService(method=final_stt_method)
        transcript = stt_service.transcribe(converted_wav_file)
        
        if os.path.exists(converted_wav_file):
            os.remove(converted_wav_file)
        
        stt_service.save_transcript(transcript, transcript_file)
        log(f"📝 텍스트 변환 완료: {transcript_file}")
        
        summary = None
        if not no_summary:
            context_text = None
            if context_file:
                with open(context_file, 'r', encoding='utf-8') as f:
                    context_text = f.read()
                log(f"ℹ️ 컨텍스트 파일 로드: {context_file}")

            final_summarize_method = summarize_method if summarize_method else config.SUMMARIZE_METHOD
            log(f"📋 텍스트 요약 시작 (방법: {final_summarize_method})...")
            summarizer = TextSummarizer()
            summarizer.method = final_summarize_method
            
            if bullet_points:
                summary = summarizer.create_bullet_points(transcript, context=context_text)
            else:
                summary = summarizer.summarize(transcript, summary_type, context=context_text)
            
            md_summary = f"""# 📝 음성 기록 요약

## 🎙️ 원본 오디오 파일
- **파일:** `{os.path.basename(audio_file)}`
- **길이:** `{audio_info['duration_formatted']}`

## 📜 요약 내용
{summary}

---
*요약 생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"""
            summarizer.save_summary(md_summary, summary_file)
            log(f"📋 요약 완료: {summary_file}")
        
        log(f"\n✅ 처리 완료!")
        
        return {
            "transcript_file": str(transcript_file),
            "summary_file": str(summary_file) if summary else None,
            "transcript": transcript,
            "summary": summary,
            "audio_info": audio_info
        }
        
    except Exception as e:
        log(f"❌ 오류 발생: {str(e)}")
        raise e
