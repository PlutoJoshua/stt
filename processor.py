import os
import uuid
from pathlib import Path
from datetime import datetime

from audio_processor import AudioProcessor
from models import get_stt_service, get_summarizer
import config

def process_file(audio_files, output_dir, stt_method, summarize_method, summary_type, 
                 context_file, no_summary, bullet_points, include_timestamps_in_summary, status_callback=None):
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
        log(f"총 {len(audio_files)}개의 파일 처리를 시작합니다.")
        os.makedirs(output_dir, exist_ok=True)
        
        all_transcripts = []
        total_duration = 0
        total_size_mb = 0

        final_stt_method = stt_method if stt_method else config.STT_METHOD
        stt_service = get_stt_service(final_stt_method)
        audio_processor = AudioProcessor()

        for i, audio_file in enumerate(audio_files):
            log(f"\n[{i+1}/{len(audio_files)}] 🎵 오디오 파일 처리 시작: {os.path.basename(audio_file)}")
            
            audio_info = audio_processor.get_audio_info(audio_file)
            total_duration += audio_info['duration']
            total_size_mb += audio_info['file_size_mb']
            log(f"📊 파일 정보: {audio_info['duration_formatted']}, {audio_info['file_size_mb']:.1f}MB")

            converted_wav_file = None
            try:
                log(f"🔊 오디오를 WAV 형식으로 변환 중...")
                converted_wav_file = audio_processor.convert_to_wav(
                    audio_file,
                    stt_method=final_stt_method,
                )

                log(f"✍️ 음성-텍스트 변환 시작 (방법: {final_stt_method})...")
                transcript = stt_service.transcribe(converted_wav_file)
                all_transcripts.append(transcript)
            finally:
                if converted_wav_file and os.path.exists(converted_wav_file):
                    os.remove(converted_wav_file)
            log(f"📝 텍스트 변환 완료.")

        # 모든 텍스트를 하나로 합치기
        full_transcript = "\n\n--- 다음 파일 ---\n\n".join(all_transcripts)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"combined_{timestamp}_{uuid.uuid4().hex[:8]}"
        transcript_file = Path(output_dir) / f"{base_name}_transcript.txt"
        summary_file = Path(output_dir) / f"{base_name}_summary.md"

        with open(transcript_file, 'w', encoding='utf-8') as f:
            f.write(full_transcript)
        log(f"\n📚 모든 텍스트가 다음 파일에 저장되었습니다: {transcript_file}")

        # 오디오 정보 종합
        minutes, seconds = divmod(total_duration, 60)
        total_duration_formatted = f"{int(minutes)}분 {int(seconds)}초"
        combined_audio_info = {
            'duration_formatted': total_duration_formatted,
            'file_size_mb': total_size_mb,
            'num_files': len(audio_files)
        }

        summary = None
        if not no_summary and full_transcript.strip():
            context_text = None
            if context_file:
                with open(context_file, 'r', encoding='utf-8') as f:
                    context_text = f.read()
                log(f"ℹ️ 컨텍스트 파일 로드: {context_file}")

            final_summarize_method = summarize_method if summarize_method else config.SUMMARIZE_METHOD
            log(f"📋 텍스트 요약 시작 (방법: {final_summarize_method})...")
            summarizer = get_summarizer(final_summarize_method)
            
            if bullet_points:
                summary = summarizer.create_bullet_points(full_transcript, context=context_text, include_timestamps=include_timestamps_in_summary)
            else:
                summary = summarizer.summarize(full_transcript, summary_type, context=context_text, include_timestamps=include_timestamps_in_summary)
            
            md_summary = f"""# 📝 음성 기록 요약

## 🎙️ 원본 오디오 파일
- **파일 개수:** `{len(audio_files)}`
- **총 길이:** `{total_duration_formatted}`

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
            "transcript": full_transcript,
            "summary": summary,
            "audio_info": combined_audio_info
        }
        
    except Exception as e:
        log(f"❌ 오류 발생: {str(e)}")
        raise
