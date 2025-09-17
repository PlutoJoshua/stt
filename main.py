#!/usr/bin/env python3

import click
import os
from pathlib import Path
from datetime import datetime

from audio_processor import AudioProcessor
from stt_service import STTService  
from summarizer import TextSummarizer
import config

@click.command()
@click.argument('audio_file', type=click.Path(exists=True))
@click.option('--output-dir', '-o', default='./output', help='출력 디렉토리 (기본값: ./output)')
@click.option('--summary-type', '-t', 
              type=click.Choice(['general', 'meeting', 'lecture', 'interview']),
              default='general',
              help='요약 유형 (기본값: general)')
@click.option('--stt-method', '-s',
              type=click.Choice(['whisper_api', 'whisper_local', 'whisper_local_diarize']),
              help='STT 방법 (기본값: 설정파일 값)')
@click.option('--summarize-method', '-m',
              type=click.Choice(['openai_api', 'local_model', 'ollama', 'gemini_api']),
              help='요약 방법 (기본값: 설정파일 값)')
@click.option('--context-file', type=click.Path(exists=True), help='요약에 참고할 컨텍스트 파일 경로')
@click.option('--no-summary', is_flag=True, help='요약하지 않고 텍스트 변환만 수행')
@click.option('--bullet-points', is_flag=True, help='불릿 포인트 형식으로 요약')
@click.option('--chunk-duration', default=10, help='(사용되지 않음) 긴 오디오 분할 시간 (분, 기본값: 10)')
def process_audio(audio_file, output_dir, summary_type, stt_method, summarize_method, 
                 context_file, no_summary, bullet_points, chunk_duration):
    """음성 파일을 텍스트로 변환하고 요약하는 프로그램"""
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        audio_path = Path(audio_file)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{audio_path.stem}_{timestamp}"
        transcript_file = Path(output_dir) / f"{base_name}_transcript.txt"
        summary_file = Path(output_dir) / f"{base_name}_summary.md"
        
        print(f"🎵 오디오 파일 처리 시작: {audio_file}")
        
        audio_processor = AudioProcessor()
        audio_info = audio_processor.get_audio_info(audio_file)
        print(f"📊 파일 정보: {audio_info['duration_formatted']}, "
              f"{audio_info['file_size_mb']:.1f}MB, "
              f"{audio_info['channels']}채널, {audio_info['frame_rate']}Hz")
        
        final_stt_method = stt_method if stt_method else config.STT_METHOD
        converted_wav_file = audio_processor.convert_to_wav(audio_file, stt_method=final_stt_method)
        
        stt_service = STTService(method=final_stt_method)
        transcript = stt_service.transcribe(converted_wav_file)
        
        if os.path.exists(converted_wav_file):
            os.remove(converted_wav_file)
        
        stt_service.save_transcript(transcript, transcript_file)
        print(f"📝 텍스트 변환 완료: {transcript_file}")
        
        if not no_summary:
            context_text = None
            if context_file:
                with open(context_file, 'r', encoding='utf-8') as f:
                    context_text = f.read()
                print(f"ℹ️ 컨텍스트 파일 로드: {context_file}")

            summarizer = TextSummarizer()
            
            if summarize_method:
                summarizer.method = summarize_method
            
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
*요약 생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
            summarizer.save_summary(md_summary, summary_file)
            print(f"📋 요약 완료: {summary_file}")
            
            print("\n" + "="*50)
            print("📄 요약 결과 미리보기:")
            print("="*50)
            print(summary[:500] + ("..." if len(summary) > 500 else ""))
            print("="*50)
        
        print(f"\n✅ 처리 완료!")
        print(f"📁 출력 디렉토리: {output_dir}")
        
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")
        raise click.Abort()

@click.group()
def cli():
    """음성 파일 텍스트 변환 및 요약 도구"""
    pass

@cli.command()
def info():
    """사용 가능한 STT 및 요약 방법을 확인합니다."""
    print("🔧 현재 설정:")
    print(f"  STT 방법: {config.STT_METHOD}")
    print(f"  요약 방법: {config.SUMMARIZE_METHOD}")
    print(f"  OpenAI API 키: {'설정됨' if config.OPENAI_API_KEY else '미설정'}")
    print(f"  Google API 키: {'설정됨' if config.GOOGLE_API_KEY else '미설정'}")
    
    print("\n📋 사용 가능한 방법:")
    
    try:
        stt_service = STTService()
        available_stt = stt_service.get_available_methods()
        print(f"  STT: {', '.join(available_stt)}")
    except Exception as e:
        print(f"  STT: 확인 실패 ({str(e)})")
    
    try:
        summarizer = TextSummarizer()
        available_summarize = summarizer.get_available_methods()
        print(f"  요약: {', '.join(available_summarize)}")
    except Exception as e:
        print(f"  요약: 확인 실패 ({str(e)})")
    
    print(f"\n🎵 지원 오디오 형식: {config.SUPPORTED_AUDIO_FORMATS}")

if __name__ == '__main__':
    cli.add_command(process_audio, name='process')
    cli()
