#!/usr/bin/env python3

import click
import config
from processor import process_file
from stt_service import STTService
from summarizer import TextSummarizer

# Get available methods for Click options
try:
    available_stt_methods = STTService.get_available_methods()
except Exception:
    available_stt_methods = ['whisper_api', 'whisper_local', 'whisper_local_diarize'] # Fallback

try:
    available_summarize_methods = TextSummarizer.get_available_methods()
except Exception:
    available_summarize_methods = ['openai_api', 'local_model', 'ollama', 'gemini_api'] # Fallback

@click.command()
@click.argument('audio_file', type=click.Path(exists=True))
@click.option('--output-dir', '-o', default='./output', help='출력 디렉토리 (기본값: ./output)')
@click.option('--summary-type', '-t', 
              type=click.Choice(['general', 'meeting', 'lecture', 'interview', 'daily_conversation']),
              default='general',
              help='요약 유형 (기본값: general)')
@click.option('--stt-method', '-s',
              type=click.Choice(available_stt_methods),
              help='STT 방법 (기본값: 설정파일 값)')
@click.option('--summarize-method', '-m',
              type=click.Choice(available_summarize_methods),
              help='요약 방법 (기본값: 설정파일 값)')
@click.option('--include-timestamps-in-summary', is_flag=True, help='요약에 타임스탬프 포함')
@click.option('--context-file', type=click.Path(exists=True), help='요약에 참고할 컨텍스트 파일 경로')
@click.option('--no-summary', is_flag=True, help='요약하지 않고 텍스트 변환만 수행')
@click.option('--bullet-points', is_flag=True, help='불릿 포인트 형식으로 요약')
@click.option('--chunk-duration', default=10, help='(사용되지 않음) 긴 오디오 분할 시간 (분, 기본값: 10)')
def process_audio_command(audio_file, output_dir, summary_type, stt_method, summarize_method, 
                         context_file, no_summary, bullet_points, chunk_duration, include_timestamps_in_summary):
    """음성 파일을 텍스트로 변환하고 요약하는 프로그램"""
    try:
        results = process_file(
            audio_file=audio_file,
            output_dir=output_dir,
            stt_method=stt_method,
            summarize_method=summarize_method,
            summary_type=summary_type,
            context_file=context_file,
            no_summary=no_summary,
            bullet_points=bullet_points,
            include_timestamps_in_summary=include_timestamps_in_summary
        )
        
        if results and results.get("summary"):
            print("\n" + "="*50)
            print("📄 요약 결과 미리보기:")
            print("="*50)
            summary = results["summary"]
            print(summary[:500] + ("..." if len(summary) > 500 else ""))
            print("="*50)
        
        if results:
            print(f"\n📁 출력 디렉토리: {output_dir}")

    except Exception as e:
        # The processor already logs the error, so we just abort.
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
    print(f"  HuggingFace 토큰: {'설정됨' if config.HUGGING_FACE_TOKEN else '미설정'}")
    
    print("\n📋 사용 가능한 방법:")
    
    try:
        available_stt = STTService.get_available_methods()
        print(f"  STT: {', '.join(available_stt)}")
    except Exception as e:
        print(f"  STT: 확인 실패 ({str(e)})")
    
    try:
        available_summarize = TextSummarizer.get_available_methods()
        print(f"  요약: {', '.join(available_summarize)}")
    except Exception as e:
        print(f"  요약: 확인 실패 ({str(e)})")
    
    print(f"\n🎵 지원 오디오 형식: {config.SUPPORTED_AUDIO_FORMATS}")

if __name__ == '__main__':
    cli.add_command(process_audio_command, name='process')
    cli()