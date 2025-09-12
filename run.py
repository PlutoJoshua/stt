#!/usr/bin/env python3

"""
간편한 실행 스크립트 
python run.py [audio_file] 형태로 사용
"""

import sys
from main import process_audio

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("사용법: python run.py <audio_file> [options]")
        print("예시: python run.py recording.mp3")
        print("예시: python run.py recording.mp4 --no-summary")
        sys.exit(1)
    
    # 첫 번째 인자를 audio_file로 전달하고 나머지는 옵션으로 처리
    import click
    ctx = click.Context(process_audio)
    
    try:
        # sys.argv[1:]를 직접 파싱
        args = sys.argv[1:]
        process_audio.main(args, standalone_mode=False)
    except Exception as e:
        print(f"오류: {e}")
        sys.exit(1)