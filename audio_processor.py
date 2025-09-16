import os
from pathlib import Path
from pydub import AudioSegment
import config

class AudioProcessor:
    def __init__(self):
        self.supported_formats = config.SUPPORTED_AUDIO_FORMATS
        self.max_size_mb = config.MAX_FILE_SIZE_MB
    
    def validate_file(self, file_path, check_size=True):
        """오디오 파일 유효성 검사"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
        
        file_ext = Path(file_path).suffix.lower()
        if file_ext not in self.supported_formats:
            raise ValueError(f"지원하지 않는 파일 형식입니다. 지원 형식: {self.supported_formats}")
        
        if check_size:
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            if file_size_mb > self.max_size_mb:
                raise ValueError(f"파일 크기가 너무 큽니다. 최대 크기: {self.max_size_mb}MB")
        
        return True
    
    def convert_to_wav(self, input_path, output_path=None, stt_method=None):
        """오디오 파일을 WAV 형식으로 변환"""
        # API를 사용할 때만 파일 크기 검사
        check_size_flag = True if stt_method == 'whisper_api' else False
        self.validate_file(input_path, check_size=check_size_flag)
        
        if output_path is None:
            input_file = Path(input_path)
            output_path = input_file.parent / f"{input_file.stem}_converted.wav"
        
        try:
            # 파일 확장자에 따라 적절한 형식으로 로드
            input_ext = Path(input_path).suffix.lower()
            
            if input_ext == '.mp4':
                audio = AudioSegment.from_file(input_path, format="mp4")
            elif input_ext == '.m4a':
                audio = AudioSegment.from_file(input_path, format="m4a")
            else:
                audio = AudioSegment.from_file(input_path)
            
            # 16kHz, 모노로 변환 (Whisper 최적화)
            audio = audio.set_frame_rate(16000).set_channels(1)
            
            # WAV 형식으로 저장
            audio.export(output_path, format="wav")
            
            print(f"오디오 파일 변환 완료: {output_path}")
            return str(output_path)
            
        except Exception as e:
            raise RuntimeError(f"오디오 변환 실패: {str(e)}")
    
    def get_audio_info(self, file_path):
        """오디오 파일 정보 반환"""
        # API 사용 여부와 관계없이 파일 유효성만 검사 (크기 제외)
        self.validate_file(file_path, check_size=False)
        
        try:
            audio = AudioSegment.from_file(file_path)
            duration_seconds = len(audio) / 1000.0
            
            info = {
                'duration': duration_seconds,
                'duration_formatted': f"{int(duration_seconds//60)}:{int(duration_seconds%60):02d}",
                'frame_rate': audio.frame_rate,
                'channels': audio.channels,
                'file_size_mb': os.path.getsize(file_path) / (1024 * 1024)
            }
            
            return info
            
        except Exception as e:
            raise RuntimeError(f"오디오 정보 가져오기 실패: {str(e)}")
    
    def split_audio(self, file_path, chunk_duration_minutes=10):
        """긴 오디오 파일을 작은 청크로 분할"""
        self.validate_file(file_path, check_size=False)
        
        try:
            audio = AudioSegment.from_file(file_path)
            chunk_duration_ms = chunk_duration_minutes * 60 * 1000
            
            chunks = []
            input_file = Path(file_path)
            
            for i, start_time in enumerate(range(0, len(audio), chunk_duration_ms)):
                chunk = audio[start_time:start_time + chunk_duration_ms]
                chunk_path = input_file.parent / f"{input_file.stem}_chunk_{i+1:03d}.wav"
                chunk.export(chunk_path, format="wav")
                chunks.append(str(chunk_path))
            
            print(f"오디오 파일을 {len(chunks)}개 청크로 분할했습니다.")
            return chunks
            
        except Exception as e:
            raise RuntimeError(f"오디오 분할 실패: {str(e)}")