# 음성 파일 텍스트 변환 및 요약 도구 (STT)

음성 녹음 파일에서 텍스트를 추출하고 요약하는 Python 프로그램입니다. OpenAI Whisper API와 로컬 Whisper 모델, 다양한 LLM API 및 로컬 모델을 이용한 요약을 지원합니다.

## 주요 기능

- 🎵 **다양한 오디오 형식 지원**: MP3, MP4, WAV, M4A, FLAC, OGG
- 🔊 **음성-텍스트 변환 (STT)**:
  - OpenAI Whisper API (유료, 고품질)
  - 로컬 Whisper 모델 (무료, CPU/GPU)
- 📝 **텍스트 요약**:
  - OpenAI GPT API (유료, 고품질)
  - Google Gemini API (유료, 고품질)
  - 로컬 T5 모델 (무료)
  - Ollama 로컬 LLM 지원
- ℹ️ **컨텍스트 제공 요약**: 별도 텍스트 파일을 제공하여 더 정확한 요약 생성
- 📊 **다양한 요약 유형**: 일반, 회의, 강의, 인터뷰
- 🎯 **불릿 포인트** 형식 요약 지원
- 📄 **마크다운 형식 출력**: 요약 결과물을 가독성 좋은 마크다운 파일로 저장

## 설치 방법

### 1. 저장소 클론
```bash
git clone <this-repo>
cd stt
```

### 2. 의존성 설치
```bash
pip install -r requirements.txt
```

### 3. 환경 설정 (선택사항)
`.env` 파일을 생성하여 API 키와 기본 설정을 구성할 수 있습니다.
```env
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here

STT_METHOD=whisper_local  # whisper_api, whisper_local
SUMMARIZE_METHOD=gemini_api # openai_api, gemini_api, local_model, ollama
```

## 사용 방법

### 기본 사용법
```bash
# 간단한 실행 (run.py 사용)
python run.py recording.mp3

# 또는 메인 스크립트 직접 실행
python main.py process recording.mp3
```

### 고급 옵션
```bash
# 출력 디렉토리 지정
python main.py process recording.mp3 -o ./my_output

# 회의 요약 형식 사용
python main.py process meeting.mp4 -t meeting

# 불릿 포인트 형식으로 요약
python main.py process lecture.m4a --bullet-points

# 요약하지 않고 텍스트 변환만
python main.py process interview.wav --no-summary

# STT 방법 지정 (API, 로컬)
python main.py process recording.mp3 -s whisper_api

# 요약 방법 지정
python main.py process recording.mp3 -m gemini_api

# 컨텍스트 파일을 참고하여 요약
python main.py process discussion.mp3 --context-file context.txt
```

### 시스템 정보 확인
현재 설정과 사용 가능한 STT/요약 방법을 확인합니다.
```bash
python main.py info
```

## 출력 파일

프로그램은 `output` 디렉토리(또는 지정된 디렉토리)에 다음 파일들을 생성합니다:
- `{파일명}_{타임스탬프}_transcript.txt`: 원본 텍스트
- `{파일명}_{타임스탬프}_summary.md`: 마크다운 형식 요약본 (옵션)

## 요약 유형

- `general`: 일반적인 요약
- `meeting`: 회의 내용 (주요 논의사항, 결정사항, 액션 아이템)
- `lecture`: 강의 내용 (핵심 개념, 중요 설명, 예시)
- `interview`: 인터뷰 내용 (주요 질문과 답변)

## 비용 및 성능 고려사항

### API 사용 시
- **OpenAI Whisper API**: $0.006/분
- **OpenAI GPT-3.5-turbo**: 비용은 토큰 사용량에 따라 다름
- **Google Gemini API**: 비용은 문자 수에 따라 다름
- 높은 품질과 빠른 처리 속도

### 로컬 모델 사용 시
- **완전 무료**
- 초기 모델 다운로드 필요
- CPU/GPU 리소스 사용
- 인터넷 연결 불필요

## 지원 파일 형식

- MP3, MP4 (핸드폰 녹음)
- WAV, M4A
- FLAC, OGG
- 최대 파일 크기: 25MB (Whisper API 사용 시 제한)

## 문제 해결

### ffmpeg 관련 오류
오디오 파일 처리를 위해 `ffmpeg`이 필요합니다.

macOS:
```bash
brew install ffmpeg
```

Ubuntu/Debian:
```bash
sudo apt update
sudo apt install ffmpeg
```

### GPU 사용 (로컬 Whisper)
CUDA 지원 PyTorch 설치 시 GPU를 사용한 추론이 가능합니다.
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Ollama 사용 시
로컬에서 Ollama 서버를 실행해야 합니다.
```bash
# Ollama 설치 및 모델 다운로드
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama2
ollama serve
```

## 라이선스

MIT License

## 기여

버그 리포트나 기능 요청은 GitHub Issues를 통해 제출해주세요.