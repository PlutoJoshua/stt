# 음성 파일 텍스트 변환 및 요약 도구

음성 녹음 파일을 텍스트로 변환하고 요약하는 웹 애플리케이션 및 CLI 도구입니다. OpenAI Whisper, 로컬 Whisper, 다양한 LLM API 및 로컬 모델을 이용한 요약을 지원합니다.

## 주요 기능

- 🌐 **웹 인터페이스**: 사용하기 쉬운 웹 UI에서 파일을 업로드하고 실시간으로 처리 과정을 확인할 수 있습니다.
- 🎵 **다양한 오디오 형식 지원**: MP3, MP4, WAV, M4A, FLAC, OGG
- 🔊 **음성-텍스트 변환 (STT)**:
  - OpenAI Whisper API (유료, 고품질)
  - 로컬 Whisper 모델 (무료, CPU/GPU)
  - 화자 분리 지원 로컬 Whisper (무료, `pyannote.audio` 사용)
- 📝 **텍스트 요약**:
  - OpenAI GPT API (유료, 고품질)
  - Google Gemini API (유료, 고품질)
  - 로컬 T5 모델 (무료)
  - Ollama 로컬 LLM 지원
- 📊 **다양한 요약 유형**: 일반, 회의, 강의, 인터뷰
- 🎯 **불릿 포인트** 형식 요약 지원
- 📄 **마크다운 형식 출력**: 요약 결과물을 가독성 좋은 마크다운 파일로 다운로드할 수 있습니다.

## 설치 및 실행 방법

### 1. 저장소 클론
```bash
git clone <this-repo>
cd stt
```

### 2. 의존성 설치
```bash
pip install -r requirements.txt
```

### 3. 환경 설정 (API 키)
프로젝트 루트 디렉토리에 `.env` 파일을 생성하고 사용하는 API 키를 추가합니다. 로컬 모델만 사용하는 경우 이 단계는 선택사항입니다.

```env
# 예시: .env 파일 내용
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
HUGGING_FACE_TOKEN=your_huggingface_token_for_pyannote
```
- `HUGGING_FACE_TOKEN`은 화자 분리(`whisper_local_diarize`) 기능을 사용할 때 필요합니다.

### 4. 웹 애플리케이션 실행
다음 명령어를 실행하여 웹 서버를 시작합니다.

```bash
python app.py
```

서버가 실행되면 웹 브라우저에서 `http://127.0.0.1:5001` 주소로 접속하여 파일을 업로드하고 변환을 시작할 수 있습니다.

## CLI (명령줄 인터페이스) 사용법

웹 UI 대신 터미널에서 직접 파일을 처리할 수도 있습니다.

### 기본 사용법
```bash
# 메인 스크립트 직접 실행
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

# STT 방법 지정 (API, 로컬, 화자분리)
python main.py process recording.mp3 -s whisper_local_diarize

# 요약 방법 지정
python main.py process recording.mp3 -m gemini_api
```

### 시스템 정보 확인
현재 설정과 사용 가능한 STT/요약 방법을 확인합니다.
```bash
python main.py info
```

## 문제 해결

### ffmpeg 관련 오류
오디오 파일 처리를 위해 `ffmpeg`이 필요합니다.

- **macOS**: `brew install ffmpeg`
- **Ubuntu/Debian**: `sudo apt update && sudo apt install ffmpeg`

### GPU 사용 (로컬 Whisper)
CUDA 지원 PyTorch 설치 시 GPU를 사용한 추론이 가능합니다.
```bash
# 기존 torch, torchaudio 삭제 후 CUDA 버전에 맞게 재설치
pip uninstall torch torchaudio
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
