# 음성 텍스트 변환 및 요약 도구

음성 파일을 텍스트로 변환하고 내용을 요약하는 웹 애플리케이션 및 CLI 도구입니다. OpenAI Whisper, 로컬 Whisper 및 여러 LLM API를 포함한 다양한 STT 및 요약 방법을 지원합니다.

## 목차

- [주요 기능](#주요-기능)
- [프로젝트 구조](#프로젝트-구조)
- [설치 방법](#설치-방법)
- [환경 설정](#환경-설정)
- [사용법](#사용법)
  - [웹 인터페이스](#웹-인터페이스)
  - [명령줄 인터페이스 (CLI)](#명령줄-인터페이스-cli)
- [문제 해결](#문제-해결)
- [향후 개선 사항](#향후-개선-사항)
- [라이선스](#라이선스)

## 주요 기능

- 🌐 **웹 인터페이스**: 파일 업로드 및 실시간 진행 상황 추적을 위한 사용하기 쉬운 UI.
- 🎵 **다양한 오디오 형식 지원**: MP3, MP4, WAV, M4A, FLAC, OGG.
- 🔊 **음성-텍스트 변환 (STT)**:
  - OpenAI Whisper API (유료, 고품질)
  - 로컬 Whisper 모델 (무료, CPU/GPU)
  - 화자 분리 기능이 포함된 로컬 Whisper (무료, `pyannote.audio` 사용)
- 📝 **텍스트 요약**:
  - OpenAI GPT API (유료, 고품질)
  - Google Gemini API (유료, 고품질)
  - 로컬 T5 모델 (무료)
  - Ollama를 통한 로컬 LLM 지원
- 📊 **다양한 요약 유형**: 일반, 회의, 강의, 인터뷰 형식.
- 🎯 **불릿 포인트 요약**: 요약 결과를 불릿 포인트 형식으로 생성하는 옵션.
- 📄 **마크다운 출력**: 요약 결과를 마크다운 파일로 다운로드.

## 프로젝트 구조

```
/
├─── app.py               # Flask 웹 애플리케이션
├─── cli.py               # CLI 시작점
├─── processor.py         # STT 및 요약을 위한 핵심 처리 로직
├─── audio_processor.py   # 오디오 파일 변환 및 처리 담당
├─── stt_service.py       # 다양한 STT 서비스(API, 로컬) 관리
├─── summarizer.py        # 다양한 요약 서비스 관리
├─── config.py            # 환경 설정 로더 (.env 파일)
├─── requirements.txt     # Python 의존성 목록
├─── templates/           # 웹 앱을 위한 HTML 템플릿
│    └─── index.html
├─── static/              # 정적 파일 (CSS, JS)
├─── uploads/             # 업로드된 오디오 파일의 기본 디렉토리
└─── output/              # 출력 파일(텍스트, 요약)의 기본 디렉토리
```

## 설치 방법

1.  **저장소 클론:**
    ```bash
    git clone <this-repo>
    cd stt
    ```

2.  **의존성 설치:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **ffmpeg 설치:**
    오디오 처리를 위해 필요합니다.
    - **macOS**: `brew install ffmpeg`
    - **Ubuntu/Debian**: `sudo apt update && sudo apt install ffmpeg`

## 환경 설정

프로젝트 루트 디렉토리에 `.env` 파일을 생성하여 API 키 및 기타 설정을 저장합니다. 로컬 모델만 사용하려는 경우 이 단계는 선택 사항입니다.

```env
# .env 파일 예시
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
HUGGING_FACE_TOKEN=your_huggingface_token_for_pyannote
```

- `HUGGING_FACE_TOKEN`은 화자 분리 기능(`whisper_local_diarize`)에 필요합니다.

## 사용법

### 웹 인터페이스

Flask 웹 서버를 시작합니다:

```bash
python app.py
```

웹 브라우저를 열고 `http://127.0.0.1:5001` 주소로 이동하여 애플리케이션을 사용합니다.

### 명령줄 인터페이스 (CLI)

터미널에서 직접 파일을 처리할 수도 있습니다.

**기본 사용법:**
```bash
python cli.py process recording.mp3
```

**고급 옵션:**
```bash
# 출력 디렉토리 지정
python cli.py process recording.mp3 -o ./my_output

# '회의' 요약 형식 사용
python cli.py process meeting.mp4 -t meeting

# 불릿 포인트 요약 생성
python cli.py process lecture.m4a --bullet-points

# 요약 없이 변환만 수행
python cli.py process interview.wav --no-summary

# STT 방법 지정 (api, local, diarize)
python cli.py process recording.mp3 -s whisper_local_diarize

# 요약 방법 지정
python cli.py process recording.mp3 -m gemini_api
```

**시스템 정보 확인:**
현재 설정 및 사용 가능한 방법을 보려면:
```bash
python cli.py info
```

## 문제 해결

### GPU 사용 (로컬 Whisper)

GPU 가속 변환을 위해 CUDA 지원 버전의 PyTorch를 설치하세요.

```bash
# 기존 torch 및 torchaudio 제거
pip uninstall torch torchaudio

# CUDA 지원 버전으로 재설치 (CUDA 11.8 예시)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Ollama 사용

Ollama와 함께 로컬 LLM을 사용하려면 Ollama 서버가 실행 중인지 확인하세요.

```bash
# Ollama 설치 및 모델 다운로드
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama2

# Ollama 서버 시작
ollama serve
```

## 향후 개선 사항

- **개선된 이름 지정:** 출력 파일 이름을 더 설명적으로 만듭니다.
- **메타데이터 저장:** 요약과 함께 파일 정보를 저장합니다.
- **명확한 화자 분리:** 텍스트에서 화자 분리의 명확성을 향상시킵니다.

## 라이선스

MIT 라이선스
