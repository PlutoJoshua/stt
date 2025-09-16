import openai
import requests
import google.generativeai as genai
from transformers import pipeline
import config

class TextSummarizer:
    def __init__(self):
        self.method = config.SUMMARIZE_METHOD
        self.openai_client = None
        self.local_summarizer = None
        self.gemini_model = None
        
    def _initialize_client(self):
        """필요한 API 클라이언트를 초기화합니다."""
        if self.method == 'openai_api' and not self.openai_client:
            if not config.OPENAI_API_KEY:
                raise ValueError("OpenAI API 키가 설정되지 않았습니다.")
            self.openai_client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
        
        elif self.method == 'gemini_api' and not self.gemini_model:
            if not config.GOOGLE_API_KEY:
                raise ValueError("Google API 키가 설정되지 않았습니다.")
            genai.configure(api_key=config.GOOGLE_API_KEY)
            self.gemini_model = genai.GenerativeModel('gemini-2.5-pro')

        elif self.method == 'local_model' and not self.local_summarizer:
            print("로컬 요약 모델을 로딩 중...")
            self.local_summarizer = pipeline(
                "summarization",
                model="eenzeenee/t5-small-korean-summarization",
                device=-1
            )

    def summarize_with_openai(self, text, summary_type="general", context=None):
        """OpenAI API를 사용한 텍스트 요약"""
        try:
            system_prompts = {
                "general": "다음 텍스트를 명확하게 요약해주세요.",
                "meeting": "다음 회의 내용을 자세히 요약해주세요. 주요 논의사항, 결정사항, 액션 아이템을 중심으로 정리해주세요.",
                "lecture": "다음 강의 내용을 요약해주세요. 핵심 개념, 중요한 설명, 예시를 중심으로 정리해주세요.",
                "interview": "다음 인터뷰 내용을 요약해주세요. 주요 질문과 답변, 핵심 내용을 중심으로 정리해주세요."
            }
            instruction = system_prompts.get(summary_type, system_prompts["meeting"])
            
            system_content = f"{instruction}"
            if context:
                system_content += f"\n\n[사전 정보]\n{context}"
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": text}
                ],
                max_tokens=1500, # OpenAI는 max_tokens 지정이 안정적
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"OpenAI 요약 실패: {str(e)}")

    def summarize_with_gemini(self, text, summary_type="general", context=None):
        """Gemini API를 사용한 텍스트 요약"""
        try:
            system_prompts = {
                "general": "다음 텍스트를 명확하게 요약해주세요.",
                "meeting": "다음 회의 내용을 자세히 요약해주세요. 주요 논의사항, 결정사항, 액션 아이템을 중심으로 정리해주세요.",
                "lecture": "다음 강의 내용을 요약해주세요. 핵심 개념, 중요한 설명, 예시를 중심으로 정리해주세요.",
                "interview": "다음 인터뷰 내용을 요약해주세요. 주요 질문과 답변, 핵심 내용을 중심으로 정리해주세요."
            }
            instruction = system_prompts.get(summary_type, system_prompts["meeting"])
            
            prompt_parts = [f"[요약 지시]\n{instruction}"]
            if context:
                prompt_parts.append(f"\n[사전 정보]\n{context}")
            
            prompt_parts.append(f"\n\n--- 요약할 텍스트 시작 ---\n{text}\n--- 요약할 텍스트 끝 ---")
            
            prompt = "\n".join(prompt_parts)
            
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]

            response = self.gemini_model.generate_content(
                prompt, 
                safety_settings=safety_settings
            )
            return response.text
        except Exception as e:
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                 raise RuntimeError(f"Gemini 요약 실패: 프롬프트가 안전 설정에 의해 차단되었습니다. Reason: {response.prompt_feedback.block_reason}")
            if not response.parts:
                 raise RuntimeError(f"Gemini 요약 실패: 모델이 생성한 결과가 비어있습니다. Finish Reason: {response.candidates[0].finish_reason}")
            raise RuntimeError(f"Gemini 요약 실패: {str(e)}")

    def summarize_with_local(self, text):
        """로컬 모델을 사용한 텍스트 요약"""
        try:
            max_length = 1024
            chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
            summaries = []
            for chunk in chunks:
                if len(chunk.strip()) > 50:
                    summary = self.local_summarizer(chunk, max_length=200, min_length=30, do_sample=False)
                    summaries.append(summary[0]['summary_text'])
            return " ".join(summaries)
        except Exception as e:
            raise RuntimeError(f"로컬 모델 요약 실패: {str(e)}")

    def summarize_with_ollama(self, text, model_name="llama2"):
        """Ollama 로컬 LLM을 사용한 텍스트 요약"""
        try:
            url = "http://localhost:11434/api/generate"
            data = {"model": model_name, "prompt": f"다음 텍스트를 간결하고 명확하게 요약해주세요:\n\n{text}", "stream": False}
            response = requests.post(url, json=data)
            response.raise_for_status()
            return response.json()['response']
        except requests.exceptions.ConnectionError:
            raise RuntimeError("Ollama 서버에 연결할 수 없습니다. Ollama가 실행 중인지 확인해주세요.")
        except Exception as e:
            raise RuntimeError(f"Ollama 요약 실패: {str(e)}")

    def summarize(self, text, summary_type="general", context=None):
        """설정된 방법에 따라 텍스트를 요약"""
        if not text or len(text.strip()) < 50:
            return "요약하기에는 텍스트가 너무 짧습니다."
        if self.method == 'none':
            return "요약 방법이 'none'으로 설정되어 요약을 건너뜁니다."

        self._initialize_client()
        print(f"텍스트 요약 시작 ({self.method})...")
        
        if self.method == 'openai_api':
            return self.summarize_with_openai(text, summary_type, context)
        elif self.method == 'gemini_api':
            return self.summarize_with_gemini(text, summary_type, context)
        elif self.method == 'local_model':
            return self.summarize_with_local(text)
        elif self.method == 'ollama':
            return self.summarize_with_ollama(text)
        else:
            raise ValueError(f"지원하지 않는 요약 방법: {self.method}")

    def create_bullet_points(self, text, context=None):
        """텍스트를 불릿 포인트 형식으로 요약"""
        self._initialize_client()
        instruction = "다음 텍스트의 주요 내용을 한국어 불릿 포인트(•) 형식으로 정리해주세요. 각 포인트는 간결하고 명확하게 작성해주세요."
        
        if self.method == 'openai_api':
            system_content = f"{instruction}"
            if context:
                system_content += f"\n\n[사전 정보]\n{context}"
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "system", "content": system_content}, {"role": "user", "content": text}],
                    max_tokens=800, temperature=0.3)
                return response.choices[0].message.content
            except Exception as e:
                raise RuntimeError(f"OpenAI 불릿 포인트 생성 실패: {str(e)}")
        
        elif self.method == 'gemini_api':
            prompt_parts = [f"[요약 지시]\n{instruction}"]
            if context:
                prompt_parts.append(f"\n[사전 정보]\n{context}")
            prompt_parts.append(f"\n\n--- 텍스트 시작 ---\n{text}\n--- 텍스트 끝 ---")
            prompt = "\n".join(prompt_parts)
            try:
                safety_settings = [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                ]
                response = self.gemini_model.generate_content(prompt, safety_settings=safety_settings)
                return response.text
            except Exception as e:
                if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                    raise RuntimeError(f"Gemini 불릿 포인트 생성 실패: 프롬프트가 안전 설정에 의해 차단되었습니다. Reason: {response.prompt_feedback.block_reason}")
                if not response.parts:
                    raise RuntimeError(f"Gemini 불릿 포인트 생성 실패: 모델이 생성한 결과가 비어있습니다. Finish Reason: {response.candidates[0].finish_reason}")
                raise RuntimeError(f"Gemini 불릿 포인트 생성 실패: {str(e)}")
        else:
            return self.summarize(text, context=context)

    def save_summary(self, summary, output_file):
        """요약을 파일로 저장"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(summary)
            print(f"요약 저장 완료: {output_file}")
        except Exception as e:
            raise RuntimeError(f"요약 저장 실패: {str(e)}")

    def get_available_methods(self):
        """사용 가능한 요약 방법 반환"""
        methods = ['local_model']
        if config.OPENAI_API_KEY:
            methods.append('openai_api')
        if config.GOOGLE_API_KEY:
            methods.append('gemini_api')
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                methods.append('ollama')
        except:
            pass
        return methods