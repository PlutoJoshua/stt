import openai
import requests
import google.generativeai as genai
import torch
from transformers import pipeline
import config

class TextSummarizer:
    def __init__(self):
        self.method = config.SUMMARIZE_METHOD
        self.openai_client = None
        self.local_summarizer = None
        self.gemini_map_model = None
        self.gemini_reduce_model = None
        
    def _initialize_client(self):
        """필요한 API 클라이언트를 초기화합니다."""
        if self.method == 'openai_api' and not self.openai_client:
            if not config.OPENAI_API_KEY:
                raise ValueError("OpenAI API 키가 설정되지 않았습니다.")
            self.openai_client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
        
        elif self.method == 'gemini_api' and (not self.gemini_map_model or not self.gemini_reduce_model):
            if not config.GOOGLE_API_KEY:
                raise ValueError("Google API 키가 설정되지 않았습니다.")
            genai.configure(api_key=config.GOOGLE_API_KEY)
            print(f"Gemini Map Model: {config.GEMINI_MODEL_FOR_SUMMARY}")
            print(f"Gemini Reduce Model: {config.GEMINI_MODEL_FOR_FINAL_SUMMARY}")
            self.gemini_map_model = genai.GenerativeModel(config.GEMINI_MODEL_FOR_SUMMARY)
            self.gemini_reduce_model = genai.GenerativeModel(config.GEMINI_MODEL_FOR_FINAL_SUMMARY)

        elif self.method == 'local_model' and not self.local_summarizer:
            print("로컬 요약 모델을 로딩 중...")
            # Forcing CPU for stability to match the main STT service
            device = -1 # -1 for CPU
            
            self.local_summarizer = pipeline(
                "summarization",
                model="eenzeenee/t5-small-korean-summarization",
                device=device
            )
            print(f"로컬 요약 모델이 CPU에 로드되었습니다.")

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
        """Gemini API를 사용한 텍스트 요약. 긴 텍스트는 분할 처리합니다."""
        
        # Gemini 1.5 Flash의 컨텍스트 창을 고려하여 보수적인 문자 수 제한 설정
        # Flash 모델의 1M 토큰은 약 200~400만 문자에 해당하나, 프롬프트/응답 공간을 고려하여 800,000자로 설정
        CHAR_LIMIT = 800000

        if len(text) < CHAR_LIMIT:
            # 텍스트가 짧으면 Reduce 모델(고성능)로 직접 요약
            print("텍스트가 짧아 직접 요약을 진행합니다.")
            return self._call_gemini_api(self.gemini_reduce_model, text, summary_type, context)
        else:
            # 텍스트가 길면 분할하여 요약 (Map-Reduce)
            print(f"텍스트가 너무 깁니다({len(text)}자). 분할하여 요약을 진행합니다.")
            return self._summarize_long_text_gemini(text, summary_type, context, CHAR_LIMIT)

    def _summarize_long_text_gemini(self, text, summary_type, context, chunk_size):
        """긴 텍스트를 Map-Reduce 방식으로 요약합니다."""
        
        # 1. 텍스트 분할 (Map)
        text_chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        print(f"{len(text_chunks)}개의 조각으로 나누어 요약을 시작합니다.")

        # 2. 각 조각 요약
        intermediate_summaries = []
        for i, chunk in enumerate(text_chunks):
            print(f"[{i+1}/{len(text_chunks)}]번째 조각 요약 중...")
            # Map 단계에서는 Map용 모델 사용
            summary = self._call_gemini_api(self.gemini_map_model, chunk, summary_type, context, is_chunk=True)
            intermediate_summaries.append(summary)

        # 3. 요약본들 취합 및 최종 요약 (Reduce)
        print("개별 요약을 취합하여 최종 요약을 생성합니다.")
        combined_summaries = "\n\n---\n\n".join(intermediate_summaries)
        
        # Reduce 단계에서는 Reduce용 모델 사용
        final_summary = self._call_gemini_api(self.gemini_reduce_model, combined_summaries, summary_type, context, is_final=True)
        
        return final_summary

    def _call_gemini_api(self, model, text, summary_type, context, is_chunk=False, is_final=False):
        """Gemini API를 직접 호출하는 내부 함수"""
        response = None # Initialize response
        try:
            # 단계별 프롬프트 선택
            if is_chunk:
                instruction = "다음은 긴 문서의 일부입니다. 이 부분을 상세하고 구조적으로 요약해주세요. 나중에 다른 부분들과 합쳐질 것을 감안하여, 주요 내용과 맥락을 최대한 보존해주세요."
            elif is_final:
                instruction = "다음은 긴 문서의 각 부분을 요약한 내용들입니다. 이 요약본들을 종합하여, 전체 문서에 대한 하나의 최종적이고 완성도 높은 요약문을 만들어주세요. 전체적인 흐름과 핵심 내용을 명확하게 정리해야 합니다."
            else: # 일반 요약
                system_prompts = {
                    "general": '''[요약 지시]
당신은 주어진 텍스트를 전문적으로 요약하는 AI 어시스턴트입니다.
텍스트의 핵심 내용을 매우 상세하고, 길고, 구조적으로 요약해주세요.
주요 내용을 절대 빠뜨리지 말고, 원본의 맥락과 뉘앙스를 완벽하게 유지하면서 최대한 길고 상세하게 작성해주세요. 짧은 요약은 허용되지 않습니다.''',
                    "meeting": '''[요약 지시]
당신은 회의록을 전문적으로 요약하는 AI 어시스턴트입니다.
아래에 제공되는 텍스트는 화자 분리(diarization)가 적용된 회의록일 수 있습니다.
각 화자의 발언 내용을 바탕으로, 회의의 핵심 내용을 매우 상세하고, 길고, 구조적으로 요약해주세요.

다음 항목들을 반드시 포함해주세요:
- **주요 논의 안건:** 어떤 주제들이 논의되었나요?
- **핵심 발언:** 각 주제에 대한 주요 의견들은 무엇이었나요?
- **결정 사항:** 어떤 결론이 내려졌나요?
- **향후 계획 (Action Items):** 앞으로 누가 무엇을 해야 하나요?

전체 내용을 빠짐없이 검토하고, 원본의 맥락을 완벽하게 유지하면서 최대한 길고 상세하게 작성해주세요. 짧은 요약은 허용되지 않습니다.''',
                    "lecture": '''[요약 지시]
당신은 강의 내용을 전문적으로 요약하는 AI 어시스턴트입니다.
아래 텍스트는 강의 내용입니다.
강의의 핵심 개념, 중요한 설명, 예시, 그리고 결론을 중심으로 매우 상세하고, 길고, 구조적으로 요약해주세요.
수강생이 강의를 듣지 않아도 내용을 완벽히 이해할 수 있도록, 사소한 내용도 빠짐없이 최대한 길고 상세하게 작성해주세요. 짧은 요약은 허용되지 않습니다.''',
                    "interview": '''[요약 지시]
당신은 인터뷰 내용을 전문적으로 요약하는 AI 어시스턴트입니다.
아래 텍스트는 인터뷰 내용입니다.
주요 질문과 답변, 대화의 핵심 주제, 그리고 인터뷰에서 드러난 중요한 정보나 견해를 중심으로 매우 상세하고, 길고, 구조적으로 요약해주세요.
인터뷰의 전체적인 흐름과 맥락이 잘 드러나도록, 사소한 내용도 빠짐없이 최대한 길고 상세하게 작성해주세요. 짧은 요약은 허용되지 않습니다.'''
                }
                instruction = system_prompts.get(summary_type, system_prompts["meeting"])

            prompt_parts = [f"{instruction}"]
            if context:
                prompt_parts.append(f"\n[사전 정보]\n{context}")
            
            prompt_parts.append(f"\n\n--- 텍스트 시작 ---\n{text}\n--- 텍스트 끝 ---")
            
            prompt = "\n".join(prompt_parts)
            
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]

            response = model.generate_content(
                prompt, 
                safety_settings=safety_settings
            )
            return response.text
        except Exception as e:
            if response and hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                 raise RuntimeError(f"Gemini 요약 실패: 프롬프트가 안전 설정에 의해 차단되었습니다. Reason: {response.prompt_feedback.block_reason}")
            if response and not response.parts:
                 raise RuntimeError(f"Gemini 요약 실패: 모델이 생성한 결과가 비어있습니다. Finish Reason: {response.candidates[0].finish_reason}")
            raise RuntimeError(f"Gemini 요약 실패: {str(e)}")

    def summarize_with_local(self, text):
        """로컬 모델을 사용한 텍스트 요약"""
        try:
            # T5 모델의 최대 입력 길이를 고려하여 보수적으로 1000자로 설정
            max_chunk_length = 1000
            chunks = [text[i:i+max_chunk_length] for i in range(0, len(text), max_chunk_length)]
            summaries = []
            for chunk in chunks:
                if len(chunk.strip()) > 50:
                    # 요약 최대 길이를 400으로 늘림
                    summary = self.local_summarizer(chunk, max_length=1500, min_length=50, do_sample=False)
                    summaries.append(summary[0]['summary_text'])
            return " ".join(summaries)
        except Exception as e:
            raise RuntimeError(f"로컬 모델 요약 실패: {str(e)}")

    def summarize_with_ollama(self, text, model_name="llama2"):
        """Ollama 로컬 LLM을 사용한 텍스트 요약"""
        try:
            url = "http://localhost:11434/api/generate"
            data = {"model": model_name, "prompt": f"다음 내용을 최대한 상세하고 명확하게 요약해주세요:\n\n{text}", "stream": False}
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
        instruction = """[요약 지시]
당신은 주어진 텍스트의 핵심 내용을 불릿 포인트(•)로 요약하는 AI 어시스턴트입니다.
아래 텍스트는 화자 분리(diarization)가 적용된 회의록일 수 있습니다.
각 화자의 발언 내용을 바탕으로, 회의의 핵심 내용을 상세하고, 길고, 구조적으로 요약해주세요.

다음 항목들을 반드시 포함하여 불릿 포인트(•)로 정리해주세요:
- **주요 논의 안건:**
- **핵심 발언:**
- **결정 사항:**
- **향후 계획 (Action Items):**

전체 내용을 빠짐없이 검토하고, 원본의 맥락을 완벽하게 유지하면서 최대한 길고 상세하게 작성해주세요."""
        response = None # Initialize response
        
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
            prompt_parts = [f"{instruction}"]
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
                response = self.gemini_reduce_model.generate_content(prompt, safety_settings=safety_settings)
                return response.text
            except Exception as e:
                if response and hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                    raise RuntimeError(f"Gemini 불릿 포인트 생성 실패: 프롬프트가 안전 설정에 의해 차단되었습니다. Reason: {response.prompt_feedback.block_reason}")
                if response and not response.parts:
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

    @staticmethod
    def get_available_methods():
        """사용 가능한 요약 방법 반환"""
        methods = ['local_model']
        if config.OPENAI_API_KEY:
            methods.append('openai_api')
        if config.GOOGLE_API_KEY:
            methods.append('gemini_api')
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                methods.append('ollama')
        except:
            pass
        return methods