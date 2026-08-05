"""
Summarization 처리를 위한 전략 패턴 구현입니다.
각 요약 방법(OpenAI, Gemini 등)은 별도의 '전략' 클래스로 캡슐화됩니다.
"""
import openai
import requests
import google.generativeai as genai
from anthropic import Anthropic
import torch
import tiktoken
from transformers import pipeline
import config

# --- 1. 기본 전략 인터페이스 ---
class BaseSummarizeStrategy:
    """모든 요약 전략을 위한 기본 인터페이스"""
    def __init__(self):
        print(f"Initializing strategy: {self.__class__.__name__}")

    def summarize(self, text: str, summary_type: str, context: str, include_timestamps: bool) -> str:
        raise NotImplementedError("summarize() 메소드가 구현되지 않았습니다.")

    def create_bullet_points(self, text: str, context: str, include_timestamps: bool) -> str:
        # 기본적으로 일반 요약과 동일한 프롬프트를 사용하지만, 각 전략에서 재정의할 수 있음
        return self.summarize(text, "meeting", context, include_timestamps)

# --- 2. 구체적인 전략 클래스들 ---

class OpenAIAPIStrategy(BaseSummarizeStrategy):
    MODEL = "gpt-5.6-terra"
    CONTEXT_WINDOW_TOKENS = 1_050_000
    SAFETY_MARGIN_TOKENS = 2048
    SUMMARY_MAX_TOKENS = 2000
    BULLET_MAX_TOKENS = 800
    PARTIAL_MAX_TOKENS = 1200

    def __init__(self):
        super().__init__()
        if not config.OPENAI_API_KEY:
            raise ValueError("OpenAI API 키가 설정되지 않았습니다.")
        self.client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
        try:
            self.encoding = tiktoken.encoding_for_model(self.MODEL)
        except KeyError:
            self.encoding = tiktoken.get_encoding("o200k_base")

    def _get_prompt(self, summary_type, include_timestamps, is_bullet_points=False):
        if is_bullet_points:
            instruction = """[요약 지시]
당신은 주어진 텍스트의 핵심 내용을 빠짐없이 불릿 포인트(•)로 요약하는 AI 어시스턴트입니다. ... (이하 생략)"""
        else:
            prompts = {
                "general": "다음 텍스트를 내용을 빠뜨리지 말고, 명확하고 상세하게 요약해주세요.",
                "meeting": "다음 회의 내용을 논의된 모든 사항을 포함하여 자세히 요약해주세요. 주요 논의사항, 결정사항, 액션 아이템을 중심으로 정리해주세요.",
                # ... 다른 프롬프트들
            }
            instruction = prompts.get(summary_type, prompts["meeting"])
        
        if include_timestamps:
            instruction += "\n\n요약 내용에 원본 텍스트의 타임스탬프를 [시작시간] 형식으로 포함하여..."
        return instruction

    def _count_tokens(self, text):
        return len(self.encoding.encode(text))

    def _input_budget(self, system_content, completion_tokens):
        # 메시지 래핑 토큰과 모델별 계산 차이를 고려해 여유 공간을 둡니다.
        return max(
            1,
            self.CONTEXT_WINDOW_TOKENS
            - completion_tokens
            - self._count_tokens(system_content)
            - self.SAFETY_MARGIN_TOKENS,
        )

    def _split_text_by_tokens(self, text, max_tokens):
        """가능하면 줄 경계를 유지하면서 텍스트를 토큰 한도 이하로 나눕니다."""
        if self._count_tokens(text) <= max_tokens:
            return [text]

        chunks = []
        current_parts = []
        current_tokens = 0

        for part in text.splitlines(keepends=True):
            part_tokens = self.encoding.encode(part)

            if len(part_tokens) > max_tokens:
                if current_parts:
                    chunks.append("".join(current_parts))
                    current_parts = []
                    current_tokens = 0
                for start in range(0, len(part_tokens), max_tokens):
                    chunks.append(self.encoding.decode(part_tokens[start:start + max_tokens]))
                continue

            if current_parts and current_tokens + len(part_tokens) > max_tokens:
                chunks.append("".join(current_parts))
                current_parts = []
                current_tokens = 0

            current_parts.append(part)
            current_tokens += len(part_tokens)

        if current_parts:
            chunks.append("".join(current_parts))

        return [chunk for chunk in chunks if chunk.strip()]

    def _call_chat_completion(self, system_content, text, max_tokens):
        response = self.client.chat.completions.create(
            model=self.MODEL,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": text},
            ],
            max_completion_tokens=max_tokens,
            reasoning_effort="none",
        )
        return response.choices[0].message.content

    def _summarize_with_chunking(self, text, instruction, context, max_tokens):
        system_content = f"{instruction}\n\n[사전 정보]\n{context}" if context else instruction
        direct_budget = self._input_budget(system_content, max_tokens)

        if self._count_tokens(text) <= direct_budget:
            return self._call_chat_completion(system_content, text, max_tokens)

        partial_system = (
            f"{system_content}\n\n"
            "[긴 텍스트 부분 요약]\n"
            "이 입력은 전체 녹취록의 일부입니다. 고유명사, 수치, 결정사항, 액션 아이템과 "
            "타임스탬프를 보존해 이 부분의 핵심 내용을 빠짐없이 요약하세요."
        )
        partial_max_tokens = min(self.PARTIAL_MAX_TOKENS, max_tokens)
        chunk_budget = self._input_budget(partial_system, partial_max_tokens)
        chunks = self._split_text_by_tokens(text, chunk_budget)
        print(f"긴 텍스트를 {len(chunks)}개 청크로 나누어 OpenAI 요약을 진행합니다.")

        partial_summaries = []
        for index, chunk in enumerate(chunks, start=1):
            chunk_text = f"[부분 {index}/{len(chunks)}]\n{chunk}"
            partial_summaries.append(
                self._call_chat_completion(partial_system, chunk_text, partial_max_tokens)
            )

        reduce_system = (
            f"{system_content}\n\n"
            "[최종 통합]\n"
            "다음 부분 요약들을 하나의 일관된 최종 요약으로 통합하세요. 중복은 제거하되 "
            "고유명사, 수치, 결정사항, 액션 아이템과 타임스탬프는 누락하지 마세요."
        )
        reduce_budget = self._input_budget(reduce_system, max_tokens)
        combined = "\n\n--- 부분 요약 ---\n\n".join(partial_summaries)

        # 매우 긴 녹취는 부분 요약 결과도 한도를 넘을 수 있으므로 반복 축약합니다.
        while self._count_tokens(combined) > reduce_budget:
            groups = self._split_text_by_tokens(combined, reduce_budget)
            combined = "\n\n--- 중간 통합 요약 ---\n\n".join(
                self._call_chat_completion(reduce_system, group, partial_max_tokens)
                for group in groups
            )

        return self._call_chat_completion(reduce_system, combined, max_tokens)

    def summarize(self, text, summary_type, context, include_timestamps):
        instruction = self._get_prompt(summary_type, include_timestamps)
        try:
            return self._summarize_with_chunking(
                text, instruction, context, self.SUMMARY_MAX_TOKENS
            )
        except Exception as e:
            raise RuntimeError(f"OpenAI 요약 실패: {e}")

    def create_bullet_points(self, text, context, include_timestamps):
        instruction = self._get_prompt("meeting", include_timestamps, is_bullet_points=True)
        try:
            return self._summarize_with_chunking(
                text, instruction, context, self.BULLET_MAX_TOKENS
            )
        except Exception as e:
            raise RuntimeError(f"OpenAI 불릿 포인트 생성 실패: {e}")

class GeminiAPIStrategy(BaseSummarizeStrategy):
    def __init__(self):
        super().__init__()
        if not config.GOOGLE_API_KEY:
            raise ValueError("Google API 키가 설정되지 않았습니다.")
        genai.configure(api_key=config.GOOGLE_API_KEY)
        self.map_model = genai.GenerativeModel(config.GEMINI_MODEL_FOR_SUMMARY)
        self.reduce_model = genai.GenerativeModel(config.GEMINI_MODEL_FOR_FINAL_SUMMARY)
        self.safety_settings = [
            {"category": c, "threshold": "BLOCK_NONE"} for c in 
            ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", 
             "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]
        ]

    def summarize(self, text, summary_type, context, include_timestamps):
        CHAR_LIMIT = 800000
        if len(text) < CHAR_LIMIT:
            return self._call_gemini_api(self.reduce_model, text, summary_type, context, include_timestamps=include_timestamps)
        else:
            return self._summarize_long_text(text, summary_type, context, CHAR_LIMIT, include_timestamps)

    def create_bullet_points(self, text, context, include_timestamps):
        instruction = "... (불릿 포인트용 프롬프트) ..."
        # ... Gemini 불릿 포인트 로직 구현 ...
        return self._call_gemini_api(self.reduce_model, text, "meeting", context, include_timestamps=include_timestamps, is_bullet_points=True)

    def _summarize_long_text(self, text, summary_type, context, chunk_size, include_timestamps):
        text_chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        summaries = [self._call_gemini_api(self.map_model, chunk, summary_type, context, is_chunk=True, include_timestamps=include_timestamps) for chunk in text_chunks]
        combined = "\n\n---\n\n".join(summaries)
        return self._call_gemini_api(self.reduce_model, combined, summary_type, context, is_final=True, include_timestamps=include_timestamps)

    def _get_prompt(self, summary_type, is_chunk=False, is_final=False, is_bullet_points=False, include_timestamps=False):
        """요약 유형에 따라 적절한 프롬프트를 반환합니다."""
        if is_bullet_points:
            base_instruction = "당신은 주어진 텍스트의 핵심 내용을 빠짐없이 불릿 포인트(•)로 요약하는 AI 어시스턴트입니다. 각 항목은 명확하고 상세하게 작성해야 합니다."
        else:
            prompts = {
                "general": "다음 텍스트를 내용을 빠뜨리지 말고, 명확하고 상세하게 요약해주세요.",
                "meeting": "다음 회의 내용을 논의된 모든 사항을 포함하여 자세히 요약해주세요. 주요 논의사항, 결정사항, 액션 아이템을 중심으로 정리해주세요."
            }
            base_instruction = prompts.get(summary_type, prompts["meeting"])

        if is_chunk:
            instruction = f"{base_instruction} 이 텍스트는 긴 내용의 일부입니다. 전체적인 맥락을 고려하여 이 부분의 핵심 내용을 상세히 요약해주세요."
        elif is_final:
            instruction = f"다음은 여러 텍스트 조각들의 요약본입니다. 이 요약본들을 종합하여 전체 내용에 대한 최종적이고 완전한 요약본을 생성해주세요. {base_instruction}"
        else:
            instruction = base_instruction

        if include_timestamps:
            instruction += "\n\n요약 내용에 원본 텍스트의 타임스탬프를 [시작시간] 형식으로 포함하여 각 내용이 언급된 시점을 명확히 표시해주세요."

        # 가장 중요한 부분: 출력 언어를 한국어로 명시
        instruction += "\n\n결과는 반드시 한국어로 작성해주세요."
        return instruction

    def _call_gemini_api(self, model, text, summary_type, context, is_chunk=False, is_final=False, include_timestamps=False, is_bullet_points=False):
        instruction = self._get_prompt(summary_type, is_chunk, is_final, is_bullet_points, include_timestamps)
        
        context_str = f"\n\n[사전 정보]\n{context}" if context else ""
        prompt = f"{instruction}{context_str}\n\n[원본 텍스트]\n{text}"

        try:
            response = model.generate_content(prompt, safety_settings=self.safety_settings)
            return response.text
        except Exception as e:
            raise RuntimeError(f"Gemini API 호출 실패: {e}")

class ClaudeAPIStrategy(BaseSummarizeStrategy):
    def __init__(self):
        super().__init__()
        if not config.ANTHROPIC_API_KEY:
            raise ValueError("Anthropic API 키가 설정되지 않았습니다.")
        self.client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
        self.model = config.CLAUDE_MODEL

    def summarize(self, text, summary_type, context, include_timestamps):
        CHAR_LIMIT = 200000
        if len(text) < CHAR_LIMIT:
            return self._call_claude_api(text, summary_type, context, include_timestamps=include_timestamps)
        else:
            return self._summarize_long_text(text, summary_type, context, CHAR_LIMIT, include_timestamps)

    def create_bullet_points(self, text, context, include_timestamps):
        return self._call_claude_api(text, "meeting", context, include_timestamps=include_timestamps, is_bullet_points=True)

    def _summarize_long_text(self, text, summary_type, context, chunk_size, include_timestamps):
        text_chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        summaries = [self._call_claude_api(chunk, summary_type, context, is_chunk=True, include_timestamps=include_timestamps) for chunk in text_chunks]
        combined = "\n\n---\n\n".join(summaries)
        return self._call_claude_api(combined, summary_type, context, is_final=True, include_timestamps=include_timestamps)

    def _get_prompt(self, summary_type, is_chunk=False, is_final=False, is_bullet_points=False, include_timestamps=False):
        if is_bullet_points:
            base_instruction = "당신은 주어진 텍스트의 핵심 내용을 빠짐없이 불릿 포인트(•)로 요약하는 AI 어시스턴트입니다. 각 항목은 명확하고 상세하게 작성해야 합니다."
        else:
            prompts = {
                "general": "다음 텍스트를 내용을 빠뜨리지 말고, 명확하고 상세하게 요약해주세요.",
                "meeting": "다음 회의 내용을 논의된 모든 사항을 포함하여 자세히 요약해주세요. 주요 논의사항, 결정사항, 액션 아이템을 중심으로 정리해주세요."
            }
            base_instruction = prompts.get(summary_type, prompts["meeting"])

        if is_chunk:
            instruction = f"{base_instruction} 이 텍스트는 긴 내용의 일부입니다. 전체적인 맥락을 고려하여 이 부분의 핵심 내용을 상세히 요약해주세요."
        elif is_final:
            instruction = f"다음은 여러 텍스트 조각들의 요약본입니다. 이 요약본들을 종합하여 전체 내용에 대한 최종적이고 완전한 요약본을 생성해주세요. {base_instruction}"
        else:
            instruction = base_instruction

        if include_timestamps:
            instruction += "\n\n요약 내용에 원본 텍스트의 타임스탬프를 [시작시간] 형식으로 포함하여 각 내용이 언급된 시점을 명확히 표시해주세요."

        instruction += "\n\n결과는 반드시 한국어로 작성해주세요."
        return instruction

    def _call_claude_api(self, text, summary_type, context, is_chunk=False, is_final=False, include_timestamps=False, is_bullet_points=False):
        instruction = self._get_prompt(summary_type, is_chunk, is_final, is_bullet_points, include_timestamps)
        context_str = f"\n\n[사전 정보]\n{context}" if context else ""
        user_content = f"{context_str}\n\n[원본 텍스트]\n{text}"

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=8192,
                system=instruction,
                messages=[{"role": "user", "content": user_content}]
            )
            return response.content[0].text
        except Exception as e:
            raise RuntimeError(f"Claude API 호출 실패: {e}")

class LocalModelStrategy(BaseSummarizeStrategy):
    def __init__(self):
        super().__init__()
        device = -1 # CPU
        self.summarizer = pipeline("summarization", model="eenzeenee/t5-small-korean-summarization", device=device)

    def summarize(self, text, summary_type, context, include_timestamps):
        max_chunk_length = 1500
        chunks = [text[i:i+max_chunk_length] for i in range(0, len(text), max_chunk_length)]
        summaries = []
        for chunk in chunks:
            if len(chunk.strip()) > 50:
                summary = self.summarizer(chunk, max_length=500, min_length=50, do_sample=False)
                summaries.append(summary[0]['summary_text'])
        return " ".join(summaries)

class OllamaStrategy(BaseSummarizeStrategy):
    def __init__(self, model_name="llama2"):
        super().__init__()
        self.model_name = model_name
        self.url = "http://localhost:11434/api/generate"

    def summarize(self, text, summary_type, context, include_timestamps):
        prompt = f"다음 내용을 빠뜨리지 말고 최대한 상세하고 명확하게 한국어로 요약해주세요:\n\n{text}"
        data = {"model": self.model_name, "prompt": prompt, "stream": False}
        try:
            response = requests.post(self.url, json=data)
            response.raise_for_status()
            return response.json()['response']
        except requests.exceptions.ConnectionError:
            raise RuntimeError("Ollama 서버에 연결할 수 없습니다.")
        except Exception as e:
            raise RuntimeError(f"Ollama 요약 실패: {e}")

# --- 3. 전략 팩토리 ---
def get_summarize_strategy(method: str) -> BaseSummarizeStrategy:
    if method == 'openai_api':
        return OpenAIAPIStrategy()
    elif method == 'gemini_api':
        return GeminiAPIStrategy()
    elif method == 'claude_api':
        return ClaudeAPIStrategy()
    elif method == 'local_model':
        return LocalModelStrategy()
    elif method == 'ollama':
        return OllamaStrategy()
    else:
        raise ValueError(f"지원하지 않는 요약 방법: {method}")

def get_available_summarize_methods() -> list:
    methods = ['local_model']
    if config.OPENAI_API_KEY: methods.append('openai_api')
    if config.GOOGLE_API_KEY: methods.append('gemini_api')
    if config.ANTHROPIC_API_KEY: methods.append('claude_api')
    try:
        if requests.get("http://localhost:11434/api/tags", timeout=2).status_code == 200:
            methods.append('ollama')
    except:
        pass
    return methods
