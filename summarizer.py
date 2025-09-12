import openai
import requests
from transformers import pipeline
import config

class TextSummarizer:
    def __init__(self):
        self.method = config.SUMMARIZE_METHOD
        self.openai_client = None
        self.local_summarizer = None
        
        if self.method == 'openai_api':
            if not config.OPENAI_API_KEY:
                raise ValueError("OpenAI API 키가 설정되지 않았습니다.")
            self.openai_client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
        elif self.method == 'local_model':
            print("로컬 요약 모델을 로딩 중...")
            self.local_summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=-1  # CPU 사용
            )
    
    def summarize_with_openai(self, text, summary_type="general"):
        """OpenAI API를 사용한 텍스트 요약"""
        try:
            system_prompts = {
                "general": "다음 텍스트를 간결하고 명확하게 요약해주세요. 주요 내용과 핵심 포인트를 중심으로 정리해주세요.",
                "meeting": "다음 회의 내용을 요약해주세요. 주요 논의사항, 결정사항, 액션 아이템을 중심으로 정리해주세요.",
                "lecture": "다음 강의 내용을 요약해주세요. 핵심 개념, 중요한 설명, 예시를 중심으로 정리해주세요.",
                "interview": "다음 인터뷰 내용을 요약해주세요. 주요 질문과 답변, 핵심 내용을 중심으로 정리해주세요."
            }
            
            system_prompt = system_prompts.get(summary_type, system_prompts["general"])
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            raise RuntimeError(f"OpenAI 요약 실패: {str(e)}")
    
    def summarize_with_local(self, text):
        """로컬 모델을 사용한 텍스트 요약"""
        try:
            # 긴 텍스트를 청크로 나누기 (BART 모델 입력 제한)
            max_length = 1024
            chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
            
            summaries = []
            for chunk in chunks:
                if len(chunk.strip()) > 50:  # 너무 짧은 청크는 건너뛰기
                    summary = self.local_summarizer(
                        chunk,
                        max_length=150,
                        min_length=30,
                        do_sample=False
                    )
                    summaries.append(summary[0]['summary_text'])
            
            return " ".join(summaries)
            
        except Exception as e:
            raise RuntimeError(f"로컬 모델 요약 실패: {str(e)}")
    
    def summarize_with_ollama(self, text, model_name="llama2"):
        """Ollama 로컬 LLM을 사용한 텍스트 요약"""
        try:
            url = "http://localhost:11434/api/generate"
            data = {
                "model": model_name,
                "prompt": f"다음 텍스트를 간결하고 명확하게 요약해주세요:\\n\\n{text}",
                "stream": False
            }
            
            response = requests.post(url, json=data)
            response.raise_for_status()
            
            return response.json()['response']
            
        except requests.exceptions.ConnectionError:
            raise RuntimeError("Ollama 서버에 연결할 수 없습니다. Ollama가 실행 중인지 확인해주세요.")
        except Exception as e:
            raise RuntimeError(f"Ollama 요약 실패: {str(e)}")
    
    def summarize(self, text, summary_type="general"):
        """설정된 방법에 따라 텍스트를 요약"""
        if not text or len(text.strip()) < 50:
            return "요약하기에는 텍스트가 너무 짧습니다."
        
        print(f"텍스트 요약 시작 ({self.method})...")
        
        if self.method == 'openai_api':
            return self.summarize_with_openai(text, summary_type)
        elif self.method == 'local_model':
            return self.summarize_with_local(text)
        elif self.method == 'ollama':
            return self.summarize_with_ollama(text)
        else:
            raise ValueError(f"지원하지 않는 요약 방법: {self.method}")
    
    def create_bullet_points(self, text):
        """텍스트를 불릿 포인트 형식으로 요약"""
        if self.method == 'openai_api':
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content": "다음 텍스트의 주요 내용을 불릿 포인트(•) 형식으로 정리해주세요. 각 포인트는 간결하고 명확하게 작성해주세요."
                        },
                        {"role": "user", "content": text}
                    ],
                    max_tokens=800,
                    temperature=0.3
                )
                
                return response.choices[0].message.content
                
            except Exception as e:
                raise RuntimeError(f"불릿 포인트 생성 실패: {str(e)}")
        else:
            # 로컬 모델의 경우 일반 요약 반환
            return self.summarize(text)
    
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
        
        # Ollama 서버 확인
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                methods.append('ollama')
        except:
            pass
        
        return methods