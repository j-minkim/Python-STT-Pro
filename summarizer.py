from openai import OpenAI
import os

class LMStudioSummarizer:
    def __init__(self, base_url="http://localhost:1234/v1", model_name="local-model"):
        # We don't need a real API key for LMStudio
        self.client = OpenAI(base_url=base_url, api_key="not-needed")
        self.model_name = model_name

    def summarize_timeline(self, diarized_transcript):
        """
        Takes a list of diarized segments and asks LMStudio to create a timeline summary.
        """
        print("Connecting to LMStudio API to generate summary...")
        
        # Format the transcript into a readable text block
        transcript_text = ""
        for seg in diarized_transcript:
            start_m, start_s = divmod(seg['start'], 60)
            end_m, end_s = divmod(seg['end'], 60)
            timestamp = f"[{int(start_m):02d}:{int(start_s):02d} - {int(end_m):02d}:{int(end_s):02d}]"
            transcript_text += f"{timestamp} {seg['speaker']}: {seg['text']}\n"
            
        system_prompt = (
            "당신은 입시 컨설팅 전문 AI 어시스턴트입니다. "
            "주어진 대화 기록(학생과 컨설턴트/강사 간의 대화)을 분석하여, "
            "타임라인 흐름에 따라 가장 중요한 핵심(주제, 성적 분석, 진로 목표 등)을 "
            "마크다운(Markdown) 포맷으로 가독성 좋게 정리해 주세요."
        )

        user_prompt = f"다음은 대화 전문입니다:\n\n{transcript_text}\n\n이 내용을 타임라인 기반으로 요약해 주세요."

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3 # Low temperature for factual summarization
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Failed to connect to LMStudio: {e}")
            print("Please ensure LMStudio 'Local Server' is running on port 1234.")
            return f"Error: {e}"
