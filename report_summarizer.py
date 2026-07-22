"""Consulting-session summary reports generated from transcripts.

Backend pattern mirrors translator.py: OpenAI API (default) or a local
LMStudio server. Long transcripts are summarized map-reduce style: chunk →
partial summaries → final report.
"""

import os

from openai import OpenAI

CHUNK_CHARS = 9000

REPORT_SYSTEM_PROMPT = (
    "당신은 입시 컨설팅 상담 기록을 정리하는 전문 어시스턴트입니다. "
    "학생과 컨설턴트의 대화 전사를 읽고, 상담에 참여하지 않은 사람도 "
    "5분 안에 파악할 수 있는 한국어 마크다운 리포트를 작성합니다. "
    "전사에 없는 내용을 지어내지 마세요. 다음 구조를 따르세요:\n"
    "## 상담 개요\n(한두 문장)\n"
    "## 학생 현황\n(성적·활동·관심사 등 파악된 사실)\n"
    "## 주요 논의 주제\n(불릿, 시각 표기 [mm:ss] 포함)\n"
    "## 컨설턴트 조언·결정사항\n(불릿)\n"
    "## 후속 액션 아이템\n(누가·무엇을·언제까지, 파악된 것만)"
)

CHUNK_SYSTEM_PROMPT = (
    "당신은 입시 컨설팅 상담 전사의 일부 구간을 요약하는 어시스턴트입니다. "
    "이 구간에서 논의된 주제, 언급된 사실(성적·활동·목표), 조언과 결정, "
    "후속 과제를 시각 표기 [mm:ss]와 함께 한국어 불릿으로 빠짐없이 정리하세요. "
    "지어내지 마세요."
)


def _format_timestamp(seconds):
    minutes, secs = divmod(int(seconds or 0), 60)
    return f"[{minutes:02d}:{secs:02d}]"


def format_transcript(segments):
    return "\n".join(
        f"{_format_timestamp(seg.get('start'))} {seg.get('text', '').strip()}"
        for seg in segments or []
        if (seg.get('text') or '').strip()
    )


def split_chunks(text, chunk_chars=CHUNK_CHARS):
    """Split on line boundaries so timestamps stay intact."""
    if len(text) <= chunk_chars:
        return [text]
    chunks, current, size = [], [], 0
    for line in text.splitlines():
        if size + len(line) + 1 > chunk_chars and current:
            chunks.append("\n".join(current))
            current, size = [], 0
        current.append(line)
        size += len(line) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks


class ReportSummarizer:
    def __init__(self, backend="openai", model=None):
        backend = (backend or "openai").lower()
        if backend == "lmstudio":
            self.client = OpenAI(
                base_url=os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"),
                api_key="not-needed",
            )
            self.model = model or os.getenv("LMSTUDIO_MODEL", "local-model")
        else:
            backend = "openai"
            key = os.getenv("OPENAI_API_KEY")
            if not key:
                raise RuntimeError(
                    "AI 요약에는 OPENAI_API_KEY가 필요합니다 (.env에 추가), "
                    "또는 요약 엔진을 LMStudio로 선택하세요."
                )
            self.client = OpenAI(api_key=key, base_url=os.getenv("OPENAI_BASE_URL"))
            self.model = model or os.getenv("OPENAI_SUMMARY_MODEL", "gpt-4o-mini")
        self.backend = backend

    def _chat(self, system_prompt, user_prompt):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )
        content = (response.choices[0].message.content or "").strip()
        if not content:
            raise RuntimeError("요약 응답이 비어 있습니다.")
        return content

    def summarize_segments(self, segments, filename=""):
        transcript = format_transcript(segments)
        if not transcript.strip():
            raise RuntimeError("요약할 전사 내용이 없습니다.")

        title = f"# 상담 요약 — {filename}\n\n" if filename else ""
        chunks = split_chunks(transcript)

        if len(chunks) == 1:
            body = self._chat(
                REPORT_SYSTEM_PROMPT,
                f"다음은 상담 전사 전문입니다:\n\n{chunks[0]}\n\n리포트를 작성해 주세요.",
            )
            return title + body

        partials = []
        for i, chunk in enumerate(chunks, 1):
            partials.append(self._chat(
                CHUNK_SYSTEM_PROMPT,
                f"상담 전사 {len(chunks)}개 구간 중 {i}번째 구간입니다:\n\n{chunk}",
            ))
        combined = "\n\n".join(
            f"### 구간 {i} 요약\n{partial}" for i, partial in enumerate(partials, 1)
        )
        body = self._chat(
            REPORT_SYSTEM_PROMPT,
            "다음은 한 상담을 구간별로 나눠 요약한 내용입니다. "
            f"이를 종합해 최종 리포트를 작성해 주세요:\n\n{combined}",
        )
        return title + body
