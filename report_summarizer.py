"""Consulting-session summary reports generated from transcripts.

Backend pattern mirrors translator.py: OpenAI API (default) or a local
LMStudio server. Long transcripts are summarized map-reduce style: chunk →
partial summaries → final report.
"""

import os
import shutil
import subprocess
import tempfile

CHUNK_CHARS = 9000  # local-model default; larger models take far larger chunks

CODEX_TIMEOUT_SECONDS = 1200


def default_chunk_chars(backend):
    """OpenAI/Codex fit an hour-long transcript in one call; local LMStudio
    models need small chunks. Override with STT_SUMMARY_CHUNK_CHARS."""
    override = os.getenv("STT_SUMMARY_CHUNK_CHARS")
    if override:
        try:
            return max(1000, int(override))
        except ValueError:
            pass
    return 120000 if backend in ("openai", "codex") else CHUNK_CHARS

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


def _resolve_codex_bin():
    override = os.getenv("STT_CODEX_BIN")
    if override:
        return override
    found = shutil.which("codex")
    if found:
        return found
    fallback = os.path.expanduser("~/.local/bin/codex")
    return fallback if os.path.exists(fallback) else None


def _sync_codex_auth(src, dst):
    """Mirror the main Codex login into the dedicated home.

    Prefers a symlink so token refreshes propagate automatically. Windows
    accounts without Developer Mode/admin can't create symlinks (WinError
    1314), so fall back to a copy, refreshing it whenever the source is newer.
    """
    if os.path.islink(dst):
        return  # symlink already tracks the source
    if os.path.exists(dst):
        try:
            if os.path.getmtime(dst) >= os.path.getmtime(src):
                return  # copy is up to date
        except OSError:
            pass
    if os.path.lexists(dst):
        os.remove(dst)
    try:
        os.symlink(src, dst)
    except (OSError, NotImplementedError, AttributeError):
        shutil.copy2(src, dst)


def _ensure_codex_home():
    """Dedicated CODEX_HOME so third-party Codex plugins can't inject notices
    into summaries. Shares the main login (symlink where possible, else copy)."""
    home = os.path.expanduser(os.getenv("STT_CODEX_HOME", "~/.codex-stt"))
    os.makedirs(home, exist_ok=True)
    auth = os.path.join(home, "auth.json")
    main_auth = os.path.expanduser("~/.codex/auth.json")
    if os.path.exists(main_auth):
        _sync_codex_auth(main_auth, auth)
    if not os.path.exists(auth):
        raise RuntimeError(
            "Codex 로그인이 필요합니다. 터미널에서 `codex login`을 실행한 뒤 다시 시도하세요."
        )
    return home


class ReportSummarizer:
    def __init__(self, backend="openai", model=None):
        backend = (backend or "openai").lower()
        if backend == "codex":
            self.codex_bin = _resolve_codex_bin()
            if not self.codex_bin:
                raise RuntimeError(
                    "Codex CLI를 찾을 수 없습니다. `npm install -g @openai/codex` 후 "
                    "`codex login`을 실행하세요."
                )
            self.codex_home = _ensure_codex_home()
            self.model = model or os.getenv("CODEX_SUMMARY_MODEL") or None
            self.client = None
        elif backend == "lmstudio":
            from openai import OpenAI
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
                    "또는 요약 엔진을 Codex 구독/LMStudio로 선택하세요."
                )
            from openai import OpenAI
            self.client = OpenAI(api_key=key, base_url=os.getenv("OPENAI_BASE_URL"))
            self.model = model or os.getenv("OPENAI_SUMMARY_MODEL", "gpt-4o-mini")
        self.backend = backend
        self.chunk_chars = default_chunk_chars(backend)

    def _codex_chat(self, system_prompt, user_prompt):
        prompt = f"{system_prompt}\n\n{user_prompt}"
        out_path = None
        try:
            fd, out_path = tempfile.mkstemp(prefix="codex_summary_", suffix=".md")
            os.close(fd)
            cmd = [
                self.codex_bin, "exec", "-",
                "--skip-git-repo-check",
                "-s", "read-only",
                "--output-last-message", out_path,
            ]
            if self.model:
                cmd += ["-m", self.model]
            env = dict(os.environ)
            env["CODEX_HOME"] = self.codex_home
            result = subprocess.run(
                cmd, input=prompt, capture_output=True, text=True,
                timeout=CODEX_TIMEOUT_SECONDS, env=env, cwd=self.codex_home,
            )
            content = ""
            if os.path.exists(out_path):
                with open(out_path, encoding="utf-8") as f:
                    content = f.read().strip()
            if result.returncode != 0 and not content:
                detail = (result.stderr or result.stdout or "").strip()[-300:]
                raise RuntimeError(f"Codex 실행 실패 (코드 {result.returncode}): {detail}")
            if not content:
                raise RuntimeError("Codex 요약 응답이 비어 있습니다.")
            return content
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Codex 요약이 {CODEX_TIMEOUT_SECONDS}초 안에 끝나지 않았습니다.")
        finally:
            if out_path and os.path.exists(out_path):
                os.remove(out_path)

    def _chat(self, system_prompt, user_prompt):
        if self.backend == "codex":
            return self._codex_chat(system_prompt, user_prompt)
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
        chunks = split_chunks(transcript, chunk_chars=self.chunk_chars)

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
