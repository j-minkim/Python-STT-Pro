"""Subtitle translation backed by an OpenAI-compatible chat API.

The same client talks to either the OpenAI cloud or a local LMStudio server
(both expose the OpenAI chat-completions interface), mirroring the pattern in
summarizer.py. Translation works on subtitle *cues* — each cue keeps its
original start/end timing and only its text is translated, so the translated
SRT lines up 1:1 with the source timeline.
"""

import os
import re

from openai import OpenAI


# Preset languages offered in the UI. Free-form input is also accepted.
LANGUAGE_PRESETS = {
    "en": "English",
    "ja": "Japanese (日本語)",
    "zh": "Chinese (简体中文)",
    "la": "Latin (Latina)",
    "es": "Spanish (Español)",
    "fr": "French (Français)",
    "de": "German (Deutsch)",
    "vi": "Vietnamese (Tiếng Việt)",
    "id": "Indonesian (Bahasa Indonesia)",
    "ru": "Russian (Русский)",
    "ar": "Arabic (العربية)",
}

# Number of neighbouring cues shown as read-only context on each side of the
# target cue. Subtitles are sentence fragments, so context lets the model resolve
# omitted subjects and split sentences while still emitting exactly one line.
CONTEXT_WINDOW = 2


def resolve_language(raw):
    """Map a user-supplied code/name to (code, display_name).

    Accepts a preset code ('ja'), a preset name, or any free-form language
    string. Returns a filename-safe code and a human-readable name to put in
    the translation prompt.
    """
    value = (raw or "").strip()
    if not value:
        return None

    lowered = value.lower()
    if lowered in LANGUAGE_PRESETS:
        return lowered, LANGUAGE_PRESETS[lowered]

    # Match against preset display names too (e.g. user typed "English").
    for code, name in LANGUAGE_PRESETS.items():
        if lowered == name.lower() or lowered == name.split(" (")[0].lower():
            return code, name

    # Free-form: slugify a short code for filenames, keep the raw text as name.
    slug = re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")
    code = (slug or "lang")[:12]
    return code, value


def _clean(content):
    """Strip <think> reasoning blocks and ```code fences from a model reply."""
    content = (content or "").strip()
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    content = re.sub(r"^```(?:\w+)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)
    return content.strip()


class SubtitleTranslator:
    def __init__(self, backend="openai", model=None, api_key=None, base_url=None):
        self.backend = backend

        if backend == "lmstudio":
            self.client = OpenAI(
                base_url=base_url or os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"),
                api_key=api_key or "not-needed",
            )
            self.model = model or os.getenv("LMSTUDIO_MODEL", "local-model")
        else:  # openai cloud
            key = api_key or os.getenv("OPENAI_API_KEY")
            if not key:
                raise ValueError(
                    "OpenAI 번역을 사용하려면 OPENAI_API_KEY 환경변수가 필요합니다. "
                    "(또는 백엔드를 '로컬 LMStudio'로 선택하세요)"
                )
            self.client = OpenAI(api_key=key, base_url=base_url or os.getenv("OPENAI_BASE_URL"))
            self.model = model or os.getenv("OPENAI_TRANSLATE_MODEL", "gpt-4o-mini")

    def _chat(self, system_prompt, user_prompt):
        kwargs = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=400,
        )
        # On LMStudio, reasoning_effort="none" disables Qwen-style thinking so the
        # answer lands in `content` (and ~40x faster). Cloud models reject the
        # param for non-reasoning models, so only send it to the local backend.
        if self.backend == "lmstudio":
            resp = self.client.chat.completions.create(
                extra_body={"reasoning_effort": "none"}, **kwargs
            )
        else:
            resp = self.client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content

    def _translate_line(self, text, prev, nxt, target_name):
        """Translate one cue, using neighbouring cues as read-only context."""
        if not text.strip():
            return text

        context = ""
        if prev:
            context += "Previous lines: " + " / ".join(prev) + "\n"
        if nxt:
            context += "Next lines: " + " / ".join(nxt) + "\n"

        system_prompt = (
            f"You are a professional subtitle translator. Translate ONLY the target line into {target_name}. "
            "These are timed subtitle lines, so keep the translation concise and natural for on-screen reading. "
            "If the target line is a sentence fragment, keep it a fragment that fits naturally with the "
            "surrounding lines — do NOT complete or merge the whole sentence. "
            "Use the surrounding lines only to resolve omitted subjects and meaning. "
            "Preserve proper nouns and numbers. "
            "Output only the translation text — no quotes, labels, or notes."
        )
        user_prompt = context + "Target line: " + text

        try:
            content = _clean(self._chat(system_prompt, user_prompt))
            if not content:  # one retry on empty reply
                content = _clean(self._chat(system_prompt, user_prompt))
            return content or text
        except Exception:
            return text  # never drop a cue — fall back to the source text

    def translate_cues(self, cues, target_name, progress_cb=None, window=CONTEXT_WINDOW):
        """Translate cues into target_name, preserving start/end timing.

        Each cue is translated by its own request with the neighbouring cues as
        context, so the output is guaranteed 1:1 with the input — no shifted,
        merged, or dropped lines. cues: list of {start, end, text};
        returns the same list with translated text. progress_cb(done, total).
        """
        texts = [cue.get("text", "") for cue in cues]
        total = len(texts)
        out_texts = []

        for i, text in enumerate(texts):
            prev = texts[max(0, i - window):i]
            nxt = texts[i + 1:i + 1 + window]
            out_texts.append(self._translate_line(text, prev, nxt, target_name))
            if progress_cb:
                progress_cb(i + 1, total)

        return [
            {"start": cue["start"], "end": cue["end"], "text": out_texts[i]}
            for i, cue in enumerate(cues)
        ]
