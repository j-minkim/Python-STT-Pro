#!/usr/bin/env python3
"""Side-by-side translation comparison: Qwen3.6 (general) vs TranslateGemma (specialist).

Both models are served by the local LM Studio server (OpenAI-compatible, port 1234).
Each is called the way it expects:
  - Qwen3.6      : general instruct -> JSON-array batch translation (one call per language)
  - TranslateGemma: translation specialist -> one direct "translate to X" call per line

Usage:
    # Start LM Studio server first:  lms server start
    # Load both models in LM Studio (or enable JIT loading), then:
    python compare_translations.py
    python compare_translations.py --srt data/outputs/<id>.srt --n 8 --langs en,ja,zh
    python compare_translations.py --qwen <model_key> --gemma <model_key>
"""

import argparse
import json
import re
import sys
import time

from openai import OpenAI

BASE_URL = "http://localhost:1234/v1"

LANG_NAMES = {
    "en": "English",
    "ja": "Japanese",
    "zh": "Chinese (Simplified)",
    "es": "Spanish",
    "fr": "French",
}


def parse_srt_texts(path, limit):
    """Return the spoken-text lines from an SRT (skip index + timestamp lines)."""
    texts = []
    with open(path, encoding="utf-8") as f:
        block_lines = []
        for raw in f:
            line = raw.rstrip("\n")
            if line.strip() == "":
                _flush_block(block_lines, texts)
                block_lines = []
            else:
                block_lines.append(line)
        _flush_block(block_lines, texts)
    return texts[:limit]


def _flush_block(block_lines, texts):
    # Block = [index, "start --> end", text...]; keep only the text part.
    if len(block_lines) >= 3:
        texts.append(" ".join(block_lines[2:]).strip())


SAMPLE = [
    "저는 매우 가난한 집에서 태어났습니다.",
    "그런데 흔히 말하는 가난한 집 클리셰 같은 부모님은 아니셨어요.",
    "경제적으로는 어려웠지만 저는 행복하게 자랐습니다.",
    "늘 사랑한다고 표현해 주셨고요.",
    "그 덕분에 저는 도전을 두려워하지 않게 되었습니다.",
]


def detect_models(client):
    """Return (qwen_id, gemma_id) by matching loaded LM Studio model keys."""
    qwen = gemma = None
    try:
        for m in client.models.list().data:
            mid = m.id.lower()
            if "translategemma" in mid or "translate-gemma" in mid:
                gemma = gemma or m.id
            elif "qwen" in mid:
                qwen = qwen or m.id
    except Exception as e:
        print(f"[warn] 모델 목록 조회 실패: {e}", file=sys.stderr)
    return qwen, gemma


def _strip_think(content):
    """Remove <think> blocks and ```json fences some models wrap output in."""
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)
    return content.strip()


def translate_qwen(client, model, texts, lang_name, no_think=True):
    """General-instruct path: translate the whole list in one JSON-array call.

    Qwen3.6 is a reasoning model — append /no_think so the answer lands in
    `content` (not the reasoning channel) and the JSON stays parseable.
    """
    system = (
        f"You are a professional subtitle translator. Translate each line into {lang_name}. "
        "Keep each translation concise and natural for on-screen reading. Preserve meaning and proper nouns. "
        'Return ONLY a JSON object {"translations": [...]} with EXACTLY '
        f"{len(texts)} items, same order as input."
    )
    user = json.dumps({"lines": texts}, ensure_ascii=False)
    # reasoning_effort="none" is the switch that actually disables Qwen3.6 thinking
    # in LM Studio (chat_template_kwargs enable_thinking is ignored) — ~40x faster.
    extra = {"reasoning_effort": "none"} if no_think else {}
    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.2,
        max_tokens=4000,
        extra_body=extra,
    )
    dt = time.time() - t0
    msg = resp.choices[0].message
    content = _strip_think(msg.content or "")
    if not content:  # answer may have gone to the reasoning channel
        content = _strip_think(getattr(msg, "reasoning_content", "") or "")
    try:
        data = json.loads(content)
        out = data["translations"] if isinstance(data, dict) else data
    except (json.JSONDecodeError, KeyError, TypeError):
        m = re.search(r"\{.*\}|\[.*\]", content, re.DOTALL)
        if m:
            data = json.loads(m.group(0))
            out = data["translations"] if isinstance(data, dict) else data
        else:
            out = [content]
    if len(out) != len(texts):
        out = (out + [""] * len(texts))[: len(texts)]
    return [str(x) for x in out], dt


def translate_gemma(client, model, texts, lang_name, source_lang="Korean"):
    """Translation-specialist path: one direct call per line via /v1/completions.

    TranslateGemma's strict chat template needs per-part lang codes that LM Studio
    strips from chat messages, so we bypass the chat template and feed a raw
    Gemma-formatted prompt to the text-completions endpoint instead.
    """
    out = []
    t0 = time.time()
    for text in texts:
        prompt = (
            "<start_of_turn>user\n"
            f"Translate the following {source_lang} text to {lang_name}. "
            "Output only the translation.\n"
            f"{text}<end_of_turn>\n<start_of_turn>model\n"
        )
        resp = client.completions.create(
            model=model,
            prompt=prompt,
            temperature=0.2,
            max_tokens=256,
            stop=["<end_of_turn>"],
        )
        out.append((resp.choices[0].text or "").strip())
    return out, time.time() - t0


def _context_block(texts, i, window):
    prev = texts[max(0, i - window):i]
    nxt = texts[i + 1:i + 1 + window]
    block = ""
    if prev:
        block += "이전 줄: " + " / ".join(prev) + "\n"
    if nxt:
        block += "다음 줄: " + " / ".join(nxt) + "\n"
    return block


SLIDING_RULES = (
    "Translate ONLY the target line into {lang}. "
    "If it is a sentence fragment, keep it a fragment that fits naturally with the "
    "surrounding lines — do NOT complete or merge the whole sentence. "
    "Use the context only to resolve omitted subjects and meaning. "
    "Output only the translation, no quotes or notes."
)


def translate_qwen_sliding(client, model, texts, lang_name, window=2):
    """1:1 per-line translation with neighbouring cues as read-only context.

    Guarantees exactly one output per cue (no shift/drop) while still giving the
    model context to handle split sentences and omitted subjects.
    """
    out = []
    t0 = time.time()
    system = SLIDING_RULES.format(lang=lang_name)
    for i, text in enumerate(texts):
        user = _context_block(texts, i, window) + "번역할 줄: " + text
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.2,
            max_tokens=300,
            extra_body={"reasoning_effort": "none"},
        )
        out.append(_strip_think(resp.choices[0].message.content or "").strip())
    return out, time.time() - t0


def translate_gemma_sliding(client, model, texts, lang_name, window=2, source_lang="Korean"):
    """Same 1:1 + context idea for TranslateGemma via the /v1/completions endpoint."""
    out = []
    t0 = time.time()
    rules = SLIDING_RULES.format(lang=lang_name)
    for i, text in enumerate(texts):
        ctx = _context_block(texts, i, window)
        prompt = (
            "<start_of_turn>user\n"
            f"You translate {source_lang} subtitles. {rules}\n"
            f"{ctx}번역할 줄: {text}<end_of_turn>\n<start_of_turn>model\n"
        )
        resp = client.completions.create(
            model=model, prompt=prompt, temperature=0.2, max_tokens=256, stop=["<end_of_turn>"]
        )
        out.append((resp.choices[0].text or "").strip())
    return out, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--srt", help="Korean SRT file to sample from")
    ap.add_argument("--n", type=int, default=5, help="number of lines to compare")
    ap.add_argument("--langs", default="en,ja,zh")
    ap.add_argument("--method", choices=["sliding", "batch"], default="sliding",
                    help="sliding = 1:1 per-line with context (safe); batch = Qwen JSON batch")
    ap.add_argument("--window", type=int, default=2, help="context lines each side (sliding)")
    ap.add_argument("--qwen", help="LM Studio model key for Qwen (auto-detected if omitted)")
    ap.add_argument("--gemma", help="LM Studio model key for TranslateGemma (auto-detected if omitted)")
    ap.add_argument("--base-url", default=BASE_URL)
    args = ap.parse_args()

    texts = parse_srt_texts(args.srt, args.n) if args.srt else SAMPLE[: args.n]
    if not texts:
        print("입력 자막이 없습니다.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(base_url=args.base_url, api_key="not-needed")

    qwen, gemma = args.qwen, args.gemma
    if not (qwen and gemma):
        d_qwen, d_gemma = detect_models(client)
        qwen = qwen or d_qwen
        gemma = gemma or d_gemma

    print(f"입력 {len(texts)}줄 | Qwen={qwen or '(없음)'} | TranslateGemma={gemma or '(없음)'}\n")

    langs = [code.strip() for code in args.langs.split(",") if code.strip()]
    for code in langs:
        lang_name = LANG_NAMES.get(code, code)
        print("=" * 78)
        print(f"  ▶ {lang_name} ({code})")
        print("=" * 78)

        print(f"  (method={args.method}"
              + (f", window={args.window}" if args.method == "sliding" else "") + ")")
        qwen_out = gemma_out = None
        if qwen:
            try:
                if args.method == "sliding":
                    qwen_out, qt = translate_qwen_sliding(client, qwen, texts, lang_name, args.window)
                else:
                    qwen_out, qt = translate_qwen(client, qwen, texts, lang_name)
                print(f"  [Qwen3.6]        {qt:5.1f}s")
            except Exception as e:
                print(f"  [Qwen3.6] 오류: {e}")
        if gemma:
            try:
                if args.method == "sliding":
                    gemma_out, gt = translate_gemma_sliding(client, gemma, texts, lang_name, args.window)
                else:
                    gemma_out, gt = translate_gemma(client, gemma, texts, lang_name)
                print(f"  [TranslateGemma] {gt:5.1f}s (줄 단위 {len(texts)}회 호출)")
            except Exception as e:
                print(f"  [TranslateGemma] 오류: {e}")

        # Alignment guard: per-line methods must return exactly one item per cue.
        for label, arr in (("Qwen", qwen_out), ("TranslateGemma", gemma_out)):
            if arr is not None and len(arr) != len(texts):
                print(f"  ⚠️ {label} 정렬 경고: {len(arr)}개 출력 (입력 {len(texts)}개)")
        print()

        for i, src in enumerate(texts):
            print(f"  [{i+1}] 원문 : {src}")
            if qwen_out:
                print(f"      Qwen : {qwen_out[i]}")
            if gemma_out:
                print(f"      Gemma: {gemma_out[i]}")
            print()


if __name__ == "__main__":
    main()
