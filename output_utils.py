import json
import os
from pathlib import Path

def get_downloads_path():
    """User's Downloads folder; override with STT_DOWNLOADS_DIR."""
    override = os.getenv("STT_DOWNLOADS_DIR")
    if override:
        return override
    return str(Path.home() / "Downloads")

def format_timestamp(seconds):
    """Converts seconds to HH:MM:SS,mmm format for SRT."""
    milliseconds = int((seconds % 1) * 1000)
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def save_as_text(results, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for segment in results:
            f.write(segment["text"] + " ")
    return output_path

# Subtitle line-length defaults: keep each cue readable (roughly 30~50 chars).
SUBTITLE_MAX_CHARS = 50
SUBTITLE_MIN_CHARS = 30
SENTENCE_ENDERS = ".?!…。？！"


def _word_text(word):
    return (word.get("text") or word.get("word") or "").strip()


def _make_word_cue(buffer, segment):
    text = " ".join(_word_text(w) for w in buffer).strip()
    starts = [w["start"] for w in buffer if w.get("start") is not None]
    ends = [w["end"] for w in buffer if w.get("end") is not None]
    start = starts[0] if starts else segment.get("start", 0)
    end = ends[-1] if ends else segment.get("end", start)
    return {"start": start, "end": end, "text": text}


def _cues_from_words(words, segment, max_chars, min_chars):
    cues = []
    buffer = []

    def buffer_text():
        return " ".join(_word_text(w) for w in buffer).strip()

    for word in words:
        wtext = _word_text(word)
        if not wtext:
            continue
        prospective = (buffer_text() + " " + wtext).strip() if buffer else wtext
        if buffer and len(prospective) > max_chars:
            cues.append(_make_word_cue(buffer, segment))
            buffer = [word]
        else:
            buffer.append(word)

        current = buffer_text()
        if len(current) >= min_chars and current[-1:] in SENTENCE_ENDERS:
            cues.append(_make_word_cue(buffer, segment))
            buffer = []

    if buffer:
        cues.append(_make_word_cue(buffer, segment))
    return cues


def _chunk_text(text, max_chars):
    chunks = []
    current = ""
    for word in text.split():
        if len(word) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            for i in range(0, len(word), max_chars):
                chunks.append(word[i:i + max_chars])
            continue
        candidate = (current + " " + word).strip() if current else word
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = word
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks or [text]


def _cues_from_text(text, segment, max_chars):
    chunks = _chunk_text(text, max_chars)
    seg_start = segment.get("start", 0)
    seg_end = segment.get("end", seg_start)
    duration = max(seg_end - seg_start, 0)
    total = sum(len(c) for c in chunks) or 1

    cues = []
    consumed = 0
    for chunk in chunks:
        c_start = seg_start + duration * (consumed / total)
        consumed += len(chunk)
        c_end = seg_start + duration * (consumed / total)
        cues.append({"start": round(c_start, 2), "end": round(c_end, 2), "text": chunk.strip()})
    return cues


def split_into_subtitle_cues(results, max_chars=SUBTITLE_MAX_CHARS, min_chars=SUBTITLE_MIN_CHARS):
    """Split transcript segments into subtitle cues of roughly max_chars characters.

    Uses word-level timestamps when available for accurate per-cue timing,
    otherwise falls back to proportional time-splitting of the segment.
    """
    cues = []
    for segment in results:
        text = (segment.get("text") or "").strip()
        if not text:
            continue
        if len(text) <= max_chars:
            cues.append({"start": segment.get("start", 0), "end": segment.get("end", 0), "text": text})
            continue

        words = segment.get("words")
        if words:
            cues.extend(_cues_from_words(words, segment, max_chars, min_chars))
        else:
            cues.extend(_cues_from_text(text, segment, max_chars))
    return cues


def write_cue_srt(cues, output_path):
    """Write a list of {start, end, text} cues as an SRT file."""
    with open(output_path, "w", encoding="utf-8") as f:
        for i, cue in enumerate(cues, 1):
            f.write(f"{i}\n")
            f.write(f"{format_timestamp(cue['start'])} --> {format_timestamp(cue['end'])}\n")
            f.write(f"{cue['text']}\n\n")
    return output_path


def write_bilingual_srt(original_cues, translated_cues, output_path):
    """Write a dual-language SRT: original line on top, translation below.

    Cue lists are assumed aligned 1:1 (translate_cues preserves order/length).
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for i, (orig, trans) in enumerate(zip(original_cues, translated_cues), 1):
            f.write(f"{i}\n")
            f.write(f"{format_timestamp(orig['start'])} --> {format_timestamp(orig['end'])}\n")
            f.write(f"{orig['text']}\n{trans['text']}\n\n")
    return output_path


def write_cue_txt(cues, output_path):
    """Write cue text only, one cue per line."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(cue["text"] for cue in cues))
    return output_path


def save_as_srt(results, output_path, max_chars=SUBTITLE_MAX_CHARS, min_chars=SUBTITLE_MIN_CHARS):
    cues = split_into_subtitle_cues(results, max_chars=max_chars, min_chars=min_chars)
    return write_cue_srt(cues, output_path)

def save_as_json(results, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    return output_path

def save_as_diarized_text(diarized_results, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for seg in diarized_results:
            start_m, start_s = divmod(seg['start'], 60)
            end_m, end_s = divmod(seg['end'], 60)
            timestamp = f"[{int(start_m):02d}:{int(start_s):02d} - {int(end_m):02d}:{int(end_s):02d}]"
            f.write(f"{timestamp} {seg['speaker']}: {seg['text']}\n")
    return output_path

def save_as_markdown(summary_text, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# AI Meeting Summary\n\n")
        f.write(summary_text)
    return output_path

def export_all(results, base_path, diarized_results=None, summary_text=None):
    save_as_text(results, base_path + ".txt")
    save_as_srt(results, base_path + ".srt")
    save_as_json(results, base_path + ".json")
    
    if diarized_results:
        save_as_diarized_text(diarized_results, base_path + "_diarized.txt")
        save_as_json(diarized_results, base_path + "_diarized.json")
        
    if summary_text:
        save_as_markdown(summary_text, base_path + "_summary.md")
        
    return base_path
