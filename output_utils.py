import json
import os
from pathlib import Path

def get_downloads_path():
    """Returns the user's Downloads folder path."""
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

def save_as_srt(results, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(results, 1):
            start = format_timestamp(segment["start"])
            end = format_timestamp(segment["end"])
            f.write(f"{i}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{segment['text']}\n\n")
    return output_path

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
