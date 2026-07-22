import os
import time
import argparse
from rich.console import Console
from rich.panel import Panel
from stt_engine import STTEngine
from audio_utils import record_audio, check_file_exists
from gdrive_utils import download_from_gdrive, is_gdrive_url
from output_utils import export_all, get_downloads_path
from media_scan import collect_supported_files
from batch_state import CompletionIndex
from diarizer import create_diarizer, align_words_with_speakers
from summarizer import LMStudioSummarizer


console = Console()

def main():
    parser = argparse.ArgumentParser(description="Python STT Pro - High Performance Speech to Text")
    parser.add_argument("--device", default=None, help="Device to use (cpu, cuda)")
    parser.add_argument("--compute_type", default=None, help="Compute type (int8, float16)")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Transcribe command
    trans_parser = subparsers.add_parser("transcribe", help="Transcribe an audio file")
    trans_parser.add_argument("file", help="Path to the audio file")
    trans_parser.add_argument("--model", default="large-v3-turbo", help="Whisper model size (base, small, medium, large-v3-turbo)")
    trans_parser.add_argument("--lang", default=None, help="Language code (e.g., 'ko', 'en')")
    trans_parser.add_argument("--prompt", default=None, help="Initial prompt to provide context for the transcription")
    trans_parser.add_argument("--diarize", action="store_true", help="Enable speaker diarization")
    trans_parser.add_argument("--summary", action="store_true", help="Enable AI summary using LMStudio")

    # Record command
    record_parser = subparsers.add_parser("record", help="Record from microphone and transcribe")
    record_parser.add_argument("--duration", type=int, default=None, help="Recording duration in seconds")
    record_parser.add_argument("--model", default="large-v3-turbo", help="Whisper model size")
    record_parser.add_argument("--output", default="data/recorded_audio.wav", help="Path to save recorded audio")
    record_parser.add_argument("--prompt", default=None, help="Initial prompt for context")

    # GDrive command
    gdrive_parser = subparsers.add_parser("gdrive", help="Download from Google Drive and transcribe")
    gdrive_parser.add_argument("url", help="Google Drive link")
    gdrive_parser.add_argument("--model", default="large-v3-turbo", help="Whisper model size")
    gdrive_parser.add_argument("--lang", default="ko", help="Forced language (default: ko)")
    gdrive_parser.add_argument("--prompt", default=None, help="Initial prompt for context")
    gdrive_parser.add_argument("--diarize", action="store_true", help="Enable speaker diarization")
    gdrive_parser.add_argument("--summary", action="store_true", help="Enable AI summary using LMStudio")

    # Batch command
    batch_parser = subparsers.add_parser("batch", help="Process a folder of media files, or links/files from a text file")
    batch_parser.add_argument("input_file", help="Folder to scan recursively, or a text file with one URL/path per line")
    batch_parser.add_argument("--model", default="large-v3-turbo", help="Whisper model size")
    batch_parser.add_argument("--lang", default="ko", help="Language code")
    batch_parser.add_argument("--prompt", default=None, help="Base prompt for all files")
    batch_parser.add_argument("--fresh", action="store_true", help="Ignore saved progress and process everything again")
    batch_parser.add_argument("--diarize", action="store_true", help="Enable speaker diarization for every file")
    batch_parser.add_argument("--num-speakers", type=int, default=None, help="Fixed number of speakers (optional)")

    args = parser.parse_args()

    console.print(Panel.fit("[bold blue]Python STT Pro[/bold blue]\n[italic]Optimized Speech-to-Text[/italic]", border_style="cyan"))

    # Initialize Engine with hardware flags
    def get_engine(model_size):
        return STTEngine(model_size=model_size, device=args.device, compute_type=args.compute_type)

    def process_advanced_features(audio_path, results):
        diarized_results = None
        summary_text = None
        
        if getattr(args, 'diarize', False) or getattr(args, 'summary', False):
            try:
                diarizer = create_diarizer()
                speaker_segments = diarizer.run_diarization(audio_path)
                diarized_results = align_words_with_speakers(results, speaker_segments)
                
                if getattr(args, 'summary', False):
                    summarizer = LMStudioSummarizer()
                    summary_text = summarizer.summarize_timeline(diarized_results)
            except (ImportError, RuntimeError) as e:
                console.print(f"[yellow]Warning: {e}[/yellow]")
                console.print("[yellow]Skipping diarization and summary. Continuing with transcription only.[/yellow]")
                
        return diarized_results, summary_text



    if args.command == "transcribe":
        if not check_file_exists(args.file):
            console.print(f"[red]Error: File '{args.file}' not found.[/red]")
            return

        engine = get_engine(args.model)
        # Always capture word timestamps so SRT cues can be split accurately.
        results, info = engine.transcribe(args.file, language=args.lang, initial_prompt=args.prompt, word_timestamps=True)
        
        diarized_results, summary_text = process_advanced_features(args.file, results)
        
        base_name = os.path.splitext(args.file)[0]
        export_all(results, base_name, diarized_results, summary_text)
        
        console.print(f"\n[bold green]Transcription complete![/bold green]")
        console.print(f"Results saved to: {base_name}.txt, .srt, .json, and more if requested.")
        
        # Show a preview
        preview = " ".join([s["text"] for s in results[:3]])
        console.print(Panel(f"[bold]Preview:[/bold]\n{preview}...", title="Output Preview"))

    elif args.command == "record":
        if not os.path.exists("data"):
            os.makedirs("data")
            
        audio_path = record_audio(args.output, duration=args.duration)
        if audio_path:
            engine = get_engine(args.model)
            results, info = engine.transcribe(audio_path, initial_prompt=args.prompt, word_timestamps=True)
        
        # Result filename in Downloads
        file_name = f"STT_{int(time.time())}"
        out_base = os.path.join(get_downloads_path(), file_name)
        export_all(results, out_base)
        
        console.print(f"\n[bold green]Transcription complete![/bold green]")
        console.print(f"Results saved to: [cyan]{out_base}.txt, .srt, .json[/cyan]")

    elif args.command == "gdrive":
        if not os.path.exists("data"):
            os.makedirs("data")
            
        temp_path = os.path.join("data", f"gdrive_file_{int(time.time())}.mp3")
        audio_path = download_from_gdrive(args.url, temp_path)
        
        if audio_path:
            engine = get_engine(args.model)
            # Always capture word timestamps so SRT cues can be split accurately.
            results, info = engine.transcribe(audio_path, language=args.lang, initial_prompt=args.prompt, word_timestamps=True)
            
            diarized_results, summary_text = process_advanced_features(audio_path, results)
            
            # Result filename based on input
            file_name = f"GDrive_STT_{int(time.time())}"
            out_base = os.path.join(get_downloads_path(), file_name)
            export_all(results, out_base, diarized_results, summary_text)
            
            console.print(f"\n[bold green]Transcription complete![/bold green]")
            console.print(f"Results saved to: [cyan]{out_base}[/cyan] (Formats: txt, srt, json, etc)")

    elif args.command == "batch":
        input_path = os.path.expanduser(args.input_file.strip().strip('"').strip("'"))

        batch_options = None
        if args.diarize:
            batch_options = {"diarize": True}
            if args.num_speakers:
                batch_options["num_speakers"] = args.num_speakers

        # Build the work list: (label, resume_key, source, is_url).
        # resume_key is None for plain file lines until the file is confirmed to exist.
        items = []
        state = CompletionIndex(options=batch_options)
        if os.path.isdir(input_path):
            media_files = collect_supported_files([input_path])
            if not media_files:
                console.print(f"[red]Error: No supported audio/video files found in '{input_path}'.[/red]")
                return
            for path in media_files:
                items.append((os.path.relpath(path, input_path), CompletionIndex.file_key(path), path, False))
            if args.fresh:
                cleared = state.reset_prefix(input_path)
                console.print(f"[yellow]--fresh: 이 폴더의 완료 기록 {cleared}개를 지우고 전부 다시 처리합니다.[/yellow]")
        elif os.path.isfile(input_path):
            with open(input_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            for line in lines:
                expanded = os.path.expanduser(line.strip('"').strip("'"))
                if is_gdrive_url(line):
                    items.append((line, "url:" + line, line, True))
                elif os.path.isdir(expanded):
                    for path in collect_supported_files([expanded]):
                        items.append((path, CompletionIndex.file_key(path), path, False))
                else:
                    items.append((expanded, None, expanded, False))
            if args.fresh:
                state.reset_files([label for label, _, _, is_url in items if not is_url])
                console.print("[yellow]--fresh: 목록에 있는 파일들의 완료 기록을 지우고 다시 처리합니다.[/yellow]")
        else:
            console.print(f"[red]Error: '{args.input_file}' is not an existing folder or batch list file.[/red]")
            return

        done_before = sum(1 for _, key, _, _ in items if key and state.is_done(key))
        console.print(
            f"Starting batch process: [bold]{len(items)}[/bold] items total, "
            f"[green]{done_before}[/green] already done, [bold]{len(items) - done_before}[/bold] to process."
        )

        diarizer = None
        if args.diarize and done_before < len(items):
            # Fail fast (missing token/model access) before transcribing anything.
            try:
                diarizer = create_diarizer()
            except (ImportError, RuntimeError) as e:
                console.print(f"[red]{e}[/red]")
                return

        engine = None  # Loaded lazily so a fully-completed batch skips the model load
        for i, (label, key, source, is_url) in enumerate(items, 1):
            if key is None and check_file_exists(source):
                key = CompletionIndex.file_key(source)

            if key and state.is_done(key):
                console.print(f"[cyan][{i}/{len(items)}] 건너뜀 (완료됨): {label}[/cyan]")
                continue

            console.print(f"\n[bold yellow]Processing [{i}/{len(items)}]: {label}[/bold yellow]")
            try:
                if is_url:
                    temp_path = os.path.join("data", f"batch_gdrive_{int(time.time())}.mp3")
                    audio_path = download_from_gdrive(source, temp_path)
                else:
                    audio_path = source

                if audio_path and check_file_exists(audio_path):
                    if engine is None:
                        engine = get_engine(args.model)
                    results, info = engine.transcribe(audio_path, language=args.lang, initial_prompt=args.prompt, word_timestamps=True)

                    diarized_results = None
                    if diarizer is not None:
                        console.print("[blue]화자 분리 중...[/blue]")
                        speaker_segments = diarizer.run_diarization(audio_path, num_speakers=args.num_speakers)
                        diarized_results = align_words_with_speakers(results, speaker_segments)

                    file_name = f"Batch_STT_{i}_{int(time.time())}"
                    out_base = os.path.join(get_downloads_path(), file_name)
                    export_all(results, out_base, diarized_results)
                    if key:
                        state.mark_done(key, {"output_base": out_base})
                    console.print(f"[green]Done. Saved to {out_base}[/green]")
                else:
                    if key:
                        state.mark_failed(key, "file not found or download failed")
                    console.print(f"[red]Skipped: Could not find or download {label}[/red]")
            except Exception as e:
                if key:
                    state.mark_failed(key, e)
                console.print(f"[red]Error processing {label}: {str(e)}[/red]")
                console.print("[yellow]Continuing with next item...[/yellow]")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
