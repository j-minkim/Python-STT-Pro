import argparse
import os
import sys
from rich.console import Console
from rich.panel import Panel
from stt_engine import STTEngine
from audio_utils import record_audio, check_file_exists
from gdrive_utils import download_from_gdrive, is_gdrive_url
from output_utils import export_all, get_downloads_path
import time

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

    # Batch command
    batch_parser = subparsers.add_parser("batch", help="Process multiple links/files from a text file")
    batch_parser.add_argument("input_file", help="Text file containing one URL or path per line")
    batch_parser.add_argument("--model", default="large-v3-turbo", help="Whisper model size")
    batch_parser.add_argument("--lang", default="ko", help="Language code")
    batch_parser.add_argument("--prompt", default=None, help="Base prompt for all files")

    args = parser.parse_args()

    console.print(Panel.fit("[bold blue]Python STT Pro[/bold blue]\n[italic]Optimized Speech-to-Text[/italic]", border_style="cyan"))

    # Initialize Engine with hardware flags
    def get_engine(model_size):
        return STTEngine(model_size=model_size, device=args.device, compute_type=args.compute_type)

    if args.command == "transcribe":
        if not check_file_exists(args.file):
            console.print(f"[red]Error: File '{args.file}' not found.[/red]")
            return

        engine = get_engine(args.model)
        results, info = engine.transcribe(args.file, language=args.lang, initial_prompt=args.prompt)
        
        base_name = os.path.splitext(args.file)[0]
        export_all(results, base_name)
        
        console.print(f"\n[bold green]Transcription complete![/bold green]")
        console.print(f"Results saved to: {base_name}.txt, .srt, .json")
        
        # Show a preview
        preview = " ".join([s["text"] for s in results[:3]])
        console.print(Panel(f"[bold]Preview:[/bold]\n{preview}...", title="Output Preview"))

    elif args.command == "record":
        if not os.path.exists("data"):
            os.makedirs("data")
            
        audio_path = record_audio(args.output, duration=args.duration)
        if audio_path:
            engine = get_engine(args.model)
            results, info = engine.transcribe(audio_path, initial_prompt=args.prompt)
        
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
            results, info = engine.transcribe(audio_path, language=args.lang, initial_prompt=args.prompt)
            
            # Result filename based on input
            file_name = f"GDrive_STT_{int(time.time())}"
            out_base = os.path.join(get_downloads_path(), file_name)
            export_all(results, out_base)
            
            console.print(f"\n[bold green]Transcription complete![/bold green]")
            console.print(f"Results saved to: [cyan]{out_base}.txt, .srt, .json[/cyan]")

    elif args.command == "batch":
        if not check_file_exists(args.input_file):
            console.print(f"[red]Error: Batch file '{args.input_file}' not found.[/red]")
            return

        with open(args.input_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        console.print(f"Starting batch process for [bold]{len(lines)}[/bold] items...")
        engine = get_engine(args.model) # Loaded once for the whole batch
        
        for i, line in enumerate(lines, 1):
            console.print(f"\n[bold yellow]Processing [{i}/{len(lines)}]: {line}[/bold yellow]")
            try:
                if is_gdrive_url(line):
                    temp_path = os.path.join("data", f"batch_gdrive_{int(time.time())}.mp3")
                    audio_path = download_from_gdrive(line, temp_path)
                else:
                    audio_path = line

                if audio_path and check_file_exists(audio_path):
                    results, info = engine.transcribe(audio_path, language=args.lang, initial_prompt=args.prompt)
                    
                    file_name = f"Batch_STT_{i}_{int(time.time())}"
                    out_base = os.path.join(get_downloads_path(), file_name)
                    export_all(results, out_base)
                    console.print(f"[green]Done. Saved to {out_base}[/green]")
                else:
                    console.print(f"[red]Skipped: Could not find or download {line}[/red]")
            except Exception as e:
                console.print(f"[red]Error processing {line}: {str(e)}[/red]")
                console.print("[yellow]Continuing with next item...[/yellow]")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
