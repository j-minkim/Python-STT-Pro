import sounddevice as sd
import soundfile as sf
import numpy as np
import os
import time
from rich.console import Console

console = Console()

def record_audio(output_path, duration=None, fs=16000):
    """
    Record audio from the microphone.
    If duration is None, it records until interrupted (Ctrl+C).
    """
    console.print(f"[bold green]Recording...[/bold green] (Press [bold red]Ctrl+C[/bold red] to stop if duration is not set)")
    
    recording = []
    
    try:
        if duration:
            # Fixed duration recording
            recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
            sd.wait()
        else:
            # Continuous recording
            def callback(indata, frames, time, status):
                if status:
                    print(status)
                recording.append(indata.copy())

            with sd.InputStream(samplerate=fs, channels=1, callback=callback):
                while True:
                    time.sleep(0.1)
                    
    except KeyboardInterrupt:
        console.print("\n[yellow]Recording stopped by user.[/yellow]")
    
    if not duration and recording:
        recording = np.concatenate(recording, axis=0)
    
    if len(recording) > 0:
        sf.write(output_path, recording, fs)
        console.print(f"Audio saved to: [cyan]{output_path}[/cyan]")
        return output_path
    else:
        console.print("[red]No audio recorded.[/red]")
        return None

def check_file_exists(file_path):
    return os.path.exists(file_path)
