import time
from faster_whisper import WhisperModel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

class STTEngine:
    def __init__(self, model_size="base", device=None, compute_type=None):
        """
        Initialize the Whisper model.
        Auto-detects CUDA if available, otherwise defaults to CPU.
        """
        self.model_size = model_size
        
        # Auto-detect CUDA
        if device is None:
            import ctranslate2
            if ctranslate2.get_cuda_device_count() > 0:
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        # Set optimal compute_type
        if compute_type is None:
            if self.device == "cuda":
                self.compute_type = "float16"
            else:
                self.compute_type = "int8"
        else:
            self.compute_type = compute_type
            
        self.model = None
        print(f"Engine initialized: device={self.device}, compute_type={self.compute_type}")

    def load_model(self):
        if self.model is None:
            print(f"Loading Whisper model '{self.model_size}'...")
            self.model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
            print("Model loaded successfully.")

    def transcribe(self, audio_path, language=None, initial_prompt=None, word_timestamps=False):
        self.load_model()
        
        # segments is an iterable
        segments, info = self.model.transcribe(
            audio_path, 
            beam_size=5, 
            language=language, 
            initial_prompt=initial_prompt,
            condition_on_previous_text=True,
            word_timestamps=word_timestamps
        )
        
        print(f"Detected language '{info.language}' with probability {info.language_probability:.2f}")
        
        results = []
        
        # We wrap the iteration in a progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("Transcribing...", total=None) # Total duration unknown initially
            
            for segment in segments:
                words = []
                if word_timestamps and segment.words:
                    for word in segment.words:
                        words.append({
                            "start": word.start,
                            "end": word.end,
                            "text": word.word.strip()
                        })
                
                results.append({
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                    "words": words if word_timestamps else None
                })
                # Update progress (optional, since segments is a generator, total is hard to guess)
                progress.update(task, advance=1, description=f"Transcribed: {segment.text[:30]}...")
                
        return results, info

if __name__ == "__main__":
    # Quick test
    engine = STTEngine(model_size="tiny")
    # engine.transcribe("sample.mp3") 
