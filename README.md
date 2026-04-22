# Python STT Pro

A high-performance Speech-to-Text (STT) project using [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and optimized for Mac hardware.

## Features

- **High Speed**: Uses `faster-whisper` for significantly faster transcription compared to the original OpenAI implementation.
- **Multi-format Export**: Exports results to `.txt`, `.srt` (for subtitles), and `.json` directly to your **Downloads** folder.
- **Google Drive Integration**: Provide a public Drive link, and the tool will handle the download and transcription automatically.
- **Beautiful CLI**: Interactive and colorful command-line interface using `rich`.

## Installation & Running

### Mac / Linux
```bash
chmod +x run.sh
./run.sh --help
```

### Windows
1. Open File Explorer and navigate to the project folder.
2. Double-click `run.bat`.
3. To use specific commands from the CMD/PowerShell:
   ```cmd
   run.bat transcribe "C:\path\to\audio.mp3"
   ```

---

## Usage Examples

### 1. High-Accuracy Transcription (Recommended)
```bash
./run.sh transcribe audio.mp3 --model large-v3-turbo --prompt "입시 컨설팅 대화"
```

### 2. Batch Processing
Create a file named `list.txt` and add one link or path per line:
```text
https://drive.google.com/file/d/LINK1/view
C:\Users\Name\Music\audio2.mp3
https://drive.google.com/file/d/LINK2/view
```
Then run:
```bash
./run.sh batch list.txt --prompt "중고등 입시 컨설팅"
```

### 3. GPU Acceleration (Windows/Linux with NVIDIA GPU)
If you have a 24GB RAM machine with an NVIDIA GPU, add `--device cuda`:
```cmd
run.bat batch list.txt --device cuda --compute_type float16
```

---

## Project Structure
- `run.sh` / `run.bat`: Entry points for easy execution.
- `main.py`: CLI logic.
- `stt_engine.py`: Whisper engine wrapper.
- `gdrive_utils.py`: Google Drive downloader.
- `output_utils.py`: Cross-platform Download folder handler.
