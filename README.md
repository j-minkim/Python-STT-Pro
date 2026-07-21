# Python STT Pro

[faster-whisper](https://github.com/SYSTRAN/faster-whisper) 기반의 고성능 음성-텍스트 변환(STT) 도구입니다. Mac 하드웨어에 최적화되어 있으며, **웹 인터페이스**와 **CLI**를 모두 제공합니다. 전사뿐 아니라 **자막 자동 분할**, **다국어 번역(영·일·중·라틴 등)**, **결과물 자동 저장**까지 한 번에 처리합니다.

---

## 주요 기능

- **고속 전사**: `faster-whisper`로 OpenAI 원본 대비 훨씬 빠른 변환. `large-v3-turbo` 등 모델 선택 가능.
- **웹 인터페이스**: 드래그&드롭 업로드, 실시간 전사 스트리밍, 다운로드 버튼을 갖춘 글래스모피즘 UI.
- **자막 자동 분할**: 한 자막이 너무 길게 나오지 않도록 **30~50자(조절 가능)** 기준으로 자동 분할. 단어 단위 타임스탬프로 타이밍을 정확히 유지.
- **다국어 번역**: 생성된 자막을 영어·일본어·중국어·라틴어 등으로 번역. **줄이 밀리거나 사라지지 않는 슬라이딩 문맥 1:1 방식**으로, 언어별 SRT·이중자막·TXT를 함께 생성.
- **다양한 출력 포맷**: `.txt`, `.srt`(자막), `.json` + 번역본. 완료 시 **자동으로 다운로드 폴더에 저장**.
- **Google Drive 연동**: 공개 Drive 파일/폴더 링크를 넣으면 자동 다운로드 후 전사(폴더 배치 처리 지원).
- **화자 분리 & 요약 (CLI)**: 누가 언제 말했는지 분리하고, LMStudio 로컬 LLM으로 회의 내용을 요약.
- **아름다운 CLI**: `rich` 기반의 인터랙티브 컬러 터미널 인터페이스.

---

## 설치

```bash
git clone https://github.com/j-minkim/Python-STT-Pro.git
cd Python-STT-Pro
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> 첫 전사 시 Whisper 모델이 캐시에 없으면 자동으로 내려받습니다.

---

## 1) 웹 인터페이스 (권장)

```bash
source .venv/bin/activate
python3 web_app.py
```

브라우저에서 **http://localhost:5000** 접속.
(5000번이 macOS AirPlay 수신기와 충돌하면, 코드의 포트를 5001 등으로 바꾸거나 시스템 설정에서 AirPlay 수신기를 끄세요.)

**사용 흐름**
1. 오디오/영상 파일 드래그(또는 Google Drive 링크 입력)
2. (선택) **자막 한 줄 길이** 최대/최소 글자 수 조정
3. (선택) **자막 번역** 언어 체크(영/일/중/라틴…) + **번역 엔진** 선택(로컬 LMStudio / OpenAI)
4. **전사 시작하기** → 실시간 전사 → 완료 후 자동 번역
5. 결과물이 **`~/Downloads` 폴더에 자동 저장**되고, UI 다운로드 버튼으로도 받을 수 있음

---

## 2) CLI

### 고정밀 전사
```bash
./run.sh transcribe audio.mp3 --model large-v3-turbo --prompt "입시 컨설팅 대화"
```

### 배치 처리
**폴더째 전사** — 폴더 경로를 주면 하위 폴더까지 재귀 스캔해 지원 미디어를 모두 전사합니다
(마운트된 원격 공유 폴더도 동일):
```bash
./run.sh batch /Volumes/shared/회의녹음 --prompt "중고등 입시 컨설팅"
```

**목록 파일 방식** — 한 줄에 링크/경로 하나씩 담은 `list.txt`를 만들고 (폴더 경로를 적으면
그 폴더의 파일들로 자동 확장):
```text
https://drive.google.com/file/d/LINK1/view
/path/to/audio2.mp3
/path/to/folder
```
실행:
```bash
./run.sh batch list.txt --prompt "중고등 입시 컨설팅"
```

**이어하기(자동)** — 배치 진행상황이 파일 단위로 `data/batch_state/`에 기록되어,
중간에 끊겨도 같은 명령을 다시 실행하면 완료된 파일은 자동으로 건너뜁니다
(실패한 파일은 재시도, 원본이 변경된 파일은 다시 전사). 처음부터 다시 돌리려면:
```bash
./run.sh batch /path/to/folder --fresh
```
웹 UI에서도 "로컬 폴더 경로" 입력과 Google Drive 폴더 배치에 동일한 이어하기가 적용됩니다.

### 화자 분리 (Speaker Diarization)
pyannote.audio 기반으로 Windows/macOS 모두 동작합니다. **최초 1회 HuggingFace 토큰 설정 필요** (아래 "화자 분리 설정" 참고).

```bash
# 단일 파일
./run.sh transcribe audio.mp3 --diarize

# 배치(폴더) 전체에 화자 분리 적용, 화자 수를 알면 지정 가능
./run.sh batch /path/to/folder --diarize --num-speakers 2
```
파일별로 `..._diarized.txt`(화자별 대화록), `..._diarized.json`이 추가 생성됩니다.
웹 UI에서는 "화자 분리" 체크박스로 켭니다.

배치 이어하기는 **옵션까지 기억**합니다: 화자 분리 없이 완료한 파일은 화자 분리를 켜고
다시 돌리면 재처리 대상이 되고, 같은 옵션으로 완료된 파일만 건너뜁니다.

### 요약
LMStudio를 열고 1234 포트로 'Local Server'를 켠 뒤:
```bash
./run.sh transcribe audio.mp3 --diarize --summary
```
`..._summary.md`가 생성됩니다.

### GPU 가속 (NVIDIA, Windows/Linux)
```bash
./run.sh batch list.txt --device cuda --compute_type float16
```

Windows는 `run.bat`을 더블클릭하거나 `run.bat transcribe "경로"` 형태로 사용하세요.

---

## 웹 앱 동작 방식

- **작업 큐**: 잡은 한 번에 하나씩 순차 실행됩니다. 여러 개를 제출하면 대기열에 쌓이고
  "앞에 N개 작업" 상태가 표시됩니다 (Whisper 모델 중복 로드 방지).
- **작업 기록**: 잡 상태가 `data/jobs/`에 저장되어 서버를 재시작해도 기록·다운로드가
  유지됩니다. 새로고침하면 진행 중인 작업에 자동으로 다시 연결됩니다.
- **품질 자동 검사**: 배치가 끝나면 반복 환각·언어 감지 이상을 자동 스캔해 로그에
  표시하고, 작업 기록의 "이상 파일 재전사" 버튼으로 해당 파일만 다시 돌릴 수 있습니다.
  CLI에서는 `python scripts/find_hallucinations.py [--reset]`.
- **자동 정리**: 업로드 임시파일 7일, 결과물 30일 경과분은 자동 삭제됩니다
  (`UPLOAD_RETENTION_DAYS`, `OUTPUT_RETENTION_DAYS`로 조정).
- **웹 서버 실행**: `./run.sh web` (Windows: `run.bat web`). waitress WSGI 서버로
  서빙되며, 미설치 시 Flask 개발 서버로 동작합니다.

---

## 화자 분리 설정 (최초 1회, 무료)

1. https://huggingface.co/join 에서 계정 생성 (이미 있으면 로그인)
2. 아래 **두 모델 페이지 모두**에서 이름·소속을 입력하고 약관 동의 (즉시 자동 승인):
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0
3. https://huggingface.co/settings/tokens → "New token" → Type: **Read** → 생성된 `hf_...` 복사
4. 프로젝트 루트 `.env`에 추가:
   ```ini
   HF_TOKEN=hf_여기에붙여넣기
   ```
5. 앱 재시작. 첫 실행 시 모델(약 30MB)이 자동 다운로드되며 이후 오프라인 동작.

Windows PC에서도 같은 토큰을 `.env`에 넣으면 됩니다. GPU(CUDA) 또는 Apple Silicon(MPS)이
있으면 자동 가속되고, `DIARIZE_DEVICE=cpu` 환경변수로 강제 지정할 수 있습니다.

---

## 번역 설정

번역은 OpenAI 호환 챗 API를 사용하며 **로컬(LMStudio)** 과 **클라우드(OpenAI)** 백엔드를 전환할 수 있습니다. 프로젝트 루트에 `.env`를 두면 자동 로드됩니다 (`.env.example` 참고).

```ini
# 로컬 LMStudio 백엔드 (무료·오프라인)
LMSTUDIO_MODEL=qwen/qwen3.6-27b
LMSTUDIO_BASE_URL=http://localhost:1234/v1

# OpenAI 클라우드 백엔드 (선택)
# OPENAI_API_KEY=sk-...
# OPENAI_TRANSLATE_MODEL=gpt-4o

# 완료 시 ~/Downloads 자동 저장 끄기
# SAVE_TO_DOWNLOADS=0
```

- **로컬**: LMStudio에서 모델(예: Qwen3.6-27B)을 로드하고 1234 포트 서버를 켠 뒤, UI에서 "로컬 LMStudio" 선택. CJK(한·일·중) 번역 품질이 우수합니다.
- **클라우드**: `OPENAI_API_KEY`를 넣고 UI에서 "OpenAI" 선택.

> 번역은 자막 큐 단위로 호출하며 앞뒤 큐를 문맥으로 제공해, **출력이 입력과 1:1로 정렬**됩니다(줄 밀림·유실 없음). 큐가 많은 긴 영상은 그만큼 호출이 늘어 시간이 더 걸립니다.

---

## 출력물

- 저장 위치: 완료 시 **`~/Downloads`** (작업 사본은 `data/outputs/`에도 보관)
- 파일명: 원본 파일명 기준 (예: `회의록.srt`, `회의록.en.srt`)
- 종류:
  - 원본: `.txt` / `.srt` / `.json`
  - 번역(언어별): `.{lang}.srt` / `.{lang}.dual.srt`(원문+번역 이중자막) / `.{lang}.txt`

---

## 프로젝트 구조

| 파일 | 역할 |
|---|---|
| `web_app.py` | Flask 웹 서버 — 업로드·전사·번역·자동 저장·다운로드 라우트 |
| `templates/index.html` | 웹 UI (글래스모피즘) |
| `main.py` | CLI 진입 로직 |
| `stt_engine.py` | faster-whisper 엔진 래퍼 |
| `output_utils.py` | 자막 분할(`split_into_subtitle_cues`) · SRT/이중자막/TXT writer · 다운로드 경로 |
| `translator.py` | 슬라이딩 문맥 1:1 번역기 (LMStudio/OpenAI 백엔드 전환) |
| `gdrive_utils.py` | Google Drive 다운로드 |
| `diarizer.py` / `summarizer.py` | 화자 분리 / LMStudio 요약 |
| `compare_translations.py` | (개발용) 번역 모델 비교 하니스 |
| `run.sh` / `run.bat` | 실행 진입점 |

---

## 라이선스

개인/연구용 프로젝트입니다.
