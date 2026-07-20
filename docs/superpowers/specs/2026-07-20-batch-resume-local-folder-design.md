# 로컬 폴더 배치 입력 + 이어하기(Resume) 설계

날짜: 2026-07-20
상태: 승인됨 (웹+CLI 모두 적용, 재실행 시 자동 스킵 방식, Windows/macOS 모두 지원)

## 목표

1. Google Drive가 아닌 **로컬/마운트된 공유 폴더 경로**를 웹 UI와 CLI에서 직접 입력해 폴더 안의
   지원 미디어 파일을 자동 스캔·배치 전사한다.
2. 배치 진행상황을 **파일 단위로 디스크에 기록**해서, 프로세스가 중간에 죽어도
   같은 소스를 다시 돌리면 **완료된 파일은 자동으로 건너뛰고** 나머지만 전사한다.
3. Windows / macOS 어느 쪽에서 실행해도 동일하게 동작한다.

## 선택한 방식

폴더(소스)별 manifest JSON 상태 파일 방식 (방식 A).

- 상태 파일 위치: `data/batch_state/<sha1(source_key) 앞 16자>.json`
- `source_key`:
  - 로컬 폴더: `dir:` + `os.path.normcase(os.path.realpath(폴더))`
  - GDrive 폴더 URL: `url:` + 정규화된 폴더 URL
  - CLI 목록 파일: `list:` + normcase(realpath(목록파일))
- 파일 식별 키(`file_key`):
  - 로컬 파일: 소스 폴더 기준 상대경로(구분자 `/`로 정규화, normcase) + `|크기|수정시각`
    → 원본이 변경되면 자동으로 재전사
  - GDrive 선택 파일: `gdrive:<파일 ID>` (재다운로드 시 mtime이 바뀌므로 ID 기반)
  - GDrive 폴더 일괄 다운로드: `이름|크기` (ID 매핑이 없는 경로)
- 파일 하나가 끝날 때마다 즉시 저장: 임시 파일에 쓴 뒤 `os.replace`로 원자적 교체
  (Windows/macOS 모두 지원). 어느 시점에 죽어도 손실은 "진행 중이던 파일 1개".
- 실패한 파일은 `failed`로 기록하되 재실행 시 **다시 시도**한다 (완료만 스킵).
- 상태 파일이 깨져 있으면 무시하고 새로 시작한다.

## 구성 요소

### 신규 모듈

- `batch_state.py` — `BatchState` 클래스 (`is_done` / `mark_done` / `mark_failed` /
  `reset`, 원자적 저장), `source_key_for_path` / `source_key_for_url` 헬퍼.
- `media_scan.py` — 기존 `web_app.py`의 미디어 판별·폴더 재귀 스캔 헬퍼
  (`allowed_file`, `looks_like_supported_media`, `list_downloaded_files`,
  `collect_supported_files`)를 공용 모듈로 이동. CLI와 웹이 공유.

### CLI (`main.py`)

- `batch` 명령 인자가 **폴더 경로**면 재귀 스캔해서 지원 미디어를 배치 전사.
  기존 목록 파일 방식도 유지하며, 목록 안의 줄이 폴더 경로면 파일로 확장.
- 완료 파일은 `건너뜀 (완료됨)`으로 표시하고 스킵. `--fresh` 플래그로 기록 무시 후 전체 재실행.

### 웹 (`web_app.py`, `templates/index.html`)

- 입력란 추가: "로컬 폴더 경로" (`local_folder_path`) + "완료 기록 무시" 체크박스
  (`local_folder_fresh`). 서버가 접근 가능한 경로(마운트된 원격 공유 폴더 포함)를 받는다.
  붙여넣은 경로의 앞뒤 따옴표 제거, `~` 확장 처리.
- `run_batch_transcription_job`에 `state`/`file_keys` 파라미터 추가. 시작 시
  "전체 N개 중 M개 완료됨 → K개 전사" 요약을 보내고, 스킵 파일마다 `file_skipped`
  SSE 이벤트 발행. 성공/실패 시마다 상태 기록.
- GDrive 배치에도 동일 적용: 선택 파일 배치는 **다운로드 전에** 완료 파일을 걸러
  재다운로드도 생략, 폴더 URL 배치는 다운로드 후 전사 단계에서 스킵.
- 진행률 표시는 전체 파일 목록 기준 인덱스를 유지 (스킵 포함).

## 크로스 플랫폼 규칙

- 모든 텍스트 파일 입출력에 `encoding="utf-8"` 명시 (Windows cp949 기본값 회피).
- 경로 비교·키 생성에 `os.path.normcase` + `os.path.realpath` 사용,
  키의 경로 구분자는 `/`로 통일.
- 원자적 쓰기는 `os.replace` (양 OS 지원). POSIX 전용 API 사용 금지.
- UNC 경로(`\\server\share`) 및 마운트 경로는 일반 경로와 동일하게 `os.walk`로 처리.

## 에러 처리

- 존재하지 않거나 폴더가 아닌 경로 → 웹은 400 + 한국어 안내, CLI는 에러 출력 후 종료.
- 폴더에 지원 미디어가 없으면 명확히 안내.
- 파일 개별 실패는 기존처럼 기록 후 다음 파일 진행. 전부 실패 시 잡 에러.

## 확장: 화자 분리 (2026-07-20 승인)

- 백엔드: pyannote.audio 3.1 (`diarizer.py`의 `PyannoteDiarizer` + `create_diarizer` 팩토리,
  NeMo는 설치된 경우에만 폴백). Windows/macOS 모두 동작. `HF_TOKEN` 필요(무료, 1회 설정).
- 오디오 디코딩은 faster-whisper의 PyAV 로더 재사용 → mp4/m4a 등 컨테이너 포맷 안전.
- 연결: 웹 단일/배치 잡 + CLI `batch --diarize [--num-speakers N]`. 배치당 diarizer 1회
  로드, 켜져 있으면 전사 시작 전에 생성해 토큰 문제를 조기에 실패시킴.
- 산출물: 파일별 `_diarized.txt`/`_diarized.json`, 다운로드 URL(`diarized_txt`/`diarized_json`),
  다운로드 폴더 내보내기 포함.
- **이어하기 상호작용**: manifest 항목에 처리 옵션(`{'diarize': true, 'num_speakers': N}`)을
  기록하고, is_done 판정 시 현재 옵션과 일치해야만 스킵. 옵션이 다르면 재처리.
- 한계(1단계 제외): 배치 내 파일 간 화자 라벨 매칭 없음(SPEAKER_00은 파일별 독립).

## 테스트

- `batch_state.py` / `media_scan.py` 단위 테스트 (`tests/`): 상태 저장·로드·스킵 판정,
  깨진 상태 파일 복구, 파일 변경 시 재전사 판정, 폴더 재귀 스캔.
- 웹/CLI는 임포트·인자 파싱 스모크 테스트 + 수동 확인.
