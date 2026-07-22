import os
import re
import sys
import json
import time
import uuid
import shutil
import threading
import unicodedata
import functools
from queue import Queue
from flask import Flask, request, jsonify, send_file, Response, render_template
from werkzeug.utils import secure_filename

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'data', 'uploads')
app.config['OUTPUT_FOLDER'] = os.path.join(os.path.dirname(__file__), 'data', 'outputs')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

jobs = {}
folder_sessions = {}

# One worker thread runs jobs sequentially so two batches can never load two
# Whisper models at once; extra submissions wait in line.
job_queue = Queue()
_worker_state = {'current': None}

# Keep at most this many segment events per job in the replayable history
# (other event types are never dropped).
EVENT_SEGMENT_CAP = 300

# Event types worth flushing the persistent job record for.
PERSIST_EVENT_TYPES = {
    'status', 'batch', 'file', 'file_done', 'file_error', 'file_skipped',
    'done', 'error', 'qa', 'diarized',
}


def persist_job(job_id):
    job = jobs.get(job_id)
    if job is not None:
        try:
            job_store.save_job(job_id, job)
        except OSError:
            pass


class JobCancelled(Exception):
    """Raised inside a running job when the user requested cancellation."""


def _check_cancelled(job_id):
    if jobs.get(job_id, {}).get('cancel_requested'):
        raise JobCancelled()


def _job_worker():
    while True:
        job_id, fn = job_queue.get()
        if jobs.get(job_id, {}).get('status') == 'cancelled':
            job_queue.task_done()
            continue
        _worker_state['current'] = job_id
        try:
            fn()
        except JobCancelled:
            job = jobs.get(job_id)
            if job is not None:
                job['status'] = 'cancelled'
                push_event(job_id, 'done', {'message': '작업이 취소되었습니다.', 'cancelled': True})
        except Exception as e:
            job = jobs.get(job_id)
            if job is not None:
                job['status'] = 'error'
                job['error'] = str(e)
                push_event(job_id, 'error', {'message': str(e)})
        finally:
            _worker_state['current'] = None
            job_queue.task_done()


def enqueue_job(job_id, fn):
    waiting = job_queue.qsize() + (1 if _worker_state['current'] else 0)
    if waiting:
        push_event(job_id, 'status', {
            'status': 'queued',
            'message': f'대기열에 추가됨 — 앞에 {waiting}개 작업이 있습니다.',
        })
    persist_job(job_id)
    job_queue.put((job_id, fn))


threading.Thread(target=_job_worker, daemon=True).start()


def cleanup_stale_data():
    """Drop old uploads/outputs and dead folder sessions."""
    now = time.time()
    upload_ttl = float(os.getenv('UPLOAD_RETENTION_DAYS', '7')) * 86400
    output_ttl = float(os.getenv('OUTPUT_RETENTION_DAYS', '30')) * 86400

    for entry in os.scandir(app.config['UPLOAD_FOLDER']):
        try:
            if now - entry.stat().st_mtime > upload_ttl:
                if entry.is_dir():
                    shutil.rmtree(entry.path, ignore_errors=True)
                else:
                    os.remove(entry.path)
        except OSError:
            continue

    for entry in os.scandir(app.config['OUTPUT_FOLDER']):
        try:
            if entry.is_dir() and now - entry.stat().st_mtime > output_ttl:
                shutil.rmtree(entry.path, ignore_errors=True)
                job_store.mark_expired(entry.name)
        except OSError:
            continue

    for session_id in [
        sid for sid, session in folder_sessions.items()
        if now - session.get('created_at', 0) > 86400
    ]:
        folder_sessions.pop(session_id, None)


def _cleanup_loop():
    while True:
        try:
            cleanup_stale_data()
        except Exception:
            pass
        time.sleep(6 * 3600)

from media_scan import (
    ALLOWED_EXTENSIONS,
    allowed_file,
    collect_supported_files,
    list_downloaded_files,
    looks_like_supported_media,
    media_duration_seconds,
)
from batch_state import CompletionIndex
import job_store
from qa_checks import qa_report

job_store.mark_interrupted_jobs()
threading.Thread(target=_cleanup_loop, daemon=True).start()


def push_event(job_id, event_type, data):
    """Append to the job's replayable event history (SSE reads by cursor)."""
    job = jobs.get(job_id)
    if job is None:
        return
    events = job.setdefault('events', [])
    events.append({'type': event_type, 'data': data})
    if event_type == 'segment':
        job['_segment_events'] = job.get('_segment_events', 0) + 1
        if job['_segment_events'] > EVENT_SEGMENT_CAP:
            for i, event in enumerate(events):
                if event['type'] == 'segment':
                    del events[i]
                    job['_segment_events'] -= 1
                    break
    if event_type in PERSIST_EVENT_TYPES:
        persist_job(job_id)


def format_srt_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def parse_subtitle_opts(form):
    from output_utils import SUBTITLE_MAX_CHARS, SUBTITLE_MIN_CHARS

    def to_int(name, default):
        try:
            return int((form.get(name) or '').strip())
        except (TypeError, ValueError):
            return default

    max_chars = to_int('subtitle_max_chars', SUBTITLE_MAX_CHARS)
    min_chars = to_int('subtitle_min_chars', SUBTITLE_MIN_CHARS)
    max_chars = max(10, min(max_chars, 200))
    min_chars = max(0, min(min_chars, max_chars))
    return {'max_chars': max_chars, 'min_chars': min_chars}


def parse_diarize_opts(form):
    enabled = (form.get('diarize') or '').strip().lower() not in ('', '0', 'false', 'off')
    num_raw = (form.get('num_speakers') or '').strip()
    try:
        num_speakers = int(num_raw) if num_raw else None
    except ValueError:
        num_speakers = None
    if num_speakers is not None:
        num_speakers = max(1, min(num_speakers, 20))
    return {'enabled': enabled, 'num_speakers': num_speakers}


def parse_summary_opts(form):
    enabled = (form.get('summary') or '').strip().lower() not in ('', '0', 'false', 'off')
    backend = (form.get('summary_backend') or 'openai').strip().lower()
    if backend not in ('openai', 'lmstudio'):
        backend = 'openai'
    return {'enabled': enabled, 'backend': backend}


def batch_options_for(diarize_opts, summary_opts=None):
    """Resume-index options: completed entries only count when processed
    with the same options."""
    options = {}
    if diarize_opts and diarize_opts.get('enabled'):
        options['diarize'] = True
        if diarize_opts.get('num_speakers'):
            options['num_speakers'] = diarize_opts['num_speakers']
    if summary_opts and summary_opts.get('enabled'):
        options['summary'] = True
    return options or None


def run_summary_for_file(job_id, summarizer, segments, base_path, filename, file_index=None, file_total=None):
    """Generate <base>.summary.md; failures warn but never fail the file."""
    prefix = f'[{file_index}/{file_total}] ' if file_index and file_total else ''
    push_event(job_id, 'status', {'status': 'summarizing', 'message': f'{prefix}{filename} AI 요약 생성 중...'})
    summary_md = summarizer.summarize_segments(segments, filename=filename)
    path = base_path + '.summary.md'
    with open(path, 'w', encoding='utf-8') as f:
        f.write(summary_md)
    push_event(job_id, 'summary', {
        'filename': filename,
        'file_index': file_index,
        'total_files': file_total,
    })
    return path


def write_diarized_files(base_path, diarized_results):
    from output_utils import save_as_diarized_text

    txt_path = base_path + '_diarized.txt'
    save_as_diarized_text(diarized_results, txt_path)
    json_path = base_path + '_diarized.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(diarized_results, f, ensure_ascii=False, indent=2)
    return {'diarized_txt': txt_path, 'diarized_json': json_path}


def run_diarization_for_file(job_id, diarizer, audio_path, segments, filename, file_index=None, file_total=None):
    from diarizer import align_words_with_speakers

    prefix = f'[{file_index}/{file_total}] ' if file_index and file_total else ''
    push_event(job_id, 'status', {'status': 'diarizing', 'message': f'{prefix}{filename} 화자 분리 중...'})
    num_speakers = (jobs.get(job_id, {}).get('diarize') or {}).get('num_speakers')
    speaker_segments = diarizer.run_diarization(audio_path, num_speakers=num_speakers)
    diarized = align_words_with_speakers(segments, speaker_segments)
    speakers = sorted({seg['speaker'] for seg in diarized})
    push_event(job_id, 'diarized', {
        'filename': filename,
        'file_index': file_index,
        'total_files': file_total,
        'speakers': len(speakers),
    })
    return diarized


def parse_translation_opts(form):
    """Read translation settings from the request form.

    Languages come from preset checkboxes (`translate_langs`, possibly multiple)
    and a free-form comma/space separated field (`translate_langs_custom`).
    """
    from translator import resolve_language

    backend = (form.get('translate_backend') or 'openai').strip()
    if backend not in ('openai', 'lmstudio'):
        backend = 'openai'

    raw_values = []
    if hasattr(form, 'getlist'):
        raw_values.extend(form.getlist('translate_langs'))
    else:
        single = form.get('translate_langs')
        if single:
            raw_values.append(single)

    custom = form.get('translate_langs_custom') or ''
    raw_values.extend(re.split(r'[,\s]+', custom))

    langs = []
    seen = set()
    for raw in raw_values:
        resolved = resolve_language(raw)
        if not resolved:
            continue
        code, name = resolved
        if code in seen:
            continue
        seen.add(code)
        langs.append({'code': code, 'name': name})

    return {'enabled': bool(langs), 'backend': backend, 'langs': langs}


def run_translations(job_id, segments, base_path, file_index=None):
    """Translate the job's subtitle cues into each requested language.

    Writes <base>.<code>.srt (translated), <base>.<code>.dual.srt (bilingual),
    and <base>.<code>.txt for every target language. Returns a dict keyed by
    language code, or None when translation is disabled. Failures are reported
    per-language via SSE and never abort the transcription job.
    """
    tr = jobs.get(job_id, {}).get('translation') or {}
    langs = tr.get('langs') or []
    if not langs:
        return None

    from output_utils import (
        split_into_subtitle_cues,
        write_cue_srt,
        write_bilingual_srt,
        write_cue_txt,
        SUBTITLE_MAX_CHARS,
        SUBTITLE_MIN_CHARS,
    )
    from translator import SubtitleTranslator

    sub = jobs.get(job_id, {}).get('subtitle') or {}
    max_chars = sub.get('max_chars') or SUBTITLE_MAX_CHARS
    min_chars = sub.get('min_chars')
    if min_chars is None:
        min_chars = SUBTITLE_MIN_CHARS
    cues = split_into_subtitle_cues(segments, max_chars=max_chars, min_chars=min_chars)

    try:
        translator = SubtitleTranslator(backend=tr.get('backend', 'openai'))
    except Exception as e:
        push_event(job_id, 'translate_error', {'message': str(e), 'file_index': file_index})
        return None

    results = {}
    for lang in langs:
        code, name = lang['code'], lang['name']
        push_event(job_id, 'translate_status', {
            'message': f'번역 중: {name}...',
            'language': name,
            'code': code,
            'file_index': file_index,
        })
        try:
            translated = translator.translate_cues(cues, name)
        except Exception as e:
            push_event(job_id, 'translate_error', {
                'message': f'{name} 번역 실패: {e}',
                'language': name,
                'code': code,
                'file_index': file_index,
            })
            continue

        srt_path = f"{base_path}.{code}.srt"
        dual_path = f"{base_path}.{code}.dual.srt"
        txt_path = f"{base_path}.{code}.txt"
        write_cue_srt(translated, srt_path)
        write_bilingual_srt(cues, translated, dual_path)
        write_cue_txt(translated, txt_path)
        results[code] = {'name': name, 'srt': srt_path, 'dual': dual_path, 'txt': txt_path}

    return results or None


def translation_download_urls(job_id, translations, file_index=None):
    """Build download URLs for a translations dict produced by run_translations."""
    urls = {}
    for code, entry in (translations or {}).items():
        if file_index is None:
            base = f"/api/download/{job_id}/translation/{code}"
        else:
            base = f"/api/download/{job_id}/file/{file_index}/translation/{code}"
        urls[code] = {
            'name': entry['name'],
            'srt': f"{base}/srt",
            'dual': f"{base}/dual",
            'txt': f"{base}/txt",
        }
    return urls


def _safe_stem(stem):
    # Unicode-safe: keeps Korean etc., strips only chars Windows/macOS forbid
    # in filenames. NFC so names match across both platforms.
    stem = unicodedata.normalize("NFC", (stem or "transcript").strip())
    stem = re.sub(r"[\\/:*?\"<>|\x00-\x1f]+", "_", stem)
    stem = stem.strip(" .")
    return stem or "transcript"


def _unique_path(directory, filename):
    """Return a path in directory that doesn't clobber an existing file."""
    base, ext = os.path.splitext(filename)
    candidate = os.path.join(directory, filename)
    i = 1
    while os.path.exists(candidate):
        candidate = os.path.join(directory, f"{base} ({i}){ext}")
        i += 1
    return candidate


def downloads_subfolder_for(media_path):
    """Folder name for auto-organizing a local file's Download copies.

    `.../컨설팅 영상_2025/9월_고1/x.mp4` → `2025년_9월_고1`: the file's
    immediate folder name, prefixed with a year found in an ancestor folder
    name (unless the name already contains one).
    """
    parent = os.path.dirname(os.path.realpath(media_path))
    folder_name = unicodedata.normalize('NFC', os.path.basename(parent))
    if not folder_name:
        return None

    year = None
    probe = parent
    for _ in range(4):
        match = re.search(r'(20\d{2})', os.path.basename(probe))
        if match:
            year = match.group(1)
            break
        upper = os.path.dirname(probe)
        if upper == probe:
            break
        probe = upper

    if year and year not in folder_name:
        return f'{year}년_{folder_name}'
    return folder_name


def export_to_downloads(outputs, translations=None, stem=None, subfolder=None):
    """Copy a job's output files into the user's Downloads folder.

    Names files by the original media stem (not the job UUID) so they're easy to
    find; local-folder batches land in a per-source subfolder (e.g.
    2025년_9월_고1). Returns the list of copied paths. Disabled by
    SAVE_TO_DOWNLOADS=0.
    """
    if os.getenv("SAVE_TO_DOWNLOADS", "1") not in ("1", "true", "True"):
        return []

    from output_utils import get_downloads_path

    downloads = get_downloads_path()
    if not os.path.isdir(downloads):
        return []
    if subfolder:
        downloads = os.path.join(downloads, _safe_stem(subfolder))
        os.makedirs(downloads, exist_ok=True)

    stem = _safe_stem(stem or (outputs or {}).get("stem"))
    copied = []

    for fmt, ext in (
        ("txt", "txt"), ("srt", "srt"), ("json", "json"),
        ("diarized_txt", "diarized.txt"), ("diarized_json", "diarized.json"),
        ("summary", "summary.md"),
    ):
        src = (outputs or {}).get(fmt)
        if src and os.path.exists(src):
            dst = _unique_path(downloads, f"{stem}.{ext}")
            shutil.copy2(src, dst)
            copied.append(dst)

    for code, entry in (translations or {}).items():
        for kind, ext in (("srt", "srt"), ("dual", "dual.srt"), ("txt", "txt")):
            src = entry.get(kind)
            if src and os.path.exists(src):
                dst = _unique_path(downloads, f"{stem}.{code}.{ext}")
                shutil.copy2(src, dst)
                copied.append(dst)

    return copied


def write_transcript_files(base_path, segments, json_payload=None, max_chars=None, min_chars=None):
    from output_utils import split_into_subtitle_cues, SUBTITLE_MAX_CHARS, SUBTITLE_MIN_CHARS

    if max_chars is None:
        max_chars = SUBTITLE_MAX_CHARS
    if min_chars is None:
        min_chars = SUBTITLE_MIN_CHARS

    txt_path = base_path + '.txt'
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(seg['text'] for seg in segments))

    srt_path = base_path + '.srt'
    cues = split_into_subtitle_cues(segments, max_chars=max_chars, min_chars=min_chars)
    with open(srt_path, 'w', encoding='utf-8') as f:
        for i, cue in enumerate(cues, 1):
            f.write(f"{i}\n{format_srt_time(cue['start'])} --> {format_srt_time(cue['end'])}\n{cue['text']}\n\n")

    json_path = base_path + '.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_payload if json_payload is not None else segments, f, ensure_ascii=False, indent=2)

    return {'txt': txt_path, 'srt': srt_path, 'json': json_path}


def save_outputs(job_id, segments, original_filename='transcript'):
    base = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
    stem = os.path.splitext(original_filename)[0] if original_filename else 'transcript'
    sub = jobs.get(job_id, {}).get('subtitle') or {}
    paths = write_transcript_files(base, segments, max_chars=sub.get('max_chars'), min_chars=sub.get('min_chars'))
    return {**paths, 'stem': stem}


def output_stem_for_file(filename, index):
    # Result files carry the source media's name (Korean intact); the
    # display name may include a subfolder (relpath), which _safe_stem
    # flattens to underscores, keeping in-batch stems unique.
    stem = _safe_stem(os.path.splitext(filename or '')[0])
    if stem == 'transcript':
        stem = f'transcript_{index:03d}'
    return stem


def save_file_outputs(job_id, file_result, index):
    output_dir = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
    os.makedirs(output_dir, exist_ok=True)

    stem = output_stem_for_file(file_result['filename'], index)
    base = os.path.join(output_dir, stem)
    payload = {
        'filename': file_result['filename'],
        'info': file_result.get('info'),
        'segments': file_result['segments'],
    }
    sub = jobs.get(job_id, {}).get('subtitle') or {}
    paths = write_transcript_files(
        base, file_result['segments'], json_payload=payload,
        max_chars=sub.get('max_chars'), min_chars=sub.get('min_chars'),
    )
    return {**paths, 'stem': stem}


def public_file_downloads(job_id, index, outputs=None):
    fmts = ['txt', 'srt', 'json']
    if outputs:
        fmts += [fmt for fmt in ('diarized_txt', 'diarized_json', 'summary') if outputs.get(fmt)]
    return {
        fmt: f"/api/download/{job_id}/file/{index}/{fmt}"
        for fmt in fmts
    }


def folder_item_to_record(item, index):
    file_id = getattr(item, "id", None)
    item_path = getattr(item, "path", None) or getattr(item, "local_path", None) or f"file_{index}"
    item_path = str(item_path)
    filename = os.path.basename(item_path) or item_path
    extension = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''

    return {
        'id': file_id,
        'index': index,
        'path': item_path,
        'filename': filename,
        'extension': extension,
        'supported': allowed_file(filename),
    }


def safe_download_name(filename, index):
    safe_name = secure_filename(os.path.basename(filename))
    if not safe_name:
        safe_name = f"gdrive_file_{index}"
    return f"{index:03d}_{safe_name}"


def download_selected_gdrive_files(job_id, session, selected_file_ids, state=None):
    selected = set(selected_file_ids)
    selected_records = [
        record for record in session['files']
        if record['id'] in selected
    ]
    if not selected_records:
        raise ValueError('선택된 Google Drive 파일이 없습니다.')

    from gdrive_utils import download_gdrive_file_by_id

    download_dir = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}_gdrive_selected")
    os.makedirs(download_dir, exist_ok=True)

    downloaded_paths = []
    display_names = {}
    file_keys = {}
    failures = []
    skipped_names = []

    for index, record in enumerate(selected_records, 1):
        resume_key = f"gdrive:{record['id']}"
        if state and state.is_done(resume_key):
            skipped_names.append(record['path'])
            continue

        output_path = os.path.join(download_dir, safe_download_name(record['filename'], index))
        downloaded = download_gdrive_file_by_id(record['id'], output_path)
        if downloaded:
            downloaded_paths.append(downloaded)
            display_names[downloaded] = record['path']
            file_keys[downloaded] = resume_key
        else:
            failures.append(record['path'])

    if not downloaded_paths and not skipped_names:
        raise ValueError(
            '선택한 Google Drive 파일 다운로드가 모두 실패했습니다.'
            + (f" 실패 파일: {', '.join(failures[:10])}" if failures else '')
        )

    return downloaded_paths, display_names, failures, file_keys, skipped_names


def load_engine(job_id, model_size):
    push_event(job_id, 'status', {'status': 'loading', 'message': f'모델 로딩 중: {model_size}...'})
    jobs[job_id]['status'] = 'loading'

    from stt_engine import STTEngine
    engine = STTEngine(model_size=model_size, device=None, compute_type=None)
    engine.load_model()
    return engine


def transcribe_with_engine(job_id, engine, audio_path, language, prompt, filename=None, file_index=None, file_total=None, progress_ctx=None):
    display_name = filename or os.path.basename(audio_path)
    if file_index and file_total:
        message = f'[{file_index}/{file_total}] {display_name} 전사 진행 중...'
    else:
        message = '전사 진행 중...'

    push_event(job_id, 'status', {'status': 'transcribing', 'message': message})
    jobs[job_id]['status'] = 'transcribing'

    # vad_filter skips silence (the main hallucination trigger) and
    # condition_on_previous_text=False keeps a repetition loop in one window
    # from contaminating the following windows.
    segments_gen, info = engine.model.transcribe(
        audio_path,
        beam_size=5,
        language=language if language else None,
        initial_prompt=prompt if prompt else None,
        condition_on_previous_text=False,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        word_timestamps=True,
    )

    lang_info = {
        'language': info.language,
        'language_probability': round(info.language_probability, 2),
        'duration': round(info.duration, 1),
        'filename': display_name,
    }
    if file_index and file_total:
        lang_info['file_index'] = file_index
        lang_info['total_files'] = file_total
    push_event(job_id, 'info', lang_info)

    # Real progress: audio-time processed over total audio time. A batch
    # passes a shared ctx; single files measure against their own duration.
    if progress_ctx is None:
        progress_ctx = {'total': info.duration or 0, 'done': 0.0}
    file_duration = info.duration or 0

    # Auto-detection can silently pick the wrong language when the clip opens
    # with silence/music, which makes Whisper output read like a translation.
    if not language and info.language_probability < 0.7:
        push_event(job_id, 'warning', {
            'message': (
                f'⚠ {display_name}: 언어 감지 신뢰도가 낮습니다 '
                f'({info.language}, {round(info.language_probability * 100)}%). '
                '결과가 이상하면 언어를 직접 지정해 다시 전사하세요.'
            ),
        })

    results = []
    for segment in segments_gen:
        _check_cancelled(job_id)
        words = []
        for word in getattr(segment, 'words', None) or []:
            words.append({
                'start': round(word.start, 2) if word.start is not None else None,
                'end': round(word.end, 2) if word.end is not None else None,
                'text': word.word.strip(),
            })

        seg_data = {
            'start': round(segment.start, 2),
            'end': round(segment.end, 2),
            'text': segment.text.strip(),
            'words': words,
        }
        if file_index and file_total:
            seg_data['filename'] = display_name
            seg_data['file_index'] = file_index
            seg_data['total_files'] = file_total
        results.append(seg_data)
        # Keep the live stream light: drop word-level detail from the UI event.
        push_event(job_id, 'segment', {k: v for k, v in seg_data.items() if k != 'words'})

        if progress_ctx.get('total'):
            position = progress_ctx['done'] + min(seg_data['end'], file_duration)
            percent = int(min(99, position / progress_ctx['total'] * 100))
            if percent != progress_ctx.get('last_percent'):
                progress_ctx['last_percent'] = percent
                push_event(job_id, 'progress', {
                    'percent': percent,
                    'file_index': file_index,
                    'total_files': file_total,
                })

    return {
        'filename': display_name,
        'segments': results,
        'info': lang_info,
    }


def run_transcription_job(job_id, audio_path, model_size, language, prompt, original_filename=''):
    try:
        engine = load_engine(job_id, model_size)
        result = transcribe_with_engine(job_id, engine, audio_path, language, prompt, original_filename)

        jobs[job_id]['segments'] = result['segments']
        jobs[job_id]['info'] = result['info']
        output_paths = save_outputs(job_id, result['segments'], original_filename)

        if (jobs[job_id].get('diarize') or {}).get('enabled'):
            from diarizer import create_diarizer
            push_event(job_id, 'status', {'status': 'loading', 'message': '화자 분리 모델 로딩 중...'})
            diarizer = create_diarizer()
            diarized = run_diarization_for_file(
                job_id, diarizer, audio_path, result['segments'], original_filename or 'transcript',
            )
            base = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
            output_paths.update(write_diarized_files(base, diarized))

        if (jobs[job_id].get('summary') or {}).get('enabled'):
            from report_summarizer import ReportSummarizer
            summarizer = ReportSummarizer(backend=jobs[job_id]['summary'].get('backend'))
            base = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
            try:
                output_paths['summary'] = run_summary_for_file(
                    job_id, summarizer, result['segments'], base, original_filename or 'transcript',
                )
            except JobCancelled:
                raise
            except Exception as e:
                push_event(job_id, 'warning', {'message': f'⚠ AI 요약 실패 — {e}'})

        jobs[job_id]['outputs'] = output_paths

        translations = run_translations(
            job_id, result['segments'], os.path.join(app.config['OUTPUT_FOLDER'], job_id)
        )
        if translations:
            jobs[job_id]['translations'] = translations
            push_event(job_id, 'translations', {
                'scope': 'single',
                'items': translation_download_urls(job_id, translations),
            })

        saved = export_to_downloads(output_paths, translations, stem=output_paths.get('stem'))
        if saved:
            push_event(job_id, 'saved', {'count': len(saved), 'directory': os.path.dirname(saved[0])})

        report = qa_report([result], requested_language=language)
        jobs[job_id]['qa'] = report
        if report['flagged']:
            push_event(job_id, 'qa', {
                'count': len(report['flagged']),
                'details': report['flagged'],
                'requeueable': False,
                'job_id': job_id,
            })

        jobs[job_id]['status'] = 'done'
        push_event(job_id, 'done', {'message': '전사 완료!', 'total_segments': len(result['segments'])})

    except JobCancelled:
        jobs[job_id]['status'] = 'cancelled'
        push_event(job_id, 'done', {'message': '작업이 취소되었습니다.', 'cancelled': True})
    except Exception as e:
        jobs[job_id]['status'] = 'error'
        jobs[job_id]['error'] = str(e)
        push_event(job_id, 'error', {'message': str(e)})


def run_batch_transcription_job(job_id, audio_paths, model_size, language, prompt, batch_name='gdrive_folder_batch', display_names=None, state=None, file_keys=None, pre_skipped=None):
    try:
        pre_skipped = pre_skipped or []
        if not audio_paths and not pre_skipped:
            raise ValueError('폴더 안에서 지원하는 오디오/비디오 파일을 찾지 못했습니다.')

        display_names = display_names or {}
        file_keys = file_keys or {}

        entries = []
        for audio_path in audio_paths:
            resume_key = file_keys.get(audio_path) or (CompletionIndex.file_key(audio_path) if state else None)
            already_done = bool(state and resume_key and state.is_done(resume_key))
            entries.append((audio_path, resume_key, already_done))

        total_files = len(pre_skipped) + len(entries)
        skipped_total = len(pre_skipped) + sum(1 for _, _, done in entries if done)

        push_event(job_id, 'batch', {
            'total_files': total_files,
            'skipped_files': skipped_total,
            'pending_files': total_files - skipped_total,
        })
        jobs[job_id]['status'] = 'batch'
        if skipped_total:
            push_event(job_id, 'status', {
                'status': 'batch',
                'message': f'이어하기: 전체 {total_files}개 중 {skipped_total}개는 이미 완료되어 건너뜁니다.',
            })

        # Time-based progress needs every pending file's duration up front;
        # if any is unreadable fall back to no progress events (total=0).
        pending_durations = {}
        total_duration = 0.0
        durations_known = True
        for audio_path, _, already_done in entries:
            if already_done:
                continue
            duration = media_duration_seconds(audio_path)
            if not duration:
                durations_known = False
                break
            pending_durations[audio_path] = duration
            total_duration += duration
        progress_ctx = {'total': total_duration if durations_known else 0, 'done': 0.0}

        # Local-folder batches auto-organize Download copies per source folder.
        organize_downloads = (jobs.get(job_id, {}).get('source') or {}).get('type') == 'local_folder'

        diarize_enabled = bool((jobs.get(job_id, {}).get('diarize') or {}).get('enabled'))
        diarizer = None
        if diarize_enabled and skipped_total < total_files:
            # Fail fast with clear instructions before transcribing anything.
            from diarizer import create_diarizer
            push_event(job_id, 'status', {'status': 'loading', 'message': '화자 분리 모델 로딩 중...'})
            diarizer = create_diarizer()

        summary_opts = jobs.get(job_id, {}).get('summary') or {}
        summarizer = None
        if summary_opts.get('enabled') and skipped_total < total_files:
            # Also fail fast (missing API key etc.) before transcribing.
            from report_summarizer import ReportSummarizer
            summarizer = ReportSummarizer(backend=summary_opts.get('backend'))

        engine = None  # Loaded lazily so a fully-completed batch skips the model load
        file_results = []
        failures = []
        index = 0

        for skipped_name in pre_skipped:
            index += 1
            push_event(job_id, 'file_skipped', {
                'filename': skipped_name,
                'file_index': index,
                'total_files': total_files,
            })

        cancelled = False
        for audio_path, resume_key, already_done in entries:
            if jobs.get(job_id, {}).get('cancel_requested'):
                cancelled = True
                break
            index += 1
            filename = display_names.get(audio_path, os.path.basename(audio_path))
            if already_done:
                push_event(job_id, 'file_skipped', {
                    'filename': filename,
                    'file_index': index,
                    'total_files': total_files,
                })
                continue

            push_event(job_id, 'file', {
                'filename': filename,
                'file_index': index,
                'total_files': total_files,
            })
            try:
                if engine is None:
                    engine = load_engine(job_id, model_size)
                result = transcribe_with_engine(
                    job_id,
                    engine,
                    audio_path,
                    language,
                    prompt,
                    filename,
                    file_index=index,
                    file_total=total_files,
                    progress_ctx=progress_ctx,
                )
                outputs = save_file_outputs(job_id, result, index)

                if diarizer is not None:
                    diarized = run_diarization_for_file(
                        job_id, diarizer, audio_path, result['segments'], filename,
                        file_index=index, file_total=total_files,
                    )
                    base = os.path.join(app.config['OUTPUT_FOLDER'], job_id, outputs['stem'])
                    outputs.update(write_diarized_files(base, diarized))

                if summarizer is not None:
                    try:
                        base = os.path.join(app.config['OUTPUT_FOLDER'], job_id, outputs['stem'])
                        outputs['summary'] = run_summary_for_file(
                            job_id, summarizer, result['segments'], base, filename,
                            file_index=index, file_total=total_files,
                        )
                    except JobCancelled:
                        raise
                    except Exception as e:
                        # Transcript is intact; summary alone failed.
                        push_event(job_id, 'warning', {'message': f'⚠ {filename}: AI 요약 실패 — {e}'})

                result['outputs'] = outputs
                result['download_urls'] = public_file_downloads(job_id, index, outputs)

                file_base = os.path.join(app.config['OUTPUT_FOLDER'], job_id, outputs['stem'])
                translations = run_translations(job_id, result['segments'], file_base, file_index=index)
                if translations:
                    result['translations'] = translations
                    result['translation_urls'] = translation_download_urls(job_id, translations, file_index=index)

                export_to_downloads(
                    outputs, translations, stem=outputs.get('stem'),
                    subfolder=downloads_subfolder_for(audio_path) if organize_downloads else None,
                )

                file_results.append(result)
                if state and resume_key:
                    state.mark_done(resume_key, outputs)
                push_event(job_id, 'file_done', {
                    'filename': filename,
                    'file_index': index,
                    'total_files': total_files,
                    'segments': len(result['segments']),
                    'downloads': result['download_urls'],
                    'translations': result.get('translation_urls'),
                })
            except JobCancelled:
                # No mark_failed: the in-flight file stays pending so the
                # next run of the same source picks it up.
                cancelled = True
                break
            except Exception as e:
                if state and resume_key:
                    state.mark_failed(resume_key, e)
                failure = {
                    'filename': filename,
                    'file_index': index,
                    'total_files': total_files,
                    'error': str(e),
                }
                failures.append(failure)
                push_event(job_id, 'file_error', failure)
            finally:
                progress_ctx['done'] += pending_durations.get(audio_path, 0)

        if cancelled:
            jobs[job_id]['batch_summary'] = {
                'total_files': total_files,
                'successful': len(file_results),
                'skipped': skipped_total,
                'failed': len(failures),
            }
            jobs[job_id]['files'] = file_results
            jobs[job_id]['status'] = 'cancelled'
            push_event(job_id, 'done', {
                'message': f'작업이 취소되었습니다. 이번에 완료한 {len(file_results)}개는 저장됐고, '
                           '같은 소스를 다시 제출하면 나머지만 이어서 전사합니다.',
                'cancelled': True,
            })
            return

        if not file_results and not skipped_total:
            raise ValueError('폴더 내 파일 전사가 모두 실패했습니다.')

        all_segments = [seg for file_result in file_results for seg in file_result['segments']]
        jobs[job_id]['segments'] = all_segments
        jobs[job_id]['files'] = file_results
        jobs[job_id]['failures'] = failures
        jobs[job_id]['outputs'] = {
            'type': 'batch',
            'directory': os.path.join(app.config['OUTPUT_FOLDER'], job_id),
            'files': [
                {
                    'filename': file_result['filename'],
                    'outputs': file_result['outputs'],
                    'download_urls': file_result['download_urls'],
                }
                for file_result in file_results
            ],
        }
        jobs[job_id]['batch_summary'] = {
            'total_files': total_files,
            'successful': len(file_results),
            'skipped': skipped_total,
            'failed': len(failures),
        }

        report = qa_report(file_results, requested_language=language)
        jobs[job_id]['qa'] = report
        if report['flagged']:
            push_event(job_id, 'qa', {
                'count': len(report['flagged']),
                'details': report['flagged'],
                'requeueable': (jobs[job_id].get('source') or {}).get('type') == 'local_folder',
                'job_id': job_id,
            })

        jobs[job_id]['status'] = 'done'

        message = f"배치 전사 완료: 성공 {len(file_results)}개"
        if skipped_total:
            message += f", 건너뜀(이미 완료) {skipped_total}개"
        if failures:
            message += f", 실패 {len(failures)}개"
        push_event(job_id, 'done', {
            'message': message,
            'total_files': total_files,
            'successful_files': len(file_results),
            'skipped_files': skipped_total,
            'failed_files': len(failures),
            'total_segments': len(all_segments),
        })

    except Exception as e:
        jobs[job_id]['status'] = 'error'
        jobs[job_id]['error'] = str(e)
        push_event(job_id, 'error', {'message': str(e)})


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/gdrive/list', methods=['POST'])
def list_gdrive_folder():
    data = request.get_json(silent=True) or request.form
    gdrive_url = data.get('gdrive_url', '').strip()

    if not gdrive_url:
        return jsonify({'error': 'Google Drive 폴더 URL을 입력해주세요.'}), 400

    from gdrive_utils import is_gdrive_folder_url, is_gdrive_url, list_gdrive_folder_files

    if not is_gdrive_url(gdrive_url):
        return jsonify({'error': '유효한 Google Drive URL이 아닙니다.'}), 400
    if not is_gdrive_folder_url(gdrive_url):
        return jsonify({'error': '파일 목록 조회는 Google Drive 폴더 링크에서만 지원합니다.'}), 400

    session_id = str(uuid.uuid4())
    output_dir = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_gdrive_list")

    try:
        normalized_url, items = list_gdrive_folder_files(gdrive_url, output_dir)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    files = [
        folder_item_to_record(item, index)
        for index, item in enumerate(items, 1)
    ]
    files = [record for record in files if record['id']]

    if not files:
        return jsonify({'error': '폴더 목록은 열렸지만 다운로드 가능한 파일 ID를 찾지 못했습니다.'}), 400

    folder_sessions[session_id] = {
        'id': session_id,
        'url': gdrive_url,
        'normalized_url': normalized_url,
        'output_dir': output_dir,
        'files': files,
        'created_at': time.time(),
    }

    return jsonify({
        'session_id': session_id,
        'normalized_url': normalized_url,
        'files': files,
        'total_files': len(files),
        'supported_files': len([record for record in files if record['supported']]),
    })


def prepare_local_folder_job(job_id, local_folder, fresh=False):
    """Validate and enqueue a local-folder batch. Returns an error message or
    None. Shared by /api/transcribe and /api/requeue."""
    local_folder = os.path.expanduser((local_folder or '').strip().strip('"').strip("'"))
    if not os.path.isdir(local_folder):
        return f'폴더를 찾을 수 없습니다: {local_folder} (서버에서 접근 가능한 경로여야 합니다.)'

    duplicate = _same_local_folder_active(local_folder)
    if duplicate and duplicate != job_id:
        return ('같은 폴더의 작업이 이미 진행 중이거나 대기 중입니다. '
                '작업 기록에서 기존 작업을 취소하거나 끝난 뒤 다시 제출하세요.')

    audio_paths = collect_supported_files([local_folder])
    if not audio_paths:
        return '폴더에서 지원하는 오디오/비디오 파일을 찾지 못했습니다.'

    job = jobs[job_id]
    params = job.get('params') or {}
    batch_state = CompletionIndex(options=batch_options_for(job.get('diarize'), job.get('summary')))
    if fresh:
        batch_state.reset_prefix(local_folder)

    file_keys = {path: CompletionIndex.file_key(path) for path in audio_paths}
    display_names = {path: os.path.relpath(path, local_folder) for path in audio_paths}
    job['filename'] = os.path.basename(os.path.normpath(local_folder)) or local_folder
    job['source'] = {'type': 'local_folder', 'path': local_folder}

    enqueue_job(job_id, functools.partial(
        run_batch_transcription_job,
        job_id, audio_paths, params.get('model', 'large-v3-turbo'),
        params.get('language'), params.get('prompt'),
        batch_name='local_folder_batch',
        display_names=display_names,
        state=batch_state,
        file_keys=file_keys,
    ))
    return None


@app.route('/api/transcribe', methods=['POST'])
def transcribe():
    model_size = request.form.get('model', 'large-v3-turbo')
    language = request.form.get('language', '') or None
    prompt = request.form.get('prompt', '') or None
    folder_session_id = request.form.get('gdrive_folder_session_id', '').strip()
    selected_file_ids_raw = request.form.get('selected_file_ids', '').strip()
    local_folder_raw = request.form.get('local_folder_path', '').strip().strip('"').strip("'")

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        'status': 'queued', 'segments': [], 'info': None, 'error': None,
        'created_at': time.time(), 'events': [],
        'params': {'model': model_size, 'language': language, 'prompt': prompt},
    }
    jobs[job_id]['subtitle'] = parse_subtitle_opts(request.form)
    jobs[job_id]['translation'] = parse_translation_opts(request.form)
    jobs[job_id]['diarize'] = parse_diarize_opts(request.form)
    jobs[job_id]['summary'] = parse_summary_opts(request.form)
    jobs[job_id]['params']['diarize'] = jobs[job_id]['diarize']
    jobs[job_id]['params']['summary'] = jobs[job_id]['summary']
    resume_options = batch_options_for(jobs[job_id]['diarize'], jobs[job_id]['summary'])

    if folder_session_id:
        session = folder_sessions.get(folder_session_id)
        if not session:
            return jsonify({'error': 'Google Drive 폴더 세션을 찾을 수 없습니다. 파일 목록을 다시 불러와 주세요.'}), 400

        try:
            selected_file_ids = json.loads(selected_file_ids_raw)
        except json.JSONDecodeError:
            return jsonify({'error': '선택 파일 목록이 올바르지 않습니다.'}), 400

        if not isinstance(selected_file_ids, list) or not selected_file_ids:
            return jsonify({'error': '전사할 파일을 하나 이상 선택해주세요.'}), 400

        jobs[job_id]['filename'] = 'gdrive_selected_files'
        batch_state = CompletionIndex(options=resume_options)

        def selected_gdrive_job():
            try:
                push_event(job_id, 'status', {'status': 'downloading', 'message': '선택한 Google Drive 파일 다운로드 중...'})
                jobs[job_id]['status'] = 'downloading'
                downloaded_paths, display_names, failures, file_keys, skipped_names = download_selected_gdrive_files(
                    job_id, session, selected_file_ids, state=batch_state,
                )
                audio_paths = collect_supported_files(downloaded_paths)
                if not audio_paths and not skipped_names:
                    downloaded_names = ', '.join(os.path.basename(path) or path for path in downloaded_paths[:10])
                    raise ValueError(
                        '선택한 파일을 다운로드했지만 지원하는 오디오/비디오 파일을 찾지 못했습니다.'
                        + (f' 다운로드 파일: {downloaded_names}' if downloaded_names else '')
                    )

                if failures:
                    push_event(job_id, 'status', {
                        'status': 'queued',
                        'message': f'다운로드 성공 {len(downloaded_paths)}개, 실패 {len(failures)}개. 성공 파일만 전사합니다.',
                    })
                else:
                    push_event(job_id, 'status', {
                        'status': 'queued',
                        'message': f'선택 파일 {len(audio_paths)}개를 전사합니다.'
                        + (f' (이미 완료된 {len(skipped_names)}개는 다운로드 없이 건너뜁니다.)' if skipped_names else ''),
                    })

                run_batch_transcription_job(
                    job_id,
                    audio_paths,
                    model_size,
                    language,
                    prompt,
                    batch_name='gdrive_selected_files',
                    display_names=display_names,
                    state=batch_state,
                    file_keys=file_keys,
                    pre_skipped=skipped_names,
                )
            except Exception as e:
                jobs[job_id]['status'] = 'error'
                jobs[job_id]['error'] = str(e)
                push_event(job_id, 'error', {'message': str(e)})

        jobs[job_id]['source'] = {'type': 'gdrive_selected', 'url': session.get('normalized_url') or session['url']}
        enqueue_job(job_id, selected_gdrive_job)

    elif local_folder_raw:
        error = prepare_local_folder_job(
            job_id,
            os.path.expanduser(local_folder_raw),
            fresh=bool(request.form.get('local_folder_fresh')),
        )
        if error:
            jobs.pop(job_id, None)
            return jsonify({'error': error}), 400

    elif 'file' in request.files and request.files['file'].filename:
        file = request.files['file']
        if not allowed_file(file.filename):
            return jsonify({'error': '지원하지 않는 파일 형식입니다.'}), 400

        original_filename = file.filename
        filename = secure_filename(f"{job_id}_{file.filename}")
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(audio_path)
        jobs[job_id]['filename'] = original_filename
        jobs[job_id]['source'] = {'type': 'upload'}

        enqueue_job(job_id, functools.partial(
            run_transcription_job, job_id, audio_path, model_size, language, prompt, original_filename,
        ))

    elif request.form.get('gdrive_url'):
        gdrive_url = request.form.get('gdrive_url')
        jobs[job_id]['filename'] = 'gdrive_file'
        jobs[job_id]['source'] = {'type': 'gdrive_url', 'url': gdrive_url}

        def gdrive_job():
            try:
                push_event(job_id, 'status', {'status': 'downloading', 'message': 'Google Drive에서 다운로드 중...'})
                jobs[job_id]['status'] = 'downloading'
                from gdrive_utils import (
                    download_folder_from_gdrive,
                    download_from_gdrive,
                    is_gdrive_folder_url,
                    is_gdrive_url,
                    normalize_gdrive_folder_url,
                )
                if not is_gdrive_url(gdrive_url):
                    raise ValueError('유효한 Google Drive URL이 아닙니다.')

                if is_gdrive_folder_url(gdrive_url):
                    download_dir = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}_gdrive_folder")
                    downloaded_paths = download_folder_from_gdrive(gdrive_url, download_dir)
                    downloaded_files = list_downloaded_files([download_dir])
                    audio_paths = collect_supported_files(downloaded_paths)
                    if not audio_paths:
                        audio_paths = collect_supported_files([download_dir])
                    if not audio_paths:
                        downloaded_names = ', '.join(
                            os.path.basename(path) or path
                            for path in downloaded_files[:10]
                        )
                        detail = f' 내려받은 파일: {downloaded_names}' if downloaded_names else ''
                        raise ValueError(
                            'Google Drive 폴더에서 지원하는 오디오/비디오 파일을 찾지 못했습니다.'
                            + detail
                        )
                    push_event(job_id, 'status', {
                        'status': 'queued',
                        'message': f'폴더에서 전사 대상 {len(audio_paths)}개를 찾았습니다. 다운로드 파일 {len(downloaded_files)}개.',
                    })

                    # Re-downloaded files get fresh mtimes, so key folder items
                    # by name+size instead of the default path|size|mtime.
                    batch_state = CompletionIndex(options=resume_options)
                    file_keys = {}
                    for path in audio_paths:
                        try:
                            size = os.path.getsize(path)
                        except OSError:
                            size = 'unknown'
                        file_keys[path] = f'gdrive-name:{os.path.normcase(os.path.basename(path))}|{size}'

                    run_batch_transcription_job(
                        job_id,
                        audio_paths,
                        model_size,
                        language,
                        prompt,
                        batch_name='gdrive_folder_batch',
                        state=batch_state,
                        file_keys=file_keys,
                    )
                    return

                temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}_gdrive.mp3")
                audio_path = download_from_gdrive(gdrive_url, temp_path)
                if not audio_path:
                    raise ValueError('Google Drive 다운로드 실패')
                run_transcription_job(job_id, audio_path, model_size, language, prompt, 'gdrive_file')
            except Exception as e:
                jobs[job_id]['status'] = 'error'
                jobs[job_id]['error'] = str(e)
                push_event(job_id, 'error', {'message': str(e)})

        enqueue_job(job_id, gdrive_job)
    else:
        jobs.pop(job_id, None)
        return jsonify({'error': '파일 또는 Google Drive URL을 제공해주세요.'}), 400

    persist_job(job_id)
    return jsonify({'job_id': job_id})


def _sse(event):
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@app.route('/api/stream/<job_id>')
def stream(job_id):
    if job_id not in jobs:
        # Job from a previous server run: replay a summary from the registry.
        record = job_store.load_job(job_id)
        if not record:
            return jsonify({'error': 'Job not found'}), 404

        def replay_stored():
            status = record.get('status')
            if status == 'done':
                yield _sse({'type': 'done', 'data': {
                    'message': '이전 세션에서 완료된 작업입니다. 작업 기록에서 결과를 내려받을 수 있습니다.',
                }})
            else:
                yield _sse({'type': 'error', 'data': {
                    'message': '서버 재시작으로 중단된 작업입니다. 같은 소스를 다시 제출하면 완료된 파일은 건너뛰고 이어서 처리됩니다.',
                }})

        return Response(replay_stored(), mimetype='text/event-stream',
                        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

    def generate():
        # Cursor over the job's event history: replays past events on
        # (re)connect, then follows live ones. Multiple viewers are fine.
        cursor = 0
        idle = 0.0
        while True:
            events = jobs[job_id].get('events') or []
            progressed = False
            while cursor < len(events):
                event = events[cursor]
                cursor += 1
                progressed = True
                yield _sse(event)
                if event['type'] in ('done', 'error'):
                    return
            if progressed:
                idle = 0.0
            else:
                time.sleep(0.4)
                idle += 0.4
                if idle >= 15:
                    idle = 0.0
                    yield _sse({'type': 'ping'})

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
    )


def _job_summary(job_id, job):
    """Uniform summary shape for both in-memory and persisted jobs."""
    qa = job.get('qa') or {}
    source = job.get('source') or {}
    return {
        'job_id': job_id,
        'status': job.get('status'),
        'filename': job.get('filename'),
        'created_at': job.get('created_at'),
        'error': job.get('error'),
        'batch_summary': job.get('batch_summary'),
        'qa_flagged': len(qa.get('flagged') or []),
        'requeueable': source.get('type') == 'local_folder',
        'expired': bool(job.get('expired')),
        'files': [
            {
                'filename': f.get('filename'),
                'download_urls': f.get('download_urls'),
            }
            for f in (job.get('files') or [])
        ],
    }


@app.route('/api/jobs')
def jobs_list():
    merged = {record['job_id']: record for record in job_store.list_jobs(limit=50)}
    merged.update({job_id: job for job_id, job in jobs.items()})  # memory is fresher
    summaries = [_job_summary(job_id, job) for job_id, job in merged.items()]
    summaries.sort(key=lambda s: s.get('created_at') or 0, reverse=True)
    active = _worker_state['current']
    return jsonify({
        'jobs': summaries[:30],
        'active_job_id': active,
        'queued_count': job_queue.qsize(),
    })


SEARCH_RESULT_LIMIT = 200


@app.route('/api/search')
def search_transcripts():
    query = unicodedata.normalize('NFC', (request.args.get('q') or '').strip()).lower()
    if len(query) < 2:
        return jsonify({'error': '검색어를 2자 이상 입력하세요.'}), 400

    import glob as _glob

    # Latest transcript per source filename (superseded outputs ignored).
    latest = {}
    for path in _glob.glob(os.path.join(app.config['OUTPUT_FOLDER'], '*', '*.json')):
        if path.endswith(('_diarized.json',)):
            continue
        try:
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict) or not data.get('filename'):
            continue
        name = unicodedata.normalize('NFC', data['filename'])
        mtime = os.path.getmtime(path)
        if name not in latest or mtime > latest[name][0]:
            latest[name] = (mtime, data)

    results = []
    truncated = False
    for name in sorted(latest):
        _, data = latest[name]
        for seg in data.get('segments') or []:
            text = unicodedata.normalize('NFC', seg.get('text') or '')
            if query in text.lower():
                results.append({
                    'filename': name,
                    'start': seg.get('start'),
                    'end': seg.get('end'),
                    'text': text,
                })
                if len(results) >= SEARCH_RESULT_LIMIT:
                    truncated = True
                    break
        if truncated:
            break

    return jsonify({
        'query': query,
        'results': results,
        'count': len(results),
        'files_scanned': len(latest),
        'truncated': truncated,
    })


@app.route('/api/cancel/<job_id>', methods=['POST'])
def cancel_job(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({'error': '작업을 찾을 수 없거나 이미 이전 세션의 작업입니다.'}), 404

    status = job.get('status')
    if status in ('done', 'error', 'cancelled'):
        return jsonify({'error': '이미 끝난 작업입니다.'}), 400

    if _worker_state['current'] == job_id:
        # Running: cooperative cancel — takes effect between segments/files.
        job['cancel_requested'] = True
        push_event(job_id, 'status', {'status': 'cancelling', 'message': '취소 요청됨 — 현재 구간까지만 처리하고 멈춥니다.'})
        return jsonify({'status': 'cancelling'})

    # Queued: mark cancelled; the worker skips it on dequeue.
    job['status'] = 'cancelled'
    push_event(job_id, 'done', {'message': '대기 중이던 작업을 취소했습니다.', 'cancelled': True})
    return jsonify({'status': 'cancelled'})


def _same_local_folder_active(folder):
    """Return the job_id of an active/queued job whose folder equals or
    overlaps (parent/child) the given folder — overlapping runs would fight
    over the same files."""
    target = os.path.normcase(os.path.realpath(folder)).replace(os.sep, '/')
    for job_id, job in jobs.items():
        if job.get('status') not in job_store.ACTIVE_STATUSES:
            continue
        source = job.get('source') or {}
        if source.get('type') != 'local_folder':
            continue
        active = os.path.normcase(os.path.realpath(source.get('path', ''))).replace(os.sep, '/')
        if active == target or active.startswith(target + '/') or target.startswith(active + '/'):
            return job_id
    return None


@app.route('/api/requeue/<job_id>', methods=['POST'])
def requeue(job_id):
    record = jobs.get(job_id) or job_store.load_job(job_id)
    if not record:
        return jsonify({'error': '작업을 찾을 수 없습니다.'}), 404

    source = record.get('source') or {}
    if source.get('type') != 'local_folder':
        return jsonify({'error': '로컬 폴더 배치만 재실행할 수 있습니다.'}), 400

    body = request.get_json(silent=True) or {}
    only_flagged = bool(body.get('only_flagged'))
    params = record.get('params') or {}
    diarize_opts = params.get('diarize') or {'enabled': False, 'num_speakers': None}

    if only_flagged:
        flagged_names = [f['filename'] for f in (record.get('qa') or {}).get('flagged') or []]
        if not flagged_names:
            return jsonify({'error': 'QA에서 이상이 감지된 파일이 없습니다.'}), 400
        state = CompletionIndex(options=batch_options_for(diarize_opts, params.get('summary')))
        reset_count = state.reset_files(flagged_names)
    else:
        reset_count = None

    new_job_id = str(uuid.uuid4())
    jobs[new_job_id] = {
        'status': 'queued', 'segments': [], 'info': None, 'error': None,
        'created_at': time.time(), 'events': [],
        'params': params,
        'subtitle': record.get('subtitle') or {},
        'translation': record.get('translation') or {'langs': [], 'backend': 'openai'},
        'diarize': diarize_opts,
        'summary': params.get('summary') or {'enabled': False, 'backend': 'openai'},
    }
    error = prepare_local_folder_job(new_job_id, source['path'])
    if error:
        jobs.pop(new_job_id, None)
        return jsonify({'error': error}), 400

    persist_job(new_job_id)
    return jsonify({'job_id': new_job_id, 'reset_files': reset_count})


@app.route('/api/download/<job_id>/<fmt>')
def download(job_id, fmt):
    job = jobs.get(job_id) or job_store.load_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    if job['status'] != 'done':
        return jsonify({'error': '아직 완료되지 않았습니다.'}), 400
    if fmt not in DOWNLOAD_FMT_EXTS:
        return jsonify({'error': '지원하지 않는 형식'}), 400
    file_path = job['outputs'].get(fmt)
    if not file_path or not os.path.exists(file_path):
        return jsonify({'error': '파일 없음'}), 404
    stem = job['outputs'].get('stem', 'transcript')
    return send_file(file_path, as_attachment=True, download_name=f"{stem}.{DOWNLOAD_FMT_EXTS[fmt]}")


@app.route('/api/download/<job_id>/file/<int:file_index>/<fmt>')
def download_batch_file(job_id, file_index, fmt):
    job = jobs.get(job_id) or job_store.load_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    if job['status'] != 'done':
        return jsonify({'error': '아직 완료되지 않았습니다.'}), 400
    if fmt not in DOWNLOAD_FMT_EXTS:
        return jsonify({'error': '지원하지 않는 형식'}), 400

    files = job.get('files') or []
    if file_index < 1 or file_index > len(files):
        return jsonify({'error': '파일 번호를 찾을 수 없습니다.'}), 404

    file_result = files[file_index - 1]
    outputs = file_result.get('outputs') or {}
    file_path = outputs.get(fmt)
    if not file_path or not os.path.exists(file_path):
        return jsonify({'error': '파일 없음'}), 404

    stem = outputs.get('stem') or output_stem_for_file(file_result['filename'], file_index)
    return send_file(file_path, as_attachment=True, download_name=f"{stem}.{DOWNLOAD_FMT_EXTS[fmt]}")


DOWNLOAD_FMT_EXTS = {
    'txt': 'txt',
    'srt': 'srt',
    'json': 'json',
    'diarized_txt': 'diarized.txt',
    'diarized_json': 'diarized.json',
    'summary': 'summary.md',
}

TRANSLATION_KINDS = {
    'srt': ('srt', 'srt'),
    'dual': ('dual', 'dual.srt'),
    'txt': ('txt', 'txt'),
}


def _send_translation(file_path, stem, code, kind, ext):
    if not file_path or not os.path.exists(file_path):
        return jsonify({'error': '파일 없음'}), 404
    return send_file(file_path, as_attachment=True, download_name=f"{stem}.{code}.{ext}")


@app.route('/api/download/<job_id>/translation/<code>/<kind>')
def download_translation(job_id, code, kind):
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    if kind not in TRANSLATION_KINDS:
        return jsonify({'error': '지원하지 않는 형식'}), 400
    job = jobs[job_id]
    if job.get('status') != 'done':
        return jsonify({'error': '아직 완료되지 않았습니다.'}), 400

    entry = (job.get('translations') or {}).get(code)
    if not entry:
        return jsonify({'error': '번역 파일 없음'}), 404

    path_key, ext = TRANSLATION_KINDS[kind]
    stem = (job.get('outputs') or {}).get('stem', 'transcript')
    return _send_translation(entry.get(path_key), stem, code, kind, ext)


@app.route('/api/download/<job_id>/file/<int:file_index>/translation/<code>/<kind>')
def download_batch_translation(job_id, file_index, code, kind):
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    if kind not in TRANSLATION_KINDS:
        return jsonify({'error': '지원하지 않는 형식'}), 400
    job = jobs[job_id]
    if job.get('status') != 'done':
        return jsonify({'error': '아직 완료되지 않았습니다.'}), 400

    files = job.get('files') or []
    if file_index < 1 or file_index > len(files):
        return jsonify({'error': '파일 번호를 찾을 수 없습니다.'}), 404

    file_result = files[file_index - 1]
    entry = (file_result.get('translations') or {}).get(code)
    if not entry:
        return jsonify({'error': '번역 파일 없음'}), 404

    path_key, ext = TRANSLATION_KINDS[kind]
    stem = (file_result.get('outputs') or {}).get('stem') or output_stem_for_file(file_result['filename'], file_index)
    return _send_translation(entry.get(path_key), stem, code, kind, ext)


@app.route('/api/status/<job_id>')
def status(job_id):
    if job_id not in jobs:
        return jsonify({'error': 'Not found'}), 404
    job = jobs[job_id]
    return jsonify({
        'status': job['status'],
        'segments_count': len(job.get('segments', [])),
        'info': job.get('info'),
        'outputs': job.get('outputs'),
        'error': job.get('error'),
    })


if __name__ == '__main__':
    print("\n[INFO] Python STT Pro - Web Interface")
    print("-" * 40)
    print("Running at: http://localhost:5000")
    print("-" * 40 + "\n")
    try:
        from waitress import serve
        # channel_timeout must exceed the SSE ping interval; threads sized so
        # a few live SSE viewers can't starve API requests.
        print("[INFO] Serving with waitress")
        serve(app, host='0.0.0.0', port=5000, threads=16, channel_timeout=120)
    except ImportError:
        print("[INFO] waitress not installed - falling back to Flask dev server")
        app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
