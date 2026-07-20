import os
import re
import sys
import json
import time
import uuid
import shutil
import threading
from queue import Queue, Empty
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
job_queues = {}
folder_sessions = {}

from media_scan import (
    ALLOWED_EXTENSIONS,
    allowed_file,
    collect_supported_files,
    list_downloaded_files,
    looks_like_supported_media,
)
from batch_state import BatchState, source_key_for_path, source_key_for_url


def push_event(job_id, event_type, data):
    if job_id in job_queues:
        job_queues[job_id].put({'type': event_type, 'data': data})


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


def batch_options_for(diarize_opts):
    """Resume-manifest options: completed entries only count when processed
    with the same options."""
    if not diarize_opts or not diarize_opts.get('enabled'):
        return None
    options = {'diarize': True}
    if diarize_opts.get('num_speakers'):
        options['num_speakers'] = diarize_opts['num_speakers']
    return options


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
    stem = (stem or "transcript").strip()
    stem = re.sub(r"[\\/:*?\"<>|]+", "_", stem)  # filename-unsafe chars
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


def export_to_downloads(outputs, translations=None, stem=None):
    """Copy a job's output files into the user's Downloads folder.

    Names files by the original media stem (not the job UUID) so they're easy to
    find. Returns the list of copied paths. Disabled by SAVE_TO_DOWNLOADS=0.
    """
    if os.getenv("SAVE_TO_DOWNLOADS", "1") not in ("1", "true", "True"):
        return []

    from output_utils import get_downloads_path

    downloads = get_downloads_path()
    if not os.path.isdir(downloads):
        return []

    stem = _safe_stem(stem or (outputs or {}).get("stem"))
    copied = []

    for fmt, ext in (
        ("txt", "txt"), ("srt", "srt"), ("json", "json"),
        ("diarized_txt", "diarized.txt"), ("diarized_json", "diarized.json"),
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
    stem = secure_filename(os.path.splitext(os.path.basename(filename))[0])
    return f"{index:03d}_{stem or 'transcript'}"


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
        fmts += [fmt for fmt in ('diarized_txt', 'diarized_json') if outputs.get(fmt)]
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


def transcribe_with_engine(job_id, engine, audio_path, language, prompt, filename=None, file_index=None, file_total=None):
    display_name = filename or os.path.basename(audio_path)
    if file_index and file_total:
        message = f'[{file_index}/{file_total}] {display_name} 전사 진행 중...'
    else:
        message = '전사 진행 중...'

    push_event(job_id, 'status', {'status': 'transcribing', 'message': message})
    jobs[job_id]['status'] = 'transcribing'

    segments_gen, info = engine.model.transcribe(
        audio_path,
        beam_size=5,
        language=language if language else None,
        initial_prompt=prompt if prompt else None,
        condition_on_previous_text=True,
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

    results = []
    for segment in segments_gen:
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

        jobs[job_id]['status'] = 'done'
        push_event(job_id, 'done', {'message': '전사 완료!', 'total_segments': len(result['segments'])})

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
            resume_key = file_keys.get(audio_path) or (BatchState.file_key(audio_path) if state else None)
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

        diarize_enabled = bool((jobs.get(job_id, {}).get('diarize') or {}).get('enabled'))
        diarizer = None
        if diarize_enabled and skipped_total < total_files:
            # Fail fast with clear instructions before transcribing anything.
            from diarizer import create_diarizer
            push_event(job_id, 'status', {'status': 'loading', 'message': '화자 분리 모델 로딩 중...'})
            diarizer = create_diarizer()

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

        for audio_path, resume_key, already_done in entries:
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
                )
                outputs = save_file_outputs(job_id, result, index)

                if diarizer is not None:
                    diarized = run_diarization_for_file(
                        job_id, diarizer, audio_path, result['segments'], filename,
                        file_index=index, file_total=total_files,
                    )
                    base = os.path.join(app.config['OUTPUT_FOLDER'], job_id, outputs['stem'])
                    outputs.update(write_diarized_files(base, diarized))

                result['outputs'] = outputs
                result['download_urls'] = public_file_downloads(job_id, index, outputs)

                file_base = os.path.join(app.config['OUTPUT_FOLDER'], job_id, outputs['stem'])
                translations = run_translations(job_id, result['segments'], file_base, file_index=index)
                if translations:
                    result['translations'] = translations
                    result['translation_urls'] = translation_download_urls(job_id, translations, file_index=index)

                export_to_downloads(outputs, translations, stem=outputs.get('stem'))

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


@app.route('/api/transcribe', methods=['POST'])
def transcribe():
    model_size = request.form.get('model', 'large-v3-turbo')
    language = request.form.get('language', '') or None
    prompt = request.form.get('prompt', '') or None
    folder_session_id = request.form.get('gdrive_folder_session_id', '').strip()
    selected_file_ids_raw = request.form.get('selected_file_ids', '').strip()
    local_folder_raw = request.form.get('local_folder_path', '').strip().strip('"').strip("'")

    job_id = str(uuid.uuid4())
    jobs[job_id] = {'status': 'queued', 'segments': [], 'info': None, 'error': None, 'created_at': time.time()}
    jobs[job_id]['subtitle'] = parse_subtitle_opts(request.form)
    jobs[job_id]['translation'] = parse_translation_opts(request.form)
    jobs[job_id]['diarize'] = parse_diarize_opts(request.form)
    resume_options = batch_options_for(jobs[job_id]['diarize'])
    job_queues[job_id] = Queue()

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
        batch_state = BatchState(source_key_for_url(session.get('normalized_url') or session['url']), options=resume_options)

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

        threading.Thread(target=selected_gdrive_job, daemon=True).start()

    elif local_folder_raw:
        local_folder = os.path.expanduser(local_folder_raw)
        if not os.path.isdir(local_folder):
            return jsonify({'error': f'폴더를 찾을 수 없습니다: {local_folder_raw} (서버에서 접근 가능한 경로여야 합니다.)'}), 400

        audio_paths = collect_supported_files([local_folder])
        if not audio_paths:
            return jsonify({'error': '폴더에서 지원하는 오디오/비디오 파일을 찾지 못했습니다.'}), 400

        batch_state = BatchState(source_key_for_path(local_folder), options=resume_options)
        if request.form.get('local_folder_fresh'):
            batch_state.reset()

        file_keys = {path: BatchState.file_key(path, base_dir=local_folder) for path in audio_paths}
        display_names = {path: os.path.relpath(path, local_folder) for path in audio_paths}
        jobs[job_id]['filename'] = os.path.basename(os.path.normpath(local_folder)) or local_folder

        thread = threading.Thread(
            target=run_batch_transcription_job,
            args=(job_id, audio_paths, model_size, language, prompt),
            kwargs={
                'batch_name': 'local_folder_batch',
                'display_names': display_names,
                'state': batch_state,
                'file_keys': file_keys,
            },
            daemon=True,
        )
        thread.start()

    elif 'file' in request.files and request.files['file'].filename:
        file = request.files['file']
        if not allowed_file(file.filename):
            return jsonify({'error': '지원하지 않는 파일 형식입니다.'}), 400

        original_filename = file.filename
        filename = secure_filename(f"{job_id}_{file.filename}")
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(audio_path)
        jobs[job_id]['filename'] = original_filename

        thread = threading.Thread(
            target=run_transcription_job,
            args=(job_id, audio_path, model_size, language, prompt, original_filename),
            daemon=True
        )
        thread.start()

    elif request.form.get('gdrive_url'):
        gdrive_url = request.form.get('gdrive_url')
        jobs[job_id]['filename'] = 'gdrive_file'

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
                    batch_state = BatchState(source_key_for_url(normalize_gdrive_folder_url(gdrive_url)), options=resume_options)
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

        threading.Thread(target=gdrive_job, daemon=True).start()
    else:
        return jsonify({'error': '파일 또는 Google Drive URL을 제공해주세요.'}), 400

    return jsonify({'job_id': job_id})


@app.route('/api/stream/<job_id>')
def stream(job_id):
    if job_id not in job_queues:
        return jsonify({'error': 'Job not found'}), 404

    def generate():
        q = job_queues[job_id]
        while True:
            try:
                event = q.get(timeout=30)
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                if event['type'] in ('done', 'error'):
                    break
            except Empty:
                yield f"data: {json.dumps({'type': 'ping'})}\n\n"

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
    )


@app.route('/api/download/<job_id>/<fmt>')
def download(job_id, fmt):
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    job = jobs[job_id]
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
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    job = jobs[job_id]
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
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
