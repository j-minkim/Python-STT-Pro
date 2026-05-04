import os
import sys
import json
import time
import uuid
import threading
from queue import Queue, Empty
from flask import Flask, request, jsonify, send_file, Response, render_template
from werkzeug.utils import secure_filename

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'data', 'uploads')
app.config['OUTPUT_FOLDER'] = os.path.join(os.path.dirname(__file__), 'data', 'outputs')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

jobs = {}
job_queues = {}

ALLOWED_EXTENSIONS = {
    'mp3', 'wav', 'mp4', 'm4a', 'ogg', 'flac', 'webm', 'aac', 'wma',
    'mov', 'm4v', 'mkv', 'avi',
}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def looks_like_supported_media(path):
    if allowed_file(path):
        return True
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return False

    try:
        with open(path, 'rb') as f:
            header = f.read(64)
    except OSError:
        return False

    if len(header) >= 12 and header[4:8] == b'ftyp':
        return True
    if header.startswith((b'ID3', b'OggS', b'fLaC')):
        return True
    if len(header) >= 12 and header.startswith(b'RIFF') and header[8:12] == b'WAVE':
        return True
    if header.startswith(b'\x1a\x45\xdf\xa3'):
        return True
    if len(header) >= 2 and header[0] == 0xFF and (header[1] & 0xE0) == 0xE0:
        return True
    return False


def list_downloaded_files(paths):
    files = []
    for path in paths or []:
        if not path:
            continue
        if os.path.isdir(path):
            for root, dirs, filenames in os.walk(path):
                dirs.sort()
                for filename in sorted(filenames):
                    files.append(os.path.join(root, filename))
        elif os.path.isfile(path):
            files.append(path)
    return files


def collect_supported_files(paths):
    supported = []
    for path in list_downloaded_files(paths):
        if looks_like_supported_media(path):
            supported.append(path)

    seen = set()
    unique_paths = []
    for path in supported:
        real_path = os.path.realpath(path)
        if real_path not in seen:
            seen.add(real_path)
            unique_paths.append(path)
    return unique_paths


def push_event(job_id, event_type, data):
    if job_id in job_queues:
        job_queues[job_id].put({'type': event_type, 'data': data})


def format_srt_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_transcript_files(base_path, segments, json_payload=None):
    txt_path = base_path + '.txt'
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(seg['text'] for seg in segments))

    srt_path = base_path + '.srt'
    with open(srt_path, 'w', encoding='utf-8') as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n{format_srt_time(seg['start'])} --> {format_srt_time(seg['end'])}\n{seg['text']}\n\n")

    json_path = base_path + '.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_payload if json_payload is not None else segments, f, ensure_ascii=False, indent=2)

    return {'txt': txt_path, 'srt': srt_path, 'json': json_path}


def save_outputs(job_id, segments, original_filename='transcript'):
    base = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
    stem = os.path.splitext(original_filename)[0] if original_filename else 'transcript'
    paths = write_transcript_files(base, segments)
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
    paths = write_transcript_files(base, file_result['segments'], json_payload=payload)
    return {**paths, 'stem': stem}


def public_file_downloads(job_id, index):
    return {
        fmt: f"/api/download/{job_id}/file/{index}/{fmt}"
        for fmt in ('txt', 'srt', 'json')
    }


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
        seg_data = {
            'start': round(segment.start, 2),
            'end': round(segment.end, 2),
            'text': segment.text.strip()
        }
        if file_index and file_total:
            seg_data['filename'] = display_name
            seg_data['file_index'] = file_index
            seg_data['total_files'] = file_total
        results.append(seg_data)
        push_event(job_id, 'segment', seg_data)

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
        jobs[job_id]['outputs'] = output_paths
        jobs[job_id]['status'] = 'done'
        push_event(job_id, 'done', {'message': '전사 완료!', 'total_segments': len(result['segments'])})

    except Exception as e:
        jobs[job_id]['status'] = 'error'
        jobs[job_id]['error'] = str(e)
        push_event(job_id, 'error', {'message': str(e)})


def run_batch_transcription_job(job_id, audio_paths, model_size, language, prompt, batch_name='gdrive_folder_batch'):
    try:
        if not audio_paths:
            raise ValueError('폴더 안에서 지원하는 오디오/비디오 파일을 찾지 못했습니다.')

        push_event(job_id, 'batch', {'total_files': len(audio_paths)})
        jobs[job_id]['status'] = 'batch'

        engine = load_engine(job_id, model_size)
        file_results = []
        failures = []
        total_files = len(audio_paths)

        for index, audio_path in enumerate(audio_paths, 1):
            filename = os.path.basename(audio_path)
            push_event(job_id, 'file', {
                'filename': filename,
                'file_index': index,
                'total_files': total_files,
            })
            try:
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
                result['outputs'] = outputs
                result['download_urls'] = public_file_downloads(job_id, index)
                file_results.append(result)
                push_event(job_id, 'file_done', {
                    'filename': filename,
                    'file_index': index,
                    'total_files': total_files,
                    'segments': len(result['segments']),
                    'downloads': result['download_urls'],
                })
            except Exception as e:
                failure = {
                    'filename': filename,
                    'file_index': index,
                    'total_files': total_files,
                    'error': str(e),
                }
                failures.append(failure)
                push_event(job_id, 'file_error', failure)

        if not file_results:
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
        if failures:
            message += f", 실패 {len(failures)}개"
        push_event(job_id, 'done', {
            'message': message,
            'total_files': total_files,
            'successful_files': len(file_results),
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


@app.route('/api/transcribe', methods=['POST'])
def transcribe():
    model_size = request.form.get('model', 'large-v3-turbo')
    language = request.form.get('language', '') or None
    prompt = request.form.get('prompt', '') or None

    job_id = str(uuid.uuid4())
    jobs[job_id] = {'status': 'queued', 'segments': [], 'info': None, 'error': None, 'created_at': time.time()}
    job_queues[job_id] = Queue()

    if 'file' in request.files and request.files['file'].filename:
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
                    run_batch_transcription_job(
                        job_id,
                        audio_paths,
                        model_size,
                        language,
                        prompt,
                        batch_name='gdrive_folder_batch',
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
    if fmt not in ('txt', 'srt', 'json'):
        return jsonify({'error': '지원하지 않는 형식'}), 400
    file_path = job['outputs'].get(fmt)
    if not file_path or not os.path.exists(file_path):
        return jsonify({'error': '파일 없음'}), 404
    stem = job['outputs'].get('stem', 'transcript')
    return send_file(file_path, as_attachment=True, download_name=f"{stem}.{fmt}")


@app.route('/api/download/<job_id>/file/<int:file_index>/<fmt>')
def download_batch_file(job_id, file_index, fmt):
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    job = jobs[job_id]
    if job['status'] != 'done':
        return jsonify({'error': '아직 완료되지 않았습니다.'}), 400
    if fmt not in ('txt', 'srt', 'json'):
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
    return send_file(file_path, as_attachment=True, download_name=f"{stem}.{fmt}")


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
