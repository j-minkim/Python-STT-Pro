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

ALLOWED_EXTENSIONS = {'mp3', 'wav', 'mp4', 'm4a', 'ogg', 'flac', 'webm', 'aac', 'wma'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def push_event(job_id, event_type, data):
    if job_id in job_queues:
        job_queues[job_id].put({'type': event_type, 'data': data})


def format_srt_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def save_outputs(job_id, segments, original_filename='transcript'):
    base = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
    stem = os.path.splitext(original_filename)[0] if original_filename else 'transcript'

    txt_path = base + '.txt'
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(seg['text'] for seg in segments))

    srt_path = base + '.srt'
    with open(srt_path, 'w', encoding='utf-8') as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n{format_srt_time(seg['start'])} --> {format_srt_time(seg['end'])}\n{seg['text']}\n\n")

    json_path = base + '.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)

    return {'txt': txt_path, 'srt': srt_path, 'json': json_path, 'stem': stem}


def run_transcription_job(job_id, audio_path, model_size, language, prompt, original_filename=''):
    try:
        push_event(job_id, 'status', {'status': 'loading', 'message': f'모델 로딩 중: {model_size}...'})
        jobs[job_id]['status'] = 'loading'

        from stt_engine import STTEngine
        engine = STTEngine(model_size=model_size, device=None, compute_type=None)
        engine.load_model()

        push_event(job_id, 'status', {'status': 'transcribing', 'message': '전사 진행 중...'})
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
            'duration': round(info.duration, 1)
        }
        push_event(job_id, 'info', lang_info)
        jobs[job_id]['info'] = lang_info

        results = []
        for segment in segments_gen:
            seg_data = {
                'start': round(segment.start, 2),
                'end': round(segment.end, 2),
                'text': segment.text.strip()
            }
            results.append(seg_data)
            push_event(job_id, 'segment', seg_data)

        jobs[job_id]['segments'] = results
        output_paths = save_outputs(job_id, results, original_filename)
        jobs[job_id]['outputs'] = output_paths
        jobs[job_id]['status'] = 'done'
        push_event(job_id, 'done', {'message': '전사 완료!', 'total_segments': len(results)})

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
                from gdrive_utils import download_from_gdrive, is_gdrive_url
                if not is_gdrive_url(gdrive_url):
                    raise ValueError('유효한 Google Drive URL이 아닙니다.')
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


@app.route('/api/status/<job_id>')
def status(job_id):
    if job_id not in jobs:
        return jsonify({'error': 'Not found'}), 404
    job = jobs[job_id]
    return jsonify({
        'status': job['status'],
        'segments_count': len(job.get('segments', [])),
        'info': job.get('info'),
        'error': job.get('error'),
    })


if __name__ == '__main__':
    print("\n[INFO] Python STT Pro - Web Interface")
    print("-" * 40)
    print("Running at: http://localhost:5000")
    print("-" * 40 + "\n")
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
