import os

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


def media_duration_seconds(path):
    """Media length from container metadata (fast, no decoding). None if unknown."""
    try:
        import av
        with av.open(path) as container:
            if container.duration:
                return container.duration / av.time_base
            for stream in container.streams:
                if stream.duration and stream.time_base:
                    return float(stream.duration * stream.time_base)
    except Exception:
        return None
    return None


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
