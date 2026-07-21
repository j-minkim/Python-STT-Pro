"""Find transcripts poisoned by Whisper repetition hallucinations.

Scans data/outputs/*/*.json for segments where one or two tokens repeat many
times (e.g. "네. 네. 네. ..."), reports affected files, and with --reset
removes those files' completion records from every resume manifest so the
next batch run re-transcribes them.

Usage (from the project root):
    python scripts/find_hallucinations.py           # report only
    python scripts/find_hallucinations.py --reset   # also reset manifests

Do NOT run --reset while a batch is running: the active job rewrites its
manifest on every file and would overwrite the reset.
"""

import argparse
import glob
import json
import os
import sys
import unicodedata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MIN_TOKENS = 15
MAX_UNIQUE = 3


def nfc(text):
    return unicodedata.normalize('NFC', text)


def is_hallucinated(text):
    tokens = (text or '').split()
    return len(tokens) >= MIN_TOKENS and len(set(tokens)) <= MAX_UNIQUE


def scan():
    """Return {source_filename: [(start, end, preview), ...]}.

    Only each source file's newest transcript is inspected, so outputs
    superseded by a re-transcription stop being flagged.
    """
    latest = {}  # filename -> (mtime, json_path)
    for path in glob.glob(os.path.join(ROOT, 'data', 'outputs', '*', '*.json')):
        if path.endswith('_diarized.json'):
            continue
        try:
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict) or not data.get('filename'):
            continue
        name = nfc(data['filename'])
        mtime = os.path.getmtime(path)
        if name not in latest or mtime > latest[name][0]:
            latest[name] = (mtime, path)

    flagged = {}
    for name, (_, path) in latest.items():
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        hits = [
            (seg.get('start'), seg.get('end'), (seg.get('text') or '')[:40])
            for seg in data.get('segments') or []
            if is_hallucinated(seg.get('text'))
        ]
        if hits:
            flagged[name] = hits
    return flagged


def reset_manifests(filenames):
    removed_total = 0
    for mpath in glob.glob(os.path.join(ROOT, 'data', 'batch_state', '*.json')):
        try:
            with open(mpath, encoding='utf-8') as f:
                manifest = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        files = manifest.get('files')
        if not isinstance(files, dict):
            continue
        kept = {
            key: entry for key, entry in files.items()
            if nfc(key.split('|')[0]) not in filenames
        }
        removed = len(files) - len(kept)
        if removed:
            manifest['files'] = kept
            tmp = mpath + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)
            os.replace(tmp, mpath)
            print(f'{os.path.basename(mpath)}: 완료 기록 {removed}개 제거')
            removed_total += removed
    return removed_total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reset', action='store_true',
                        help='remove flagged files from resume manifests so they re-transcribe')
    args = parser.parse_args()

    flagged = scan()
    if not flagged:
        print('반복 환각이 감지된 파일이 없습니다.')
        return 0

    print(f'반복 환각 감지: {len(flagged)}개 파일\n')
    for name in sorted(flagged):
        hits = flagged[name]
        total = sum((end or 0) - (start or 0) for start, end, _ in hits)
        print(f'- {name}  (환각 구간 {len(hits)}개, 약 {round(total)}초)')
        for start, end, preview in hits[:2]:
            print(f'    [{start}s~{end}s] {preview}...')

    if args.reset:
        print()
        removed = reset_manifests(set(flagged))
        print(f'\n총 {removed}개 완료 기록을 제거했습니다. 같은 폴더를 다시 제출하면 해당 파일만 재전사됩니다.')
    else:
        print('\n재전사 대상으로 표시하려면: python scripts/find_hallucinations.py --reset')
    return 0


if __name__ == '__main__':
    sys.exit(main())
