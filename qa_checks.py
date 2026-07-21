"""Automatic quality checks for finished transcripts.

Two failure modes this catches, both discovered in production:
- Repetition hallucinations: Whisper loops one or two tokens for many
  seconds ("네. 네. 네. ..." / "and energy, and energy, ...").
- Unreliable language auto-detection: clips opening with silence/music get
  misdetected and come out reading like a translation.
"""

import unicodedata

HALLUCINATION_MIN_TOKENS = 15
HALLUCINATION_MAX_UNIQUE = 3
LOW_CONFIDENCE = 0.7


def nfc(text):
    return unicodedata.normalize('NFC', text or '')


def is_hallucinated(text):
    tokens = (text or '').split()
    return len(tokens) >= HALLUCINATION_MIN_TOKENS and len(set(tokens)) <= HALLUCINATION_MAX_UNIQUE


def hallucination_spans(segments):
    """[(start, end, preview)] for segments that look like repetition loops."""
    return [
        (seg.get('start'), seg.get('end'), (seg.get('text') or '')[:40])
        for seg in segments or []
        if is_hallucinated(seg.get('text'))
    ]


def check_file(filename, segments, info=None, requested_language=None):
    """Return an issue dict for one transcribed file, or None if clean."""
    issues = []

    spans = hallucination_spans(segments)
    if spans:
        total = sum((end or 0) - (start or 0) for start, end, _ in spans)
        issues.append({
            'type': 'hallucination',
            'message': f'반복 환각 {len(spans)}개 구간 (약 {round(total)}초)',
            'spans': [
                {'start': start, 'end': end, 'preview': preview}
                for start, end, preview in spans[:5]
            ],
        })

    info = info or {}
    detected = info.get('language')
    probability = info.get('language_probability')
    if not requested_language and detected and probability is not None and probability < LOW_CONFIDENCE:
        issues.append({
            'type': 'language',
            'message': f'언어 감지 신뢰도 낮음 ({detected}, {round(probability * 100)}%)',
        })

    if not issues:
        return None
    return {'filename': nfc(filename), 'issues': issues}


def qa_report(file_results, requested_language=None):
    """Scan a job's file results. Returns {'flagged': [...], 'checked': N}."""
    flagged = []
    for result in file_results or []:
        issue = check_file(
            result.get('filename'),
            result.get('segments'),
            info=result.get('info'),
            requested_language=requested_language,
        )
        if issue:
            flagged.append(issue)
    return {'checked': len(file_results or []), 'flagged': flagged}
