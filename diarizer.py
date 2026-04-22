import os
import json
import uuid
import collections
from nemo.collections.asr.models import ClusteringDiarizer
from omegaconf import OmegaConf

class NeMoDiarizer:
    def __init__(self, out_dir="data/diarization_output"):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    def _create_manifest(self, audio_path):
        manifest_path = os.path.join(self.out_dir, f"manifest_{uuid.uuid4().hex}.json")
        with open(manifest_path, 'w', encoding='utf-8') as f:
            manifest_data = {
                "audio_filepath": audio_path,
                "offset": 0,
                "duration": None,
                "label": "infer",
                "text": "-",
                "num_speakers": None # Let it auto-detect, or pass an int if known
            }
            f.write(json.dumps(manifest_data) + '\n')
        return manifest_path

    def run_diarization(self, audio_path, num_speakers=None):
        print("Starting NeMo Speaker Diarization...")
        manifest_filepath = self._create_manifest(audio_path)
        
        cfg = OmegaConf.create({
            "diarizer": {
                "manifest_filepath": manifest_filepath,
                "out_dir": self.out_dir,
                "oracle_vad": False,
                "collar": 0.25,
                "ignore_overlap": True,
                "vad": {
                    "model_path": "vad_multilingual_marblenet",
                    "parameters": {
                        "onset": 0.1,
                        "offset": 0.08,
                        "pad_onset": 0.1,
                        "pad_offset": 0.1,
                    }
                },
                "speaker_embeddings": {
                    "model_path": "titanet_large",
                    "parameters": {
                        "window_length_in_sec": 1.5,
                        "shift_length_in_sec": 0.75,
                        "multiscale_weights": [1, 1, 1],
                        "save_embeddings": False
                    }
                },
                "clustering": {
                    "parameters": {
                        "oracle_num_speakers": False if num_speakers is None else True,
                        "max_num_speakers": 8,
                        "enhanced_count_thres": 80,
                        "max_rp_threshold": 0.25,
                        "sparse_search_volume": 30,
                        "maj_vote_spk_count": False
                    }
                }
            }
        })
        
        if num_speakers is not None:
             cfg.diarizer.clustering.parameters.max_num_speakers = num_speakers

        sd_model = ClusteringDiarizer(cfg=cfg)
        sd_model.diarize()
        
        # The output RTTM file will be in out_dir/pred_rttms/
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        rttm_path = os.path.join(self.out_dir, "pred_rttms", f"{base_name}.rttm")
        
        # Parse RTTM to get segments: list of (start_time, end_time, speaker_label)
        segments = []
        if os.path.exists(rttm_path):
            with open(rttm_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 8 and parts[0] == "SPEAKER":
                        start = float(parts[3])
                        duration = float(parts[4])
                        speaker = parts[7]
                        segments.append((start, start + duration, speaker))
        else:
             print(f"Warning: RTTM file not found at {rttm_path}")
             
        # Cleanup
        try:
            os.remove(manifest_filepath)
        except:
             pass

        return segments

def align_words_with_speakers(whisper_results, speaker_segments):
    """
    Given whisper_results (containing start, end, text, and words) and speaker_segments (from RTTM),
    assigns a speaker to each word, then groups them.
    """
    if not speaker_segments:
        # Fallback if diarization failed
        return [{"speaker": "UNKNOWN", "text": seg["text"], "start": seg["start"], "end": seg["end"]} for seg in whisper_results]

    diarized_transcript = []
    current_speaker = None
    current_text = []
    current_start = None
    last_end = None

    def get_speaker_for_time(time_pt):
        # simple boundary check
        for start, end, spk in speaker_segments:
            if start <= time_pt <= end:
                return spk
        # if no exact match, find nearest
        closest_spk = "UNKNOWN"
        min_dist = float('inf')
        for start, end, spk in speaker_segments:
            dist = min(abs(time_pt - start), abs(time_pt - end))
            if start <= time_pt <= end:
                 return spk
            if dist < min_dist:
                min_dist = dist
                closest_spk = spk
        return closest_spk

    for segment in whisper_results:
        # If words exist, we do precise word-level matching
        words = segment.get("words", [])
        if not words:
             # Fallback to segment level
             mid_point = (segment["start"] + segment["end"]) / 2
             spk = get_speaker_for_time(mid_point)
             diarized_transcript.append({
                 "speaker": spk,
                 "text": segment["text"],
                 "start": segment["start"],
                 "end": segment["end"]
             })
             continue
             
        for word in words:
            mid_point = (word["start"] + word["end"]) / 2
            spk = get_speaker_for_time(mid_point)
            
            if spk != current_speaker:
                if current_speaker is not None:
                    diarized_transcript.append({
                        "speaker": current_speaker,
                        "text": " ".join(current_text),
                        "start": current_start,
                        "end": last_end
                    })
                current_speaker = spk
                current_text = [word["text"]]
                current_start = word["start"]
            else:
                current_text.append(word["text"])
            
            last_end = word["end"]
            
    if current_speaker is not None and current_text:
        diarized_transcript.append({
            "speaker": current_speaker,
            "text": " ".join(current_text),
            "start": current_start,
            "end": last_end
        })
        
    return diarized_transcript
