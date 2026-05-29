
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from vllm.logger import init_logger

from ..utils import (
    boundary2Interval,
    denorm_f0,
    f0_to_coarse,
    load_model_ckpt,
    load_mono_audio,
    load_rosvot_config,
    norm_interp_f0,
    pad_or_cut_xd,
    resample_mono,
)
from .mel import MelNet
from .rosvot import MidiExtractor
from .utils import align_word, get_mel_len, regulate_ill_slur, regulate_real_note_itv

logger = init_logger(__name__)


class RosvotModel(nn.Module):
    """ROSVOT: mel + optional shared RMVPE + note boundary/pitch prediction."""

    def __init__(
        self,
        rosvot_ckpt: str | Path,
        *,
        config_path: str | Path = "",
        pe: nn.Module | None = None,
        thr: float = 0.85,
        verbose: bool = False,
    ):
        super().__init__()
        ckpt = Path(rosvot_ckpt)
        resolved_config = Path(config_path) if config_path else ckpt.with_name("config.yaml")
        self.hparams = load_rosvot_config(
            resolved_config,
            hparams_str=f"note_bd_threshold={thr}",
        )
        if verbose:
            logger.info("ROSVOT config: %s", resolved_config)

        self.midi = MidiExtractor(self.hparams)
        self.mel_net = MelNet(self.hparams)
        if pe is not None and self.hparams.get("use_pitch_embed", False):
            self.pe = pe
        else:
            self.pe = None
        self._checkpoint_path = str(ckpt)
        self.load_checkpoint(str(ckpt), verbose=verbose)
        self.eval()

    def load_checkpoint(self, checkpoint_path: str | None = None, *, verbose: bool = False) -> None:
        path = checkpoint_path or self._checkpoint_path
        load_model_ckpt(self.midi, path, verbose=verbose)

    @torch.no_grad()
    def forward(self, wav: torch.Tensor, word_durs: list[float]) -> dict[str, Any]:
        hparams = self.hparams
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)

        mel_len = get_mel_len(wav.shape[-1], hparams["hop_size"])
        min_word_dur = hparams.get("min_word_dur", 20) / 1000
        wd_raw = list(word_durs)
        word_durs_filtered: list[float] = []
        for i, wd in enumerate(wd_raw):
            if wd < min_word_dur:
                if i == 0 and len(wd_raw) > 1:
                    wd_raw[i + 1] += wd
                elif word_durs_filtered:
                    word_durs_filtered[-1] += wd
            else:
                word_durs_filtered.append(wd)

        mel2word, _ = align_word(
            word_durs_filtered,
            mel_len,
            hparams["hop_size"],
            hparams["audio_sample_rate"],
        )
        mel2word = np.asarray(mel2word)
        if mel2word.size > 0 and mel2word[0] == 0:
            mel2word = mel2word + 1

        real_len = min(mel_len, int(np.sum(mel2word > 0)))
        T = math.ceil(min(real_len, hparams["max_frames"]) / hparams["frames_multiple"]) * hparams["frames_multiple"]
        device = wav.device
        target_samples = T * hparams["hop_size"]
        wav_t = wav.float()
        if wav_t.shape[-1] < target_samples:
            wav_t = pad_or_cut_xd(wav_t, target_samples, 1)

        pitch_coarse = uv_t = None
        if self.pe is not None:
            f0s, uvs = self.pe.get_pitch_batch(
                wav_t,
                sample_rate=hparams["audio_sample_rate"],
                hop_size=hparams["hop_size"],
                lengths=[real_len],
                fmax=hparams["f0_max"],
                fmin=hparams["f0_min"],
            )
            f0_1d, uv_1d = norm_interp_f0(f0s[0][:T])
            f0_t = pad_or_cut_xd(torch.as_tensor(f0_1d, device=device, dtype=torch.float32), T, 0).unsqueeze(0)
            uv_t = pad_or_cut_xd(torch.as_tensor(uv_1d, device=device, dtype=torch.float32), T, 0).long().unsqueeze(0)
            pitch_coarse = f0_to_coarse(denorm_f0(f0_t, uv_t)).to(device)

        mel = pad_or_cut_xd(self.mel_net(wav_t)[0], T, dim=0).unsqueeze(0)
        mel_nonpadding_mask = torch.zeros(1, T, device=device)
        mel_nonpadding_mask[:, :real_len] = 1.0
        mel = (mel.transpose(1, 2) * mel_nonpadding_mask.unsqueeze(1)).transpose(1, 2)
        mel_nonpadding = mel.abs().sum(-1) > 0

        mel2word_t = pad_or_cut_xd(torch.as_tensor(mel2word, device=device, dtype=torch.long), T, 0)
        word_bd = torch.zeros_like(mel2word_t)
        word_bd[1:] = (mel2word_t[1:] != mel2word_t[:-1]).long()
        word_bd[real_len:] = 0
        word_bd = word_bd.unsqueeze(0)

        outputs = self.midi(
            mel=mel[:, :, : hparams.get("use_mel_bins", 80)],
            word_bd=word_bd,
            pitch=pitch_coarse,
            uv=uv_t,
            non_padding=mel_nonpadding,
        )
        outputs["word_durs_filtered"] = word_durs_filtered
        outputs["real_len"] = real_len
        outputs["word_bd"] = word_bd
        return outputs

    @staticmethod
    def _load_wav(
        wav_src: str | np.ndarray,
        sample_rate: int,
        *,
        src_sample_rate: int | None = None,
    ) -> np.ndarray:
        if isinstance(wav_src, str):
            wav, _ = load_mono_audio(wav_src, target_sr=sample_rate)
            return wav
        wav = np.asarray(wav_src, dtype=np.float32)
        if src_sample_rate is not None and src_sample_rate != sample_rate:
            wav = resample_mono(wav, orig_sr=src_sample_rate, target_sr=sample_rate)
        return wav

    @staticmethod
    def _normalize_note2words(note2words: list[int]) -> list[int]:
        if not note2words:
            return []
        normalized = [note2words[0]]
        for idx in range(1, len(note2words)):
            normalized.append(max(note2words[idx], normalized[-1]))
        return normalized

    @staticmethod
    def _build_ep_types(note2words: list[int], align_words: list[str]) -> list[int]:
        ep_types: list[int] = []
        prev = -1
        for i, w in zip(note2words, align_words):
            if w == "<SP>":
                ep_types.append(1)
            else:
                ep_types.append(2 if i != prev else 3)
            prev = i
        return ep_types

    @torch.no_grad()
    def transcribe(
        self,
        item: dict[str, Any],
        *,
        segment_info: dict[str, Any] | None = None,
        verbose: bool = False,
    ) -> dict[str, Any]:
        if "word_durs" not in item:
            raise ValueError('item must contain "word_durs" from lyric transcription')

        wav_fn = item.get("wav_fn")
        if wav_fn:
            wav = self._load_wav(wav_fn, self.hparams["audio_sample_rate"])
        elif item.get("wav") is not None:
            wav = self._load_wav(
                item["wav"],
                self.hparams["audio_sample_rate"],
                src_sample_rate=item.get("sample_rate"),
            )
        else:
            raise ValueError('item must contain "wav_fn" or "wav"')

        device = next(self.parameters()).device
        outputs = self(torch.from_numpy(wav).float().to(device), list(item["word_durs"]))
        real_len = int(outputs["real_len"])
        word_durs_filtered = outputs["word_durs_filtered"]
        word_bd = outputs["word_bd"]
        item_name = item.get("item_name", "")

        note_lengths = outputs["note_lengths"].detach().cpu().numpy()
        note_bd_pred = outputs["note_bd_pred"][0].detach().cpu().numpy()[:real_len]
        note_pred = outputs["note_pred"][0].detach().cpu().numpy()[: note_lengths[0]]

        if note_pred.shape == (0,):
            rosvot_out = {"item_name": item_name, "pitches": [], "note_durs": [], "note2words": None}
        else:
            note_itv_pred = boundary2Interval(note_bd_pred)
            word_bd_for_reg = word_bd[0].detach().cpu().numpy()[:real_len]
            word_durs_for_reg = np.array(word_durs_filtered)

            if self.hparams.get("infer_regulate_real_note_itv", True):
                try:
                    note_itv_pred_secs, note2words = regulate_real_note_itv(
                        note_itv_pred,
                        note_bd_pred,
                        word_bd_for_reg,
                        word_durs_for_reg,
                        self.hparams["hop_size"],
                        self.hparams["audio_sample_rate"],
                    )
                    note_pred, note_itv_pred_secs, note2words = regulate_ill_slur(
                        note_pred, note_itv_pred_secs, note2words
                    )
                except Exception:
                    if verbose:
                        logger.exception("ROSVOT postprocess failed")
                    note_itv_pred_secs = note_itv_pred * self.hparams["hop_size"] / self.hparams["audio_sample_rate"]
                    note2words = None
            else:
                note_itv_pred_secs = note_itv_pred * self.hparams["hop_size"] / self.hparams["audio_sample_rate"]
                note2words = None

            rosvot_out = {
                "item_name": item_name,
                "pitches": note_pred.tolist(),
                "note_durs": [float(itv[1] - itv[0]) for itv in note_itv_pred_secs],
                "note2words": note2words.tolist() if note2words is not None else None,
            }

        note2words_raw = rosvot_out.get("note2words") or []
        align_words = [
            item["words"][idx - 1]
            for idx in note2words_raw
            if 0 < idx <= len(item["words"])
        ]
        ep_types = self._build_ep_types(self._normalize_note2words(note2words_raw), align_words) if align_words else []
        seg = segment_info or item

        return {
            "item_name": seg.get("item_name", item_name),
            "wav_fn": seg.get("wav_fn", item.get("wav_fn", "")),
            "origin_wav_fn": seg.get("origin_wav_fn", item.get("origin_wav_fn", "")),
            "start_time_ms": seg.get("start_time_ms", ""),
            "end_time_ms": seg.get("end_time_ms", ""),
            "language": item.get("language", ""),
            "note_text": align_words,
            "note_dur": rosvot_out.get("note_durs", []),
            "note_type": ep_types,
            "note_pitch": rosvot_out.get("pitches", []),
        }

