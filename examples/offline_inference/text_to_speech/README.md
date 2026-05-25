# Text-To-Speech

vLLM-Omni supports several autoregressive TTS models. They share a mostly
common CLI shape (`--text`, `--ref-audio`, `--ref-text`, plus an
output-path flag — `--output-dir` for most, `--output` for OmniVoice) and
live together in this hub. Each model has its own subdirectory containing
a single `end2end.py` script; this README is the single doc entry point.

For online serving, see [`examples/online_serving/text_to_speech/`](../../online_serving/text_to_speech/README.md). For the full
list of supported architectures across all modalities, see
[Supported Models](../../../docs/models/supported_models.md).

## Supported Models

| Model | HuggingFace repo | Stages | Voice cloning | Streaming | Special modes | Sample rate |
|---|---|---|---|---|---|---|
| CosyVoice3 | `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` | 2 (talker + code2wav) | ✓ | ✓ | — | 24 kHz |
| Fish Speech S2 Pro | `fishaudio/s2-pro` | dual-AR | ✓ | ✓ | — | 44.1 kHz |
| GLM-TTS | `zai-org/GLM-TTS` | 2 (AR + DiT) | ✓ (required) | ✓ | — | 24 kHz |
| Ming-omni-tts | `inclusionAI/Ming-omni-tts-0.5B` | 2 (AR + audio VAE) | ✓ | ✓ | style / IP / dialect / TTA / podcast | 44.1 kHz |
| Ming-flash-omni-TTS | `Jonathan1909/Ming-flash-omni-2.0` | single (talker only) | — (caption-controlled) | — | style / IP / basic captions | 44.1 kHz |
| MOSS-TTS-Nano | `OpenMOSS-Team/MOSS-TTS-Nano` | single (AR + codec) | ✓ (required) | ✓ | voice_clone, continuation | 48 kHz |
| OmniVoice | `k2-fsa/OmniVoice` | 2 (gen + dec) | ✓ | — | voice design, language hint | 24 kHz |
| Qwen3-TTS | `Qwen/Qwen3-TTS-12Hz-1.7B-{CustomVoice,VoiceDesign,Base}` | 2 (talker + code2wav) | ✓ (Base) | ✓ | 3 task variants | 24 kHz |
| VoxCPM2 | `openbmb/VoxCPM2` | single (native AR) | ✓ | ✓ (online) | continuation | 48 kHz |
| Voxtral TTS | `mistralai/Voxtral-4B-TTS-2603` | varies | ✓ | ✓ | voice presets | 24 kHz |

## Common Quick Start

Most models share this invocation shape:

```bash
python examples/offline_inference/text_to_speech/<model>/end2end.py \
    --text "Hello, this is a test." \
    --ref-audio /path/to/reference.wav \
    --ref-text  "Transcript of the reference audio."
```

`--ref-audio` and `--ref-text` are optional (text-only synthesis works without them) and must be provided together for voice cloning. The exotic scripts — Qwen3-TTS, Voxtral TTS, CosyVoice3 — accept additional model-specific flags documented in their per-model section below. Qwen3-TTS in particular uses its own argparse surface (`--query-type`, `--audio-path`, etc.) and does not follow the common shape; see its section.

---

## CosyVoice3

2-stage TTS pipeline (`talker` + `code2wav`) at 24 kHz.

### Prerequisites
```bash
uv pip install -e .
# Includes soundfile, onnxruntime, x-transformers, einops via requirements.
```

Download the model snapshot:
```python
from huggingface_hub import snapshot_download
snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512',
                  local_dir='pretrained_models/Fun-CosyVoice3-0.5B')
```

If your downloaded checkpoint lacks `config.json`, add it:
```json
{
    "model_type": "cosyvoice3",
    "architectures": ["CosyVoice3Model"]
}
```
This is required because `AutoConfig.register("cosyvoice3", CosyVoice3Config)` only registers the class mapping; the loader still reads `model_type` from `config.json` to select the class.

### Quick start
```bash
python examples/offline_inference/text_to_speech/cosyvoice3/end2end.py \
    --model pretrained_models/Fun-CosyVoice3-0.5B \
    --tokenizer pretrained_models/Fun-CosyVoice3-0.5B/CosyVoice-BlankEN
```

### Voice cloning
If `--ref-audio` is omitted, the script downloads the upstream
[`zero_shot_prompt.wav`](https://github.com/FunAudioLLM/CosyVoice/blob/main/asset/zero_shot_prompt.wav) from the CosyVoice repo into the current directory.
To use your own clip, pass `--ref-audio /path/to/reference.wav`, and modify `--prompt-text` correspondingly.
```bash
python examples/offline_inference/text_to_speech/cosyvoice3/end2end.py \
    --model pretrained_models/Fun-CosyVoice3-0.5B \
    --tokenizer pretrained_models/Fun-CosyVoice3-0.5B/CosyVoice-BlankEN \
    --ref-audio /path/to/reference.wav \
    --prompt-text "You are a helpful assistant.<|endofprompt|>Trascript in your ref audio clip"
```

### Streaming
Streaming is enabled by default via `async_chunk: true` in `vllm_omni/deploy/cosyvoice3.yaml`. Pass `--no-async-chunk` on `vllm serve` to switch to the legacy synchronous path.

### Notes
- Stage 0 (`talker`) emits speech tokens; stage 1 (`code2wav`) runs flow matching + HiFiGAN to synthesize waveform.
- Deploy config auto-loads from `vllm_omni/deploy/cosyvoice3.yaml` based on HF `model_type`. Pass `--deploy-config <path>` to override.

---

## GLM-TTS

2-stage TTS pipeline (AR + DiT flow-matching) at 24 kHz. Every request requires reference audio and its transcript for zero-shot voice cloning.

### Quick start
```bash
python examples/offline_inference/text_to_speech/glm_tts/end2end.py \
    --model zai-org/GLM-TTS \
    --text "你好，这是语音合成测试。" \
    --ref-audio /path/to/reference.wav \
    --ref-text "这是参考音频的文本内容。" \
    --output-dir ./output
```

### Architecture
```
Text → [Stage 0: AR] → Speech Tokens → [Stage 1: DiT + HiFT] → Audio (24 kHz)
        (Llama-based)    (32k vocab)      (Flow Matching)
```

### Notes
- `--ref-audio` and `--ref-text` are **required** together; GLM-TTS does not support text-only synthesis.
- Reference audio should be 3-10 seconds.
- First run may be slow due to lazy loading of WhisperVQ tokenizer and CampPlus ONNX speaker embedder.
- Default sampling: temperature=1.0, top_k=25, top_p=0.8 (RAS method).
- The `--model` path should point to the repository root (not `llm/` subdirectory).

---

## Fish Speech S2 Pro

4B dual-AR text-to-speech model from FishAudio with the DAC codec at 44.1 kHz.

### Prerequisites
```bash
pip install fish-speech
```

### Quick start
```bash
python examples/offline_inference/text_to_speech/fish_speech/end2end.py \
    --text "Hello, this is a test of the Fish Speech text to speech system."
```

### Voice cloning
```bash
python examples/offline_inference/text_to_speech/fish_speech/end2end.py \
    --text "Hello, this is a cloned voice." \
    --ref-audio /path/to/reference.wav \
    --ref-text  "Transcript of the reference audio."
```

### Streaming
```bash
python examples/offline_inference/text_to_speech/fish_speech/end2end.py \
    --text "Hello, this is a streaming test." \
    --streaming
```
Streaming requires `async_chunk: true` in the stage config.

### Notes
- Output: 44.1 kHz mono WAV.
- DAC codec weights (`codec.pth`) are loaded lazily from the model directory.

---

## Ming-omni-tts

Dense 0.5B two-stage TTS pipeline (`AR + flow` + audio VAE) at 44.1 kHz. The example covers style, IP voice, music-only generation, text-to-audio events, emotion, dialect, zero-shot cloning, podcast, speech+BGM, and speech+environment-sound cases.

### Quick start
```bash
python examples/offline_inference/text_to_speech/ming_tts/end2end.py \
    --case style \
    --deploy-config vllm_omni/deploy/ming_tts.yaml \
    --enforce-eager
```

### Voice cloning
```bash
python examples/offline_inference/text_to_speech/ming_tts/end2end.py \
    --case zero_shot \
    --ref-audio /path/to/reference.wav \
    --ref-text "在此奉劝大家别乱打美白针。" \
    --deploy-config vllm_omni/deploy/ming_tts.yaml \
    --enforce-eager
```

### Streaming
```bash
python examples/offline_inference/text_to_speech/ming_tts/end2end.py \
    --case basic \
    --ref-audio /path/to/reference.wav \
    --streaming \
    --deploy-config vllm_omni/deploy/ming_tts.yaml \
    --enforce-eager
```

### Notes
- `style`, `ip`, `bgm`, and `tta` do not require reference audio.
- Reference-audio cases use `--ref-audio`; `zero_shot` also requires `--ref-text`.
- `podcast` uses multiple references via `--ref-audio-paths`.
- Full case details live in [`ming_tts/README.md`](ming_tts/README.md).

---

## Ming-flash-omni-TTS

Standalone talker-only deployment of Ming-flash-omni-2.0 at 44.1 kHz. Voice is controlled through caption fields (`风格` / `IP` / `语速`/`基频`/`音量`) rather than reference audio.

### Prerequisites
The example calls into `vllm_omni.model_executor.models.ming_flash_omni.prompt_utils` for the default prompt and instruction builder; no extra pip install on top of the base vLLM-Omni install.

### Quick start
```bash
python examples/offline_inference/text_to_speech/ming_flash_omni_tts/end2end.py --case style
```

### Cases
```bash
# ASMR-style whisper (caption-driven)
python examples/offline_inference/text_to_speech/ming_flash_omni_tts/end2end.py --case style

# IP voice (preset character voice via caption)
python examples/offline_inference/text_to_speech/ming_flash_omni_tts/end2end.py --case ip

# Basic speed/pitch/volume control
python examples/offline_inference/text_to_speech/ming_flash_omni_tts/end2end.py --case basic
```

Override the default text per case with `--text`, write to a custom path with `--output`.

### Notes
- Talker-only deployment — for the multimodal Ming-flash-omni example, see [`examples/offline_inference/ming_flash_omni/`](../../ming_flash_omni/).
- Deploy config: `vllm_omni/deploy/ming_flash_omni_tts.yaml` (single GPU, `enforce_eager`, `max_num_seqs: 1`).
- Decode defaults from the Ming cookbook: `max_decode_steps=200`, `cfg=2.0`, `sigma=0.25`, `temperature=0.0`, `use_zero_spk_emb=True`.

---

## MOSS-TTS-Nano

Single-stage 0.1B AR LM + MOSS-Audio-Tokenizer-Nano codec at 48 kHz mono (mixed down from upstream stereo). ZH / EN / JA. Every request requires a reference clip via `--ref-audio`.

> **No built-in speaker presets.** `--ref-audio` is required on every call. Default `--mode voice_clone` matches upstream's recommended workflow; `--mode continuation` is exposed for completeness but upstream's continuation-with-prompt path emits very short / near-silent output, so it is rarely useful in practice. Sample reference clips ship in the upstream repo under [`assets/audio/`](https://github.com/OpenMOSS/MOSS-TTS-Nano/tree/main/assets/audio) (e.g. `zh_1.wav`, `en_2.wav`, `jp_2.wav`).

### Quick start
```bash
# Fetch a sample reference clip (one-off, user-scoped cache).
REF_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/moss-tts-nano"
mkdir -p "$REF_DIR"
[ -s "$REF_DIR/zh_1.wav" ] || \
    curl -L -o "$REF_DIR/zh_1.wav" https://raw.githubusercontent.com/OpenMOSS/MOSS-TTS-Nano/main/assets/audio/zh_1.wav

python examples/offline_inference/text_to_speech/moss_tts_nano/end2end.py \
    --text "你好，这是MOSS-TTS-Nano的语音合成演示。" \
    --ref-audio "$REF_DIR/zh_1.wav"
```
The first run downloads `OpenMOSS-Team/MOSS-TTS-Nano` and `OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano` from Hugging Face.

### Reproducible runs
```bash
python examples/offline_inference/text_to_speech/moss_tts_nano/end2end.py \
    --text "Deterministic test." \
    --ref-audio "$REF_DIR/en_2.wav" \
    --seed 42
```

### Notes
- Output: 48 kHz mono WAV (the tokenizer is internally stereo at 48 kHz; the wrapper averages to mono before reaching the engine).
- Deploy config: `vllm_omni/deploy/moss_tts_nano.yaml` (auto-loaded; override with `--deploy-config`).
- Default `--max-new-frames 375` ≈ 14 s of audio; raise for longer outputs.
- `--ref-text` is rejected in `voice_clone` mode and required only with `--mode continuation`.
- Run `--help` for the full sampling-knob surface (`--audio-temperature`, `--audio-top-k`, `--audio-top-p`, `--text-temperature`).

---

## OmniVoice

Zero-shot multilingual TTS supporting 600+ languages, with three modes (auto / clone / design).

### Prerequisites
```bash
huggingface-cli download k2-fsa/OmniVoice
```
Voice cloning requires `transformers>=5.3.0`. Auto and design modes work with `transformers>=4.57.0`.

### Quick start (auto voice)
```bash
python examples/offline_inference/text_to_speech/omnivoice/end2end.py \
    --model k2-fsa/OmniVoice \
    --text "Hello, this is a test."
```

### Voice cloning
```bash
python examples/offline_inference/text_to_speech/omnivoice/end2end.py \
    --model k2-fsa/OmniVoice \
    --text "Hello, this is a test." \
    --ref-audio ref.wav \
    --ref-text  "This is the reference transcription."
```

### Voice design
```bash
python examples/offline_inference/text_to_speech/omnivoice/end2end.py \
    --model k2-fsa/OmniVoice \
    --text "Hello, this is a test." \
    --instruct "female, low pitch, british accent"
```

### Language hint
```bash
python examples/offline_inference/text_to_speech/omnivoice/end2end.py \
    --model k2-fsa/OmniVoice \
    --text "你好，这是一个测试。" \
    --lang zh
```

### Seed for Reproducibility
```bash
python examples/offline_inference/text_to_speech/omnivoice/end2end.py \
    --model k2-fsa/OmniVoice \
    --text "Hello, this is a test." \
    --seed 42
```

### Notes
- Stage 0 (Generator): Qwen3-0.6B with 32-step iterative unmasking.
- Stage 1 (Decoder): HiggsAudioV2 RVQ + DAC at 24 kHz.

---

## Qwen3-TTS

3-task-variant TTS with 24 kHz output. Has its own argparse surface (this script does not follow the common `--text` / `--ref-audio` shape).

### Prerequisites
For ROCm builds, replace `onnxruntime` with `onnxruntime-rocm`:
```bash
pip uninstall onnxruntime
pip install onnxruntime-rocm
```

### Task variants
- `CustomVoice`: predefined speaker (speaker ID) with optional style instruction.
- `VoiceDesign`: text + descriptive instruction designs a new voice.
- `Base`: voice cloning from reference audio + transcript.

```bash
# Single sample
python examples/offline_inference/text_to_speech/qwen3_tts/end2end.py --query-type CustomVoice
python examples/offline_inference/text_to_speech/qwen3_tts/end2end.py --query-type VoiceDesign
python examples/offline_inference/text_to_speech/qwen3_tts/end2end.py --query-type Base

# Base with a custom reference audio (Qwen3-TTS uses --audio-path, not --ref-audio):
python examples/offline_inference/text_to_speech/qwen3_tts/end2end.py \
    --query-type Base --audio-path /path/to/reference.wav

# Base variant has an additional mode flag:
python examples/offline_inference/text_to_speech/qwen3_tts/end2end.py --query-type Base --mode-tag icl       # default
python examples/offline_inference/text_to_speech/qwen3_tts/end2end.py --query-type Base --mode-tag xvec_only # x_vector_only_mode

# Batch (multiple prompts in one run)
python examples/offline_inference/text_to_speech/qwen3_tts/end2end.py --query-type CustomVoice --use-batch-sample
```

### Streaming
```bash
python examples/offline_inference/text_to_speech/qwen3_tts/end2end.py \
    --query-type CustomVoice \
    --streaming \
    --output-dir /tmp/out_stream
```
Streaming requires `async_chunk: true` in the stage config.

### Batched decoding
The Code2Wav stage supports batched decoding through the SpeechTokenizer. Pass multiple prompts via `--txt-prompts` and set `--batch-size` accordingly. To raise `max_num_seqs` on either stage, point `--stage-configs-path` at a stage configs YAML with the desired values (see `vllm_omni/model_executor/stage_configs/` for templates):
```bash
python examples/offline_inference/text_to_speech/qwen3_tts/end2end.py \
    --query-type CustomVoice \
    --txt-prompts examples/offline_inference/text_to_speech/qwen3_tts/benchmark_prompts.txt \
    --batch-size 4 \
    --stage-configs-path /path/to/qwen3_tts_batched.yaml
```
`--batch-size` must match a CUDA-graph capture size (1, 2, 4, 8, 16…).

### Notes
- Run `--help` for the full argument surface.
- See `qwen3_tts/end2end.py` for the prompt-length-estimation logic the Talker uses.

---

## VoxCPM2

Single-stage native AR TTS at 48 kHz. Pipeline: `feat_encoder → MiniCPM4 → FSQ → residual_lm → LocDiT → AudioVAE`.

### Prerequisites
```bash
pip install voxcpm
# or, for a local source checkout:
export VLLM_OMNI_VOXCPM_CODE_PATH=/path/to/voxcpm
```

### Quick start
```bash
python examples/offline_inference/text_to_speech/voxcpm2/end2end.py \
    --model openbmb/VoxCPM2 \
    --text "Hello, this is a VoxCPM2 demo."
```

### Voice cloning
Pass a reference audio for isolated cloning, or both `--ref-audio` + `--ref-text` for prompt continuation:
```bash
python examples/offline_inference/text_to_speech/voxcpm2/end2end.py \
    --text "Hello, this is a voice clone demo." \
    --ref-audio /path/to/reference.wav \
    --ref-text  "Transcript of the reference audio."
```

### Streaming
Streaming is exposed through the online OpenAI Speech API (`stream=true`). See [`examples/online_serving/text_to_speech/voxcpm2/gradio_demo.py`](../../online_serving/text_to_speech/voxcpm2/gradio_demo.py) for an AudioWorklet-based gapless streaming player; the offline `end2end.py` script does not expose a streaming path.

### Notes
- Output: 48 kHz mono WAV.
- Deploy config: `vllm_omni/deploy/voxcpm2.yaml` (auto-loaded by HF `model_type`).

---

## Voxtral TTS

Voxtral-4B-TTS (Mistral). Has its own argparse surface; uses voice presets and the `mistral_common` `SpeechRequest` protocol.

### Prerequisites
Latest `mistral_common` with `SpeechRequest` support:
```bash
pip install -e /path/to/mistral-common  # or upgrade from PyPI when available
```

### Quick start (voice preset)
```bash
python examples/offline_inference/text_to_speech/voxtral_tts/end2end.py \
    --write-audio --voice cheerful_female \
    --model mistralai/Voxtral-4B-TTS-2603 \
    --text "That eerie silence after the first storm was just the calm before another round of chaos, wasn't it?"
```

### Voice cloning (capability gated upstream)
```bash
python examples/offline_inference/text_to_speech/voxtral_tts/end2end.py \
    --write-audio \
    --model mistralai/Voxtral-4B-TTS-2603 \
    --text "This is a test message." \
    --ref-audio path/to/reference_audio.wav
```

### Streaming + concurrency
```bash
python examples/offline_inference/text_to_speech/voxtral_tts/end2end.py \
    --num-prompts 32 --concurrency 8 --streaming --write-audio --voice neutral_female \
    --model mistralai/Voxtral-4B-TTS-2603 \
    --text "..."
```
Available voice presets are listed on the HF model card (`mistralai/Voxtral-4B-TTS-2603`).

### Notes
- `--num-prompts N` replicates the prompt for performance measurement.
- `--concurrency M` requires `--streaming` and must evenly divide `--num-prompts`.
- Run `--help` for the full argument surface.

---

## SoulX-Singer

Single-stage flow-matching DiT for **singing voice synthesis (SVS)** and **singing voice conversion (SVC)** @ 24 kHz. Heavy preprocessing (vocal extract, lyrics, notes, optional MIDI edit) runs in the upstream [SoulX-Singer](https://github.com/Soul-AILab/SoulX-Singer) repo — vLLM-Omni only consumes **already produced** json/wav paths.

One HuggingFace download contains **two checkpoints** in the same folder: `model.pt` (SVS) and `model-svc.pt` (SVC). vLLM-Omni picks the pipeline from each weight directory’s `config.json` `architectures[0]` (or from `--model-class-name` / `end2end.py --svs`, same as other diffusion models).

### Prerequisites

```bash
# 1. Download weights once
BASE=path/to/SoulX-Singer

huggingface-cli download Soul-AILab/SoulX-Singer --local-dir "$BASE"

# 2. Routing: two view directories
# Same pattern as using separate `--model` paths for SVS vs SVC— no need to edit `config.json`
# when switching modes. Symlink the shared files; only `config.json` differs:
SVC_DIR=path/to/SoulX-Singer-svc

mkdir -p "$SVC_DIR"
cp $BASE/{config.yaml,README.md,assets} $SVC_DIR
mv $BASE/model-svc.pt $SVC_DIR/model-svc.pt

cat > "$BASE/config.json" <<'EOF'
{
  "model_type": "soulxsinger",
  "architectures": ["SoulXSingerPipeline"],
  "max_num_seqs": 1
}
EOF

cat > "$SVC_DIR/config.json" <<'EOF'
{
  "model_type": "soulxsinger",
  "architectures": ["SoulXSingerSVCPipeline"],
  "max_num_seqs": 1
}
EOF
```

| Directory | Pipeline | Checkpoint loaded |
|-----------|----------|-------------------|
| `$BASE` | `SoulXSingerPipeline` | `model.pt` |
| `$SVC_DIR` | `SoulXSingerSVCPipeline` | `model-svc.pt` |

Runtime hyper-parameters still come from `config.yaml` in `$BASE` (symlinked into both dirs). Do not confuse it with vLLM-Omni’s `config.json` routing file.


### Preprocess (upstream, required for SVS)

Omni does not run preprocess in `vllm serve` or `end2end.py`. Clone [SoulX-Singer](https://github.com/Soul-AILab/SoulX-Singer), follow [`preprocess/README.md`](https://github.com/Soul-AILab/SoulX-Singer/tree/main/preprocess), edit `example/preprocess.sh`, and run `bash example/preprocess.sh` from that repo. Outputs `metadata.json` under each configured `save_dir`; pass absolute paths to `end2end.py` below.

### Quick start (SVS)

```bash
python examples/offline_inference/text_to_speech/soulxsinger/end2end.py \
    --model "$SVS_DIR" \
    --svs \
    --prompt-metadata-path tests/assets/soulxsinger/zh_prompt.json \
    --target-metadata-path tests/assets/soulxsinger/zh_target.json \
    --audio-path tests/assets/soulxsinger/zh_prompt.mp3 \
    --control score \
    --num-inference-steps 16 \
    --guidance-scale 3.0 \
    --output output.wav
```

With the two-directory layout, `--svs` is optional if `$SVS_DIR/config.json` already lists `SoulXSingerPipeline`. One `generate` call merges all target segments inside the pipeline. Fixtures under `tests/assets/soulxsinger/` still need a real `--audio-path`.

### Singing voice conversion (SVC)


```bash
python examples/offline_inference/text_to_speech/soulxsinger/end2end.py \
    --model "$SVC_DIR" \
    --prompt-wav-path tests/assets/soulxsinger/zh_prompt.mp3 \
    --target-wav-path tests/assets/soulxsinger/music.mp3 \
    --prompt-f0-path tests/assets/soulxsinger/zh_prompt_f0.npy \
    --target-f0-path tests/assets/soulxsinger/music_f0.npy \
    --num-inference-steps 16 \
    --guidance-scale 3.0 \
    --output output_svc.wav
```

### Notes

- Output: 24 kHz mono WAV (full utterance per request; **no PCM streaming**).
- SVS `--control`: `score` (note pitches from metadata) or `melody` (F0). Tuning: `--num-inference-steps`, `--guidance-scale`, `--auto-shift` / `--pitch-shift` (`end2end.py --help`).
- Preprocess + MIDI correction: upstream [SoulX-Singer](https://github.com/Soul-AILab/SoulX-Singer) only; poor lyrics/note labels hurt SVS quality.
- Preprocess (upstream repo, `uv` env): `uv venv && source .venv/bin/activate && uv pip install -r preprocess/requirements.txt`; extra deps below often surface only at runtime.
    - Install `ffmpeg` on `PATH` (system package, e.g. `apt install ffmpeg`) — `funasr` checks it on import; not in `requirements.txt`.
    - `uv pip install "nemo_toolkit[asr]==2.6.1"` — English lyrics via Parakeet; pulls `lightning` and related ASR deps.
    - `uv pip install lhotse==1.32.2` — pin after NeMo install to avoid `DynamicCutSampler` / PyTorch `Sampler` mismatch.
    - `python -c "import nltk; nltk.download('cmudict'); nltk.download('averaged_perceptron_tagger_eng')"` — NLTK data for English G2P.
    - Set `language=English` in `example/preprocess.sh` for English audio; default `Mandarin` uses Paraformer + Chinese G2P.
    - Parakeet on PyTorch 2.10: use upstream `lyric_transcription.py` with CUDA graphs disabled if decode fails on `cuda-bindings` API mismatch.

---
