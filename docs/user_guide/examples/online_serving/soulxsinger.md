# SoulX-Singer Online Serving

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/online_serving/text_to_speech/soulxsinger>.

SVS / SVC singing voice synthesis with integrated preprocess (stage 0) and flow-matching DiT
(stage 1). Uses the OpenAI-compatible `/v1/chat/completions` endpoint with `input_audio` and
`extra_args` for target accompaniment paths.

**Current execution mode**: Pure batch (full-payload). SoulX-Singer pipelines use `async_chunk: false`
and the `soulx_preprocess` + full `soulx_preprocessed` payload handoff.

## Quick start

```bash
cd examples/online_serving/text_to_speech/soulxsinger
chmod +x run_server.sh
MODEL=/path/to/SoulX-Singer ./run_server.sh
```

```bash
python openai_chat_client.py \
    --prompt-audio /path/on/server/zh_prompt.mp3 \
    --target-audio /path/on/server/music.mp3 \
    --preprocess-weights-dir /path/on/server/SoulX-Singer-Preprocess \
    -o output.wav
```

## Documentation

- Example README: [soulxsinger/README.md](../../../examples/online_serving/text_to_speech/soulxsinger/README.md)
- Offline weights & phoneset: [text_to_speech offline guide](../../../examples/offline_inference/text_to_speech/README.md#soulx-singer)
