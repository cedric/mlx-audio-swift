"""Regenerate the Chatterbox parity fixtures in Tests/media from mlx-audio Python (0.5.9).

    HF_HOME=<hf home with mlx-community/S3TokenizerV2> HF_HUB_OFFLINE=1 \
        python scripts/make_chatterbox_parity_fixtures.py <chatterbox-multilingual-v3 dir>

chatterbox_resample_golden.safetensors: resample_audio on a synthetic signal the Swift test rebuilds.
chatterbox_v3_conds_conversational_a.safetensors: prepare_conditionals for media/conversational_a.wav.
chatterbox_v3_first_step.safetensors: T3 prompt tokens and CFG logits before the first speech token.
chatterbox_v3_s3gen.safetensors: flow noise, speech tokens -> mel, and mel -> waveform with a seeded vocoder.
"""

import sys
import wave
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_audio.tts.models.chatterbox.chatterbox import SPEECH_VOCAB_SIZE, drop_invalid_tokens
from mlx_audio.tts.models.chatterbox.t3.t3 import make_prompt_cache
from mlx_audio.tts.utils import load_model
from mlx_audio.utils import resample_audio

MEDIA = Path(__file__).resolve().parent.parent / "Tests" / "media"


def synthetic(n: int, rate: int) -> np.ndarray:
    # Keep in step with `syntheticSignal` in ChatterboxTests.swift.
    t = np.arange(n, dtype=np.float64) / rate
    noise = ((np.arange(n) * 7919) % 101) / 50.0 - 1.0
    x = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 7900 * t) + 0.1 * noise
    return x.astype(np.float32)


def resample_golden() -> None:
    arrays = {}
    for src, dst, n in [(24000, 16000, 4801), (44100, 24000, 4410)]:
        x = synthetic(n, src)
        arrays[f"{src}_{dst}"] = mx.array(resample_audio(x, src, dst))
    mx.save_safetensors(str(MEDIA / "chatterbox_resample_golden.safetensors"), arrays)


FIRST_STEP_TEXT = "Hello there, this is a test."  # Keep in step with ChatterboxTests.swift.


def first_step_golden(model, conds) -> None:
    # The prompt generate() builds, through to the CFG-combined logits for the first speech token.
    t3 = model.t3
    conds.t3.emotion_adv = mx.ones((1, 1, 1)) * 0.5
    tokens = model._tokenize_text(FIRST_STEP_TEXT, "en")
    tokens = mx.concatenate([tokens, tokens], axis=0)
    sot = mx.full((2, 1), t3.hp.start_text_token, dtype=mx.int32)
    eot = mx.full((2, 1), t3.hp.stop_text_token, dtype=mx.int32)
    tokens = mx.concatenate([sot, tokens, eot], axis=1)
    embeds = t3._prepare_inference_context(t3_cond=conds.t3, text_tokens=tokens, cfg_weight=0.5)
    hidden = t3.tfmr.model(inputs=None, input_embeddings=embeds, cache=make_prompt_cache(t3.tfmr))
    logits = t3.speech_head(hidden[:, -1:, :]).squeeze(1)
    logits = logits[0:1] + 0.5 * (logits[0:1] - logits[1:2])
    mx.save_safetensors(
        str(MEDIA / "chatterbox_v3_first_step.safetensors"),
        {"text_tokens": tokens[0].astype(mx.int32), "prompt_length": mx.array([embeds.shape[1]]), "logits": logits},
    )


def s3gen_golden(model, conds) -> None:
    # Tokens from a seeded T3 run, cleaned as generate() does; any real token sequence works.
    t3 = model.t3
    tokens = model._tokenize_text(FIRST_STEP_TEXT, "en")
    tokens = mx.concatenate([tokens, tokens], axis=0)
    sot = mx.full((2, 1), t3.hp.start_text_token, dtype=mx.int32)
    eot = mx.full((2, 1), t3.hp.stop_text_token, dtype=mx.int32)
    tokens = mx.concatenate([sot, tokens, eot], axis=1)
    mx.random.seed(7)
    speech = t3.inference(t3_cond=conds.t3, text_tokens=tokens, max_new_tokens=1000, temperature=0.8,
                          cfg_weight=0.5, repetition_penalty=1.2, min_p=0.05, top_p=1.0)
    speech = drop_invalid_tokens(speech[0:1])
    ids = np.array(speech).reshape(-1)
    speech = mx.array(ids[ids < SPEECH_VOCAB_SIZE].reshape(1, -1).astype(np.int32))

    mel = model.s3gen.flow_inference(speech, conds.gen, finalize=True)
    mx.random.seed(123)
    wav, _ = model.s3gen.hift_inference(mel)
    mx.save_safetensors(
        str(MEDIA / "chatterbox_v3_s3gen.safetensors"),
        {
            "flow_noise_head": model.s3gen.flow.decoder.rand_noise[:, :, :64],
            "speech_tokens": speech,
            "mel": mel,
            "wav": wav.astype(mx.float16),
        },
    )


def conds_golden(model_dir: str) -> None:
    with wave.open(str(MEDIA / "conversational_a.wav")) as w:
        assert w.getframerate() == 24000 and w.getnchannels() == 1 and w.getsampwidth() == 2
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    audio = mx.array(pcm.astype(np.float32) / 32768.0)
    model = load_model(model_dir)
    conds = model.prepare_conditionals(audio, 24000)
    mx.save_safetensors(
        str(MEDIA / "chatterbox_v3_conds_conversational_a.safetensors"),
        {
            "t3.speaker_emb": conds.t3.speaker_emb,
            "t3.cond_prompt_speech_tokens": conds.t3.cond_prompt_speech_tokens.astype(mx.int32),
            "gen.embedding": conds.gen["embedding"],
            "gen.prompt_token": conds.gen["prompt_token"].astype(mx.int32),
            "gen.prompt_feat": conds.gen["prompt_feat"],
        },
    )
    first_step_golden(model, conds)
    s3gen_golden(model, conds)


if __name__ == "__main__":
    resample_golden()
    conds_golden(sys.argv[1])
