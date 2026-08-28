import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mlx.core as mx
import mlx.nn as nn

from mlx_vlm.models.cache import RotatingKVCache
from mlx_vlm.speculative.common import (
    _get_ema_scaled_block_size,
    _record_speculative_round,
    _update_speculative_ema,
)
from mlx_vlm.speculative.dflash import _dflash_next_block_size
from mlx_vlm.speculative.drafters.dflash2.dflash2 import DFlash2DraftModel
from mlx_vlm.speculative.drafters.eagle3.config import Eagle3Config, TextConfig
from mlx_vlm.speculative.drafters.eagle3.eagle3 import Eagle3DraftModel
from mlx_vlm.speculative.eagle3 import _eagle3_next_block_size
from mlx_vlm.speculative.drafters.gemma4_assistant.config import (
    Gemma4AssistantConfig,
    TextConfig as GemmaTextConfig,
)
from mlx_vlm.speculative.drafters.gemma4_assistant.gemma4_assistant import (
    Gemma4AssistantDraftModel,
)
from mlx_vlm.speculative.drafters.qwen3_5_mtp.config import (
    Qwen3_5MTPConfig,
    TextConfig as QwenTextConfig,
)
from mlx_vlm.speculative.drafters.qwen3_5_mtp.qwen3_5_mtp import (
    Qwen3_5MTPDraftModel,
)
from mlx_vlm.speculative.mtp import _mtp_next_block_size


class TestLongContextSpeculativeMitigations(unittest.TestCase):
    def test_ema_acceptance_tracking_and_scaling(self):
        draft_model = SimpleNamespace(
            accept_lens=[],
            draft_lens=[],
            _ema_accept_rate=None,
            _ema_scale_down=0,
        )

        # High acceptance rounds: EMA stays high
        for _ in range(5):
            _record_speculative_round(draft_model, accepted=3, draft_count=3)
        self.assertGreaterEqual(draft_model._ema_accept_rate, 0.9)
        self.assertEqual(_get_ema_scaled_block_size(draft_model, 4, threshold=0.35), 4)

        # Low acceptance rounds: EMA drops below 35%
        for _ in range(10):
            _record_speculative_round(draft_model, accepted=0, draft_count=3)
        self.assertLess(draft_model._ema_accept_rate, 0.35)

        # Dynamically scales down: 3 -> 2 -> 1 -> 0
        scaled = _get_ema_scaled_block_size(draft_model, 3, threshold=0.35)
        self.assertEqual(scaled, 2)
        scaled = _get_ema_scaled_block_size(draft_model, 3, threshold=0.35)
        self.assertEqual(scaled, 1)
        scaled = _get_ema_scaled_block_size(draft_model, 3, threshold=0.35)
        self.assertEqual(scaled, 0)

    def test_dflash_and_eagle_and_mtp_next_block_size_ema(self):
        draft_model = SimpleNamespace(
            accept_lens=[0] * 10,
            draft_lens=[3] * 10,
            _ema_accept_rate=0.10,
            _ema_scale_down=0,
        )
        # Scaled down due to low EMA
        dflash_bs = _dflash_next_block_size(draft_model, 4, 10)
        self.assertLess(dflash_bs, 4)

        eagle_bs = _eagle3_next_block_size(draft_model, 4, 4, 10, adaptive=False)
        self.assertLess(eagle_bs, 4)

        mtp_bs = _mtp_next_block_size(draft_model, 4, 4, 10)
        self.assertLess(mtp_bs, 4)

    def test_eagle3_sliding_window_cache_and_prefill(self):
        text_cfg = TextConfig(
            vocab_size=100,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=16,
        )
        cfg = Eagle3Config(transformer_layer_config=text_cfg)
        cfg.draft_window_size = 32
        cfg.draft_sink_tokens = 4
        cfg.confidence_threshold = 0.40

        drafter = Eagle3DraftModel(cfg)
        cache = drafter.make_cache()
        self.assertEqual(len(cache), 2)
        self.assertIsInstance(cache[0], RotatingKVCache)
        self.assertEqual(cache[0].max_size, 32)
        self.assertEqual(cache[0].keep, 4)

        # Test prefill slicing on long sequence (> window_size)
        target_model = SimpleNamespace(
            embed_tokens=nn.Embedding(100, 32),
            model=SimpleNamespace(embed_tokens=nn.Embedding(100, 32)),
        )
        drafter.reset(target_model)

        long_input_ids = mx.zeros((1, 64), dtype=mx.int32)
        long_hidden = mx.zeros((1, 64, 32), dtype=mx.float32)
        bonus_token = 5

        # Prefill should run without throwing errors and retain sink + window
        drafter.prefill_from_target_hidden(
            long_input_ids, long_hidden, bonus_token, lambda x: mx.argmax(x, axis=-1)
        )
        self.assertIsNotNone(drafter._seed_token)

    def test_eagle3_confidence_based_early_exit(self):
        text_cfg = TextConfig(
            vocab_size=100,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=16,
        )
        cfg = Eagle3Config(transformer_layer_config=text_cfg)
        cfg.confidence_threshold = 0.40

        drafter = Eagle3DraftModel(cfg)
        target_model = SimpleNamespace(
            embed_tokens=nn.Embedding(100, 32),
            model=SimpleNamespace(embed_tokens=nn.Embedding(100, 32)),
        )
        drafter.reset(target_model)

        hidden = mx.zeros((1, 1, 32), dtype=mx.float32)
        sampler = lambda x: mx.argmax(x, axis=-1)

        # Case 1: Low confidence token 1 (< 0.40) -> halts early and returns only 1 draft token
        with patch.object(drafter, "_logits", return_value=mx.zeros((1, 1, 100))):
            # Uniform logits give softmax prob = 1/100 = 0.01 < 0.40
            draft_tokens = drafter.draft_block(1, hidden, None, block_size=4, sampler=sampler)
            self.assertEqual(draft_tokens.shape[1], 1)

        # Case 2: High confidence token 1 (> 0.40) -> drafts full K=3 tokens
        high_conf_logits = mx.zeros((1, 1, 100))
        high_conf_logits[0, 0, 10] = 50.0  # Sharp spike -> prob ≈ 1.0 > 0.40
        with patch.object(drafter, "_logits", return_value=high_conf_logits):
            draft_tokens = drafter.draft_block(1, hidden, None, block_size=4, sampler=sampler)
            self.assertEqual(draft_tokens.shape[1], 3)

    def test_gemma4_assistant_sparse_kv_and_confidence_exit(self):
        text_cfg = GemmaTextConfig(
            vocab_size=100,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=16,
        )
        cfg = Gemma4AssistantConfig(text_config=text_cfg)
        cfg.draft_window_size = 16
        cfg.draft_sink_tokens = 2
        cfg.confidence_threshold = 0.40

        drafter = Gemma4AssistantDraftModel(cfg)
        target_model = SimpleNamespace(
            language_model=SimpleNamespace(
                model=SimpleNamespace(
                    embed_tokens=nn.Embedding(100, 32),
                    embed_scale=1.0,
                )
            ),
        )
        drafter.reset(target_model)

        # Test sparse shared KV slicing
        keys = mx.zeros((1, 2, 40, 16))
        values = mx.zeros((1, 2, 40, 16))
        shared_kv = {"full_attention": (keys, values)}
        drafter.set_shared_kv(shared_kv, kv_offset=40)

        sparse_k, sparse_v = drafter._shared_kv["full_attention"]
        # Length should be clamped to draft_window_size = 16
        self.assertEqual(sparse_k.shape[2], 16)
        self.assertEqual(sparse_v.shape[2], 16)

        # Test early confidence exit on token 1
        hidden = mx.zeros((1, 1, 32), dtype=mx.float32)
        sampler = lambda x: mx.argmax(x, axis=-1)

        # Uniform logits -> prob = 0.01 < 0.40 -> halts after 1 token
        with patch.object(drafter, "_forward_hidden", return_value=(hidden, hidden)):
            with patch.object(drafter, "_lm_head_fn", return_value=mx.zeros((1, 1, 100))):
                draft_tokens = drafter.draft_block(1, hidden, None, block_size=4, sampler=sampler)
                self.assertEqual(draft_tokens.shape[1], 1)

    def test_dflash2_confidence_exit(self):
        cfg = SimpleNamespace(
            mask_token_id=0,
            output_multiplier=1.0,
            final_logit_softcapping=None,
            confidence_threshold=0.40,
        )
        drafter = DFlash2DraftModel.__new__(DFlash2DraftModel)
        drafter.config = cfg
        drafter.confidence_threshold = 0.40
        drafter.candidate_selector = SimpleNamespace(
            select=lambda hidden, logits, anchor, sampler: mx.array([[10, 20, 30]])
        )
        drafter._hidden = MagicMock(return_value=mx.zeros((1, 4, 32)))

        # Uniform logits -> confidence prob = 1/100 = 0.01 < 0.40 -> truncated to 1 token
        drafter._logits = MagicMock(return_value=mx.zeros((1, 3, 100)))
        tokens = drafter.draft_block(1, mx.zeros((1, 1, 32)), None, block_size=4, sampler=lambda x: x)
        self.assertEqual(tokens.shape[1], 1)

        # Sharp logits -> confidence prob ≈ 1.0 > 0.40 -> keeps all 3 tokens
        sharp_logits = mx.zeros((1, 3, 100))
        sharp_logits[0, 0, 10] = 50.0
        drafter._logits = MagicMock(return_value=sharp_logits)
        tokens = drafter.draft_block(1, mx.zeros((1, 1, 32)), None, block_size=4, sampler=lambda x: x)
        self.assertEqual(tokens.shape[1], 3)


if __name__ == "__main__":
    unittest.main()
