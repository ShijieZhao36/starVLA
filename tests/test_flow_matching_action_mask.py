"""CPU-only sampler regression tests; no pretrained weights or DiT construction."""

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import (
    LayerwiseFlowmatchingActionHead,
)


class RecordingIdentity(nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, actions):
        self.inputs.append(actions.detach().clone())
        return actions


class ZeroTimeEncoding(nn.Module):
    def forward(self, timesteps):
        return torch.zeros(*timesteps.shape, 4)


class FirstHalf(nn.Module):
    def forward(self, features):
        return features[..., :4]


class TinyEncoder(nn.Module):
    """Expose the same layers used by simulated-delay's inline encoding."""

    def __init__(self):
        super().__init__()
        self.layer1 = RecordingIdentity()
        self.pos_encoding = ZeroTimeEncoding()
        self.layer2 = FirstHalf()
        self.layer3 = nn.Identity()

    def forward(self, actions, timesteps):
        return self.layer1(actions)


class IdentityDiT(nn.Module):
    def forward(self, hidden_states, **kwargs):
        return hidden_states


class CoupledVelocity(nn.Module):
    """Each output depends on every input, as in the real action encoder."""

    def __init__(self, inactive_bias):
        super().__init__()
        self.inactive_bias = inactive_bias

    def forward(self, features):
        return 1 + 0.1 * features + 0.1 * features.sum(-1, keepdim=True) + self.inactive_bias


def tiny_head(inactive_bias=0):
    head = LayerwiseFlowmatchingActionHead.__new__(LayerwiseFlowmatchingActionHead)
    nn.Module.__init__(head)
    head.action_dim = 4
    head.action_horizon = 3
    head.num_inference_timesteps = 4
    head.num_timestep_buckets = 100
    head.config = SimpleNamespace(add_pos_embed=False)
    head.action_encoder = TinyEncoder()
    head.action_decoder = CoupledVelocity(inactive_bias)
    head.model = IdentityDiT()
    head.future_tokens = nn.Embedding(1, 4)
    return head


class FlowMatchingActionMaskTest(unittest.TestCase):
    modes = ("regular", "fallback", "pigdm", "simulated_delay")

    def sample(self, head, mode, mask=None, prev=None):
        embeddings = [torch.zeros(2, 1, 4)]
        torch.manual_seed(123)
        if mode == "regular":
            return head.predict_action(embeddings, action_mask=mask)
        if mode == "fallback":
            return head.predict_action_realtime(embeddings, action_mask=mask)
        if prev is None:
            prev = torch.full((2, 3, 4), 2.0)
        return head.predict_action_realtime(
            embeddings,
            prev_action_chunk=prev,
            mode=mode,
            action_mask=mask,
        )

    def test_inactive_inputs_and_outputs_stay_zero_for_mixed_robots(self):
        mask = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=torch.bool)
        expanded = mask[:, None, :].expand(2, 3, 4)
        # Nonzero inactive predictions and a dirty previous chunk must not leak.
        for mode in self.modes:
            with self.subTest(mode=mode):
                head = tiny_head((~mask[:, None, :]).float() * 1000)
                prev = torch.full((2, 3, 4), 2.0).masked_fill(~expanded, 900)
                result = self.sample(head, mode, mask, prev)
                self.assertTrue(torch.equal(result[~expanded], torch.zeros_like(result[~expanded])))
                self.assertEqual(len(head.action_encoder.layer1.inputs), head.num_inference_timesteps)
                for observed in head.action_encoder.layer1.inputs:
                    self.assertTrue(torch.equal(observed[~expanded], torch.zeros_like(observed[~expanded])))
                self.assertTrue(torch.isfinite(result).all())

    def test_active_dynamics_ignore_inactive_decoder_outputs_and_prefix(self):
        mask = torch.tensor([1, 0, 1, 0], dtype=torch.bool)
        expanded = mask[None, None, :].expand(2, 3, 4)
        for mode in self.modes:
            with self.subTest(mode=mode):
                reference = tiny_head()
                noisy = tiny_head((~mask).float() * 1000)
                clean_prev = torch.full((2, 3, 4), 2.0).masked_fill(~expanded, 0)
                dirty_prev = clean_prev.masked_fill(~expanded, 900)
                expected = self.sample(reference, mode, mask, clean_prev)
                actual = self.sample(noisy, mode, mask, dirty_prev)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_all_ones_is_identical_to_legacy_unmasked_sampling(self):
        for mode in self.modes:
            with self.subTest(mode=mode):
                head = tiny_head()
                expected = self.sample(head, mode)
                actual = self.sample(head, mode, torch.ones(4))
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_supported_shapes_and_time_varying_mask(self):
        mask = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1]])
        shapes = (mask, mask[:, None, :], mask[:, None, :].expand(2, 3, 4))
        for mode in self.modes:
            expected = self.sample(tiny_head(), mode, mask)
            for candidate in shapes:
                with self.subTest(mode=mode, shape=tuple(candidate.shape)):
                    actual = self.sample(tiny_head(), mode, candidate)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            temporal = mask[:, None, :].expand(2, 3, 4).clone().bool()
            temporal[:, 1] = ~temporal[:, 1]
            actual = self.sample(tiny_head(), mode, temporal)
            self.assertTrue(torch.equal(actual[~temporal], torch.zeros_like(actual[~temporal])))

    def test_invalid_shapes_and_values_fail_explicitly(self):
        invalid = (
            torch.ones(3),
            torch.ones(3, 4),  # (T,D) must not accidentally broadcast as (B,D).
            torch.ones(1, 4),
            torch.ones(2, 2, 4),
            torch.ones(1, 2, 3, 4),
            torch.tensor([1.0, 0.5, 0.0, 1.0]),
            torch.tensor([1.0, float("nan"), 0.0, 1.0]),
        )
        for mask in invalid:
            for mode in self.modes:
                with self.subTest(mode=mode, shape=tuple(mask.shape)):
                    with self.assertRaises(ValueError):
                        self.sample(tiny_head(), mode, mask)

    def test_zero_delay_fallback_forwards_mask(self):
        head = tiny_head()
        mask = torch.tensor([1, 0, 1, 0])
        expected = self.sample(head, "regular", mask)
        torch.manual_seed(123)
        actual = head.predict_action_realtime(
            [torch.zeros(2, 1, 4)],
            prev_action_chunk=torch.ones(2, 3, 4),
            inference_delay=0,
            action_mask=mask,
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
