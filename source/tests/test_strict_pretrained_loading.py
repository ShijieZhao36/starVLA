"""CPU checks without importing the trainer's accelerator/model dependencies."""

import ast
import logging
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch


def _source_function(relative_path, name, namespace):
    """Execute the actual function body from source with minimal dependencies."""
    source = Path(__file__).resolve().parents[1] / relative_path
    tree = ast.parse(source.read_text())
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(matches) != 1:
        raise AssertionError(f"Expected one {name}, found {len(matches)}")
    function = matches[0]
    function.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(module, str(source), "exec"), namespace)
    return namespace[name]


LOAD = _source_function(
    "starVLA/training/trainer_utils/trainer_tools.py",
    "load_pretrained_backbones",
    {"torch": torch, "_dist_rank": lambda: 1, "_is_safetensors_path": lambda path: str(path).endswith(".safetensors")},
)


def _tiny_model():
    model = torch.nn.Module()
    model.backbone = torch.nn.Linear(2, 3)
    model.head = torch.nn.Linear(3, 1)
    model.register_buffer("scale", torch.tensor(2.0))
    return model


class StrictPretrainedLoadingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.checkpoint = Path(self.tmp.name) / "model.pt"
        self.source = _tiny_model()
        self.state = {key: value.clone() for key, value in self.source.state_dict().items()}

    def _save(self, state=None):
        torch.save(self.state if state is None else state, self.checkpoint)
        return str(self.checkpoint)

    def test_complete_checkpoint_strict_loads_parameters_and_buffers(self):
        target = _tiny_model()
        returned = LOAD(target, self._save(), strict=True)
        self.assertIs(returned, target)
        for key, value in target.state_dict().items():
            self.assertTrue(torch.equal(value, self.state[key]), key)

    def test_strict_missing_key_rejected(self):
        del self.state["head.bias"]
        with self.assertRaisesRegex(RuntimeError, "Missing key.*head.bias"):
            LOAD(_tiny_model(), self._save(), strict=True)

    def test_strict_unexpected_key_rejected(self):
        self.state["obsolete.weight"] = torch.zeros(1)
        with self.assertRaisesRegex(RuntimeError, "Unexpected key.*obsolete.weight"):
            LOAD(_tiny_model(), self._save(), strict=True)

    def test_shape_mismatch_rejected_in_both_modes(self):
        self.state["head.weight"] = torch.zeros(2, 3)
        checkpoint = self._save()
        for strict in [False, True]:
            with self.subTest(strict=strict), self.assertRaisesRegex(RuntimeError, "size mismatch for head.weight"):
                LOAD(_tiny_model(), checkpoint, strict=strict)

    def test_strict_partial_reload_rejected_before_checkpoint_io(self):
        with self.assertRaisesRegex(ValueError, "strict.*reload_modules"):
            LOAD(_tiny_model(), "/nonexistent/checkpoint.pt", reload_modules="backbone", strict=True)

    def test_strict_empty_reload_modules_is_full_load(self):
        target = _tiny_model()
        LOAD(target, self._save(), reload_modules="", strict=True)
        self.assertTrue(torch.equal(target.head.weight, self.state["head.weight"]))

    def test_default_mode_accepts_missing_and_unexpected_keys(self):
        target = _tiny_model()
        unchanged = target.head.bias.detach().clone()
        del self.state["head.bias"]
        self.state["obsolete.weight"] = torch.zeros(1)
        LOAD(target, self._save())
        self.assertTrue(torch.equal(target.head.bias, unchanged))
        self.assertTrue(torch.equal(target.backbone.weight, self.state["backbone.weight"]))

    def test_default_partial_reload_still_works(self):
        target = _tiny_model()
        unchanged = target.head.weight.detach().clone()
        LOAD(target, self._save(), reload_modules="backbone")
        self.assertTrue(torch.equal(target.backbone.weight, self.state["backbone.weight"]))
        self.assertTrue(torch.equal(target.head.weight, unchanged))


class TrainerStrictFlagWiringTest(unittest.TestCase):
    def test_pretrained_flag_opt_in_and_resume_unchanged(self):
        import os

        init = _source_function(
            "starVLA/training/train_starvla.py", "_init_checkpointing",
            {"os": os, "logger": logging.getLogger(__name__)},
        )
        for strict, resume in [(None, False), (True, False), (True, True)]:
            with self.subTest(strict=strict, resume=resume), tempfile.TemporaryDirectory() as directory:
                trainer_config = SimpleNamespace(pretrained_checkpoint="pretrained.pt", is_resume=resume)
                if strict is not None:
                    trainer_config.strict_pretrained_load = strict
                calls = []

                def record_load(model, checkpoint, **kwargs):
                    calls.append((checkpoint, kwargs))
                    return model

                fake = SimpleNamespace(
                    config=SimpleNamespace(output_dir=directory, trainer=trainer_config), model=object(),
                    load_pretrained_backbones=record_load,
                    _get_latest_checkpoint=lambda _: ("resume.pt", 123),
                )
                init(fake)
                if resume:
                    self.assertEqual(calls, [("resume.pt", {"reload_modules": None})])
                    self.assertEqual(fake.completed_steps, 123)
                else:
                    self.assertEqual(calls, [("pretrained.pt", {"reload_modules": None, "strict": bool(strict)})])
                    self.assertEqual(fake.completed_steps, 0)


if __name__ == "__main__":
    unittest.main()
