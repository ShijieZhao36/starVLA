"""Test stage handoff gates without model weights, GPUs or the real dataset."""
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

SCRIPT = Path(__file__).parents[1] / "train_files/pipeline_checks.py"
spec = importlib.util.spec_from_file_location("droid_pipeline_checks", SCRIPT)
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)


def make_run(root, stage="pretrain", source=None, steps=10):
    root.mkdir()
    (root / "final_model").mkdir()
    (root / "final_model/pytorch_model.pt").write_bytes(b"synthetic model fixture")
    tag = checks.TAGS[stage]
    cfg = {
        "framework": {"name": "QwenPI", "action_model": {"action_dim": 90, "state_dim": 90, "action_horizon": 8, "decoder_input_norm": True}},
        "datasets": {"vla_data": {"use_relative_action": True, "per_device_batch_size": 8,
            "data_mix": "droid_stage3_unmasked" if stage == "pretrain" else "libero_plus_stage1"}},
        "trainer": {"max_train_steps": steps, "gradient_accumulation_steps": 1, "is_resume": False,
            "pretrained_checkpoint": str(source / "final_model/pytorch_model.pt") if source else None,
            "reload_modules": None, "strict_pretrained_load": True},
    }
    (root / "config.full.yaml").write_text(yaml.safe_dump(cfg))
    (root / "train_exit_code").write_text("0\n")
    (root / "dataset_statistics.json").write_text(json.dumps({tag: {"action": {"mask": [True] * (17 if stage == "pretrain" else 10)}}}))
    metric = {"step": steps, "action_dit_loss": .04, "source_exposure/cumulative_examples": steps * 64,
              f"source_exposure/{tag}/cumulative_examples": steps * 64}
    (root / "metrics.jsonl").write_text(json.dumps(metric) + "\n")
    (root / "train.log").write_text("loaded <full_model> model parameters\nTraining complete. Final model saved at final_model\n")
    if source:
        (root / "training_start_model_sha256.json").write_text(json.dumps({"phase": "training_start", "configured_pretrained_checkpoint": cfg["trainer"]["pretrained_checkpoint"]}))
    return root


def test_valid_pretrain_handoff(tmp_path):
    run = make_run(tmp_path / "pretrain")
    marker = checks.validate_training(run, "pretrain", 10, 8)
    assert checks.validate_source(run, expected_steps=10) == marker


def make_ready_stage(tmp_path, stage):
    source = None
    if stage == "posttrain":
        source = make_run(tmp_path / "source")
        checks.validate_training(source, "pretrain", 10, 8)
    return make_run(tmp_path / stage, stage, source), source


@pytest.mark.parametrize("stage", ["pretrain", "posttrain"])
@pytest.mark.parametrize("log_format", ["plain", "rich_wrapped", "ansi_rich_wrapped"])
def test_completion_accepts_realistic_log_formatting(tmp_path, stage, log_format):
    run, source = make_ready_stage(tmp_path, stage)
    # Rich's source column interrupts the original sentence. Collapsing whitespace
    # alone cannot turn this output back into the old contiguous completion text.
    completion = (
        "11/08 [06:59:54] INFO     | >> Training complete. Final     train_starvla.py:521\n"
        "                          model saved at\n"
        "                          /mnt/zhaoshijie/checkpoints/starV\n"
        "                          LA/droid_pretrain/final_model\n"
    )
    if log_format == "plain":
        completion = "Training complete. Final model saved at final_model\n"
    elif log_format == "ansi_rich_wrapped":
        completion = completion.replace(
            "Training complete.", "\x1b[1mTraining\x1b[0m \x1b[32mcomplete.\x1b[0m"
        )
    (run / "train.log").write_text("loaded <full_model> model parameters\n" + completion)

    marker = checks.validate_training(run, stage, 10, 8, source)

    assert marker["validated"] is True
    assert marker["stage"] == stage
    assert checks.validate_source(run, stage, expected_steps=10) == marker


@pytest.mark.parametrize("stage", ["pretrain", "posttrain"])
def test_missing_completion_event_cannot_publish_success(tmp_path, stage):
    run, source = make_ready_stage(tmp_path, stage)
    (run / "train.log").write_text(
        "loaded <full_model> model parameters\n"
        "Step 10 finished; Training incomplete. Final model not saved.\n"
    )

    with pytest.raises(ValueError, match="checkpoint saving"):
        checks.validate_training(run, stage, 10, 8, source)

    assert not (run / f"{stage}_success.json").exists()


@pytest.mark.parametrize("stage", ["pretrain", "posttrain"])
def test_empty_final_checkpoint_cannot_publish_success(tmp_path, stage):
    run, source = make_ready_stage(tmp_path, stage)
    (run / "final_model/pytorch_model.pt").write_bytes(b"")

    with pytest.raises(ValueError, match="missing or empty"):
        checks.validate_training(run, stage, 10, 8, source)

    assert not (run / f"{stage}_success.json").exists()


@pytest.mark.parametrize("damage", ["exit", "step", "loss", "exposure", "stats", "checkpoint"])
def test_incomplete_pretrain_cannot_publish_success(tmp_path, damage):
    run = make_run(tmp_path / "pretrain")
    if damage == "exit":
        (run / "train_exit_code").write_text("1\n")
    elif damage in {"step", "loss", "exposure"}:
        item = json.loads((run / "metrics.jsonl").read_text())
        key, value = {"step": ("step", 9), "loss": ("action_dit_loss", float("nan")),
                      "exposure": ("source_exposure/cumulative_examples", 320)}[damage]
        item[key] = value
        (run / "metrics.jsonl").write_text(json.dumps(item))
    elif damage == "stats":
        (run / "dataset_statistics.json").write_text('{"libero_plus_rot6d": {}}')
    else:
        (run / "final_model/pytorch_model.pt").unlink()
    with pytest.raises((ValueError, KeyError)):
        checks.validate_training(run, "pretrain", 10, 8)
    assert not (run / "pretrain_success.json").exists()


def test_changed_source_is_rejected(tmp_path):
    run = make_run(tmp_path / "pretrain")
    checks.validate_training(run, "pretrain", 10, 8)
    (run / "final_model/pytorch_model.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="size changed"):
        checks.validate_source(run)


def test_posttrain_uses_full_source_and_own_statistics(tmp_path):
    source = make_run(tmp_path / "pretrain")
    checks.validate_training(source, "pretrain", 10, 8)
    post = make_run(tmp_path / "posttrain", "posttrain", source)
    marker = checks.validate_training(post, "posttrain", 10, 8, source)
    assert marker["source_checkpoint"] == str(source / "final_model/pytorch_model.pt")
    cfg = yaml.safe_load((post / "config.full.yaml").read_text())
    cfg["trainer"]["strict_pretrained_load"] = False
    (post / "config.full.yaml").write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="strict full-model"):
        checks.validate_training(post, "posttrain", 10, 8, source)


def test_posttrain_requires_source_success(tmp_path):
    source = make_run(tmp_path / "pretrain")
    post = make_run(tmp_path / "posttrain", "posttrain", source)
    with pytest.raises(FileNotFoundError):
        checks.validate_training(post, "posttrain", 10, 8, source)


@pytest.mark.parametrize(
    "damage",
    [
        "non_strict_load",
        "partial_reload",
        "wrong_checkpoint",
        "missing_source_marker",
        "missing_load_message",
        "missing_start_audit",
        "wrong_audit_phase",
        "wrong_audit_source",
    ],
)
def test_posttrain_transfer_gates_cannot_publish_success(tmp_path, damage):
    run, source = make_ready_stage(tmp_path, "posttrain")
    if damage in {"non_strict_load", "partial_reload", "wrong_checkpoint"}:
        cfg_path = run / "config.full.yaml"
        cfg = yaml.safe_load(cfg_path.read_text())
        key, value = {
            "non_strict_load": ("strict_pretrained_load", False),
            "partial_reload": ("reload_modules", "action_model"),
            "wrong_checkpoint": ("pretrained_checkpoint", str(tmp_path / "other_model.pt")),
        }[damage]
        cfg["trainer"][key] = value
        cfg_path.write_text(yaml.safe_dump(cfg))
    elif damage == "missing_source_marker":
        (source / "pretrain_success.json").unlink()
    elif damage == "missing_load_message":
        (run / "train.log").write_text("Training complete. Final model saved at final_model\n")
    elif damage == "missing_start_audit":
        (run / "training_start_model_sha256.json").unlink()
    else:
        audit_path = run / "training_start_model_sha256.json"
        audit = json.loads(audit_path.read_text())
        if damage == "wrong_audit_phase":
            audit["phase"] = "initial"
        else:
            audit["configured_pretrained_checkpoint"] = str(tmp_path / "other_model.pt")
        audit_path.write_text(json.dumps(audit))

    with pytest.raises((ValueError, FileNotFoundError)):
        checks.validate_training(run, "posttrain", 10, 8, source)

    assert not (run / "posttrain_success.json").exists()


def test_smoke_source_cannot_be_used_as_formal_training(tmp_path):
    source = make_run(tmp_path / "pretrain_smoke", steps=3)
    marker = checks.validate_training(source, "pretrain", 3, 8, smoke=True)
    assert marker["smoke"] is True and marker["source_exposure"] == 192
    assert checks.validate_source(source, expected_steps=3, smoke=True) == marker
    with pytest.raises(ValueError, match="smoke/formal"):
        checks.validate_source(source)
    post = make_run(tmp_path / "posttrain", "posttrain", source, steps=3)
    with pytest.raises(ValueError, match="smoke/formal"):
        checks.validate_training(post, "posttrain", 3, 8, source)
    assert not (post / "posttrain_success.json").exists()


def test_smoke_cannot_silently_run_a_long_training_budget(tmp_path):
    run = make_run(tmp_path / "pretrain")
    with pytest.raises(ValueError, match="exactly three steps"):
        checks.validate_training(run, "pretrain", 10, 8, smoke=True)
    assert not (run / "pretrain_success.json").exists()


@pytest.mark.parametrize("episode_count", [0, 1, 2])
def test_smoke_evaluation_requires_exactly_one_completed_episode(tmp_path, episode_count):
    source = make_run(tmp_path / "pretrain_smoke", steps=3)
    checks.validate_training(source, "pretrain", 3, 8, smoke=True)
    post = make_run(tmp_path / "posttrain_smoke", "posttrain", source, steps=3)
    checks.validate_training(post, "posttrain", 3, 8, source, smoke=True)
    output = tmp_path / "eval_smoke"
    (output / "logs/libero_goal").mkdir(parents=True)
    (output / "logs/libero_goal/0_to_1.json").write_text(json.dumps({"category": {"total_count": episode_count, "success_count": 0}}))
    if episode_count != 1:
        with pytest.raises(ValueError, match="exactly one episode"):
            checks.validate_evaluation(tmp_path, post, output, smoke=True)
        assert not (post / "eval_success.json").exists()
    else:
        result = checks.validate_evaluation(tmp_path, post, output, smoke=True)
        assert result["validated"] is True and result["smoke"] is True and result["episode_count"] == 1
        with pytest.raises(ValueError, match="smoke/formal"):
            checks.validate_evaluation(tmp_path, post, output)
