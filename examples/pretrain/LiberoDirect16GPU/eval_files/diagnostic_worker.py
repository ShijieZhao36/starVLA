#!/usr/bin/env python3
"""Observe the original worker without changing actions, results or RNG state."""
from __future__ import annotations

import argparse
import faulthandler
import json
from pathlib import Path
import sys
import time


def event(name, **fields):
    print("DIAGNOSTIC " + json.dumps({"event": name, "time": time.time(), **fields},
                                    sort_keys=True, allow_nan=False), flush=True)


def attach_observers(worker):
    import numpy as np
    original_get_env = worker._get_libero_env
    original_rollout = worker.rollout_episode

    def get_env(*args, **kwargs):
        env, description = original_get_env(*args, **kwargs)
        original_step, original_close = env.step, env.close
        observed_steps = [0]

        def step(action, *step_args, **step_kwargs):
            observed_steps[0] += 1
            values = np.asarray(action, dtype=np.float64)
            finite = np.isfinite(values)
            finite_values = values[finite]
            event("env_step_begin", env_step=observed_steps[0], action_finite=bool(finite.all()),
                  action_shape=list(values.shape), action_nonfinite_count=int((~finite).sum()),
                  action_finite_min=float(finite_values.min()) if finite_values.size else None,
                  action_finite_max=float(finite_values.max()) if finite_values.size else None)
            # Pass the exact original action object; no clipping/casting/replacement.
            result = original_step(action, *step_args, **step_kwargs)
            event("env_step_end", env_step=observed_steps[0])
            return result

        def close(*close_args, **close_kwargs):
            event("env_close_begin", observed_env_steps=observed_steps[0])
            result = original_close(*close_args, **close_kwargs)
            event("env_close_end", observed_env_steps=observed_steps[0])
            return result

        env.step, env.close = step, close
        event("environment_created", description=description)
        return env, description

    def rollout(*args, **kwargs):
        event("rollout_begin", max_action_steps=kwargs.get("max_steps"),
              waiting_steps=kwargs.get("num_steps_wait"))
        try:
            result = original_rollout(*args, **kwargs)
        except BaseException as exc:
            event("rollout_exception", exception=repr(exc))
            raise
        event("rollout_return", success=bool(result[0]), frame_count=len(result[1]))
        return result

    worker._get_libero_env = get_env
    worker.rollout_episode = rollout


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--repo", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    parallel = args.repo.resolve() / "examples/simBenchmarks/LIBERO-plus/eval_files/parallel_eval"
    sys.path.insert(0, str(parallel))
    faulthandler.enable(all_threads=True)
    import eval_worker
    attach_observers(eval_worker)
    sys.argv = [str(parallel / "eval_worker.py")] + remaining
    event("worker_begin", original_worker=str(parallel / "eval_worker.py"))
    eval_worker.main()
    event("worker_return")


if __name__ == "__main__":
    main()
