"""Stateless action-chunk server. One process per GPU.

The returned chunk uses ``PolicyModel.infer_action_chunk`` (last-7 slice and
min/max unnormalization, including the gripper 0.5 threshold). It does not use
``PolicyServerWrapper``.
"""

import argparse
import logging

import numpy as np

from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
from policy_model import PolicyModel


class ChunkPolicy:
    def __init__(self, model: PolicyModel) -> None:
        import threading

        self._model = model
        self._lock = threading.Lock()

    def predict_action(self, example, **kwargs):
        del kwargs
        with self._lock:
            actions = self._model.infer_action_chunk(example)
        return {"actions": np.asarray(actions)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained-path", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--use-bf16", action="store_true")
    parser.add_argument("--use-canonical-forward", choices=("true", "false"), default="false")
    parser.add_argument("--gripper-open-sign", choices=("positive", "negative"), default="positive")
    parser.add_argument("--idle-timeout", type=int, default=-1)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, force=True)
    use_canonical = args.use_canonical_forward == "true"
    model = PolicyModel(
        policy_ckpt_path=args.pretrained_path,
        use_bf16=args.use_bf16,
        use_canonical_forward=use_canonical,
        gripper_open_sign=args.gripper_open_sign,
    )
    policy = ChunkPolicy(model)
    metadata = {"action_chunk_size": int(model.action_chunk_size)}
    logging.info("chunk server listening on %s:%s metadata=%s", args.host, args.port, metadata)
    server = WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        idle_timeout=args.idle_timeout,
        metadata=metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
