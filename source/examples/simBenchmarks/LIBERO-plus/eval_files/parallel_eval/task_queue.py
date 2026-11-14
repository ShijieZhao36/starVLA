"""One-episode task queue shared by LIBERO-plus sim workers."""

import json
import socket
import threading
from collections import deque
from pathlib import Path

CATEGORY_LABELS = (
    ("Camera Viewpoints", "Camera"),
    ("Robot Initial States", "Robot"),
    ("Language Instructions", "Language"),
    ("Light Conditions", "Light"),
    ("Background Textures", "Background"),
    ("Sensor Noise", "Noise"),
    ("Objects Layout", "Layout"),
)


def interleave_tasks(suite_counts: dict, suite_order, num_trials: int) -> list:
    """Round-robin suites so long libero_10 episodes are not a single tail."""
    indices = {name: 0 for name in suite_order}
    tasks = []
    while any(indices[name] < suite_counts[name] for name in suite_order):
        for name in suite_order:
            if indices[name] < suite_counts[name]:
                task_id = indices[name]
                indices[name] += 1
                for episode_idx in range(num_trials):
                    tasks.append((name, task_id, episode_idx))
    return tasks


class _LineSocket:
    def __init__(self, sock: socket.socket) -> None:
        self._sock = sock
        self._buf = b""

    def read_json(self):
        while b"\n" not in self._buf:
            chunk = self._sock.recv(1 << 20)
            if not chunk:
                return None
            self._buf += chunk
        line, self._buf = self._buf.split(b"\n", 1)
        if not line:
            return None
        return json.loads(line.decode("utf-8"))

    def write_json(self, payload: dict) -> None:
        data = json.dumps(payload).encode("utf-8") + b"\n"
        self._sock.sendall(data)


class TaskQueueServer:
    """Hand out one episode at a time. A dropped connection requeues the episode.

    ``max_attempts`` counts the first try. The default of 3 is the first try plus
    two retries.
    """

    def __init__(self, tasks: list, results_path: Path, max_attempts: int = 3) -> None:
        self._pending = deque((suite, task_id, episode_idx, 1) for suite, task_id, episode_idx in tasks)
        self._inflight = {}
        self._cond = threading.Condition()
        self._results_path = Path(results_path)
        self._results_path.parent.mkdir(parents=True, exist_ok=True)
        self._results_file = open(self._results_path, "a", encoding="utf-8")
        self.max_attempts = max_attempts
        self.total = len(tasks)
        self.completed = 0
        self._next_conn_id = 0
        self._sock = None

    def start(self, host: str, port: int) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        sock.listen(256)
        self._sock = sock
        thread = threading.Thread(target=self._accept_loop, name="task-queue", daemon=True)
        thread.start()

    def stop(self) -> None:
        sock = self._sock
        self._sock = None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
        with self._cond:
            self._cond.notify_all()
        self._results_file.flush()
        self._results_file.close()

    def progress(self) -> tuple:
        with self._cond:
            return self.completed, self.total, len(self._pending), len(self._inflight)

    def _accept_loop(self) -> None:
        while True:
            try:
                conn, _addr = self._sock.accept()
            except OSError:
                return
            with self._cond:
                self._next_conn_id += 1
                conn_id = self._next_conn_id
            threading.Thread(target=self._handle, args=(conn, conn_id), name=f"queue-{conn_id}", daemon=True).start()

    def _handle(self, conn: socket.socket, conn_id: int) -> None:
        client = _LineSocket(conn)
        try:
            while True:
                message = client.read_json()
                if message is None:
                    break
                op = message.get("op")
                if op == "next":
                    task = self._take(conn_id)
                    if task is None:
                        client.write_json({"ok": True, "done": True})
                        break
                    suite, task_id, episode_idx, attempt = task
                    client.write_json(
                        {
                            "ok": True,
                            "done": False,
                            "suite": suite,
                            "task_id": task_id,
                            "episode_idx": episode_idx,
                            "attempt": attempt,
                        }
                    )
                elif op == "result":
                    self._finish(conn_id, message)
                    client.write_json({"ok": True})
                elif op == "release":
                    retried = self._release(conn_id)
                    client.write_json({"ok": True, "retried": retried})
                else:
                    client.write_json({"ok": False, "error": f"unknown op {op}"})
        finally:
            self._drop(conn_id)
            try:
                conn.close()
            except OSError:
                pass

    def _take(self, conn_id: int):
        with self._cond:
            if conn_id in self._inflight:
                raise RuntimeError(f"worker {conn_id} requested a task while one is still in flight")
            while not self._pending and self._inflight:
                self._cond.wait(timeout=1.0)
            if self._pending:
                task = self._pending.popleft()
                self._inflight[conn_id] = task
                return task
            return None

    def _finish(self, conn_id: int, message: dict) -> None:
        with self._cond:
            task = self._inflight.pop(conn_id, None)
            if task is None:
                raise RuntimeError(f"worker {conn_id} reported a result without a task")
            suite, task_id, episode_idx, attempt = task
            if (message.get("suite"), message.get("task_id"), message.get("episode_idx")) != (
                suite,
                task_id,
                episode_idx,
            ):
                self._inflight[conn_id] = task
                raise RuntimeError(f"worker {conn_id} result does not match the assigned episode")
            record = {
                "suite": suite,
                "task_id": task_id,
                "episode_idx": episode_idx,
                "attempt": attempt,
                "success": bool(message.get("success")),
                "category": message.get("category"),
                "name": message.get("name"),
                "worker": message.get("worker"),
            }
            if message.get("error"):
                record["error"] = message["error"]
            self._write_record_locked(record)
            self.completed += 1
            self._cond.notify_all()

    def _release(self, conn_id: int) -> bool:
        with self._cond:
            task = self._inflight.pop(conn_id, None)
            if task is None:
                return False
            retried = self._requeue_locked(task, error="worker_released")
            self._cond.notify_all()
            return retried

    def _drop(self, conn_id: int) -> None:
        with self._cond:
            task = self._inflight.pop(conn_id, None)
            if task is None:
                return
            self._requeue_locked(task, error="worker_lost")
            self._cond.notify_all()

    def _requeue_locked(self, task, error: str) -> bool:
        suite, task_id, episode_idx, attempt = task
        if attempt < self.max_attempts:
            self._pending.append((suite, task_id, episode_idx, attempt + 1))
            return True
        self._write_record_locked(
            {
                "suite": suite,
                "task_id": task_id,
                "episode_idx": episode_idx,
                "attempt": attempt,
                "success": False,
                "category": None,
                "name": None,
                "worker": None,
                "error": error,
            }
        )
        self.completed += 1
        return False

    def _write_record_locked(self, record: dict) -> None:
        self._results_file.write(json.dumps(record) + "\n")
        self._results_file.flush()


class TaskQueueClient:
    def __init__(self, host: str, port: int) -> None:
        sock = socket.create_connection((host, port))
        self._client = _LineSocket(sock)
        self._sock = sock

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass

    def next_task(self):
        self._client.write_json({"op": "next"})
        reply = self._client.read_json()
        if reply is None or not reply.get("ok", False):
            raise RuntimeError(f"task queue next failed: {reply}")
        if reply.get("done"):
            return None
        return reply

    def report(self, record: dict) -> None:
        payload = dict(record)
        payload["op"] = "result"
        self._client.write_json(payload)
        reply = self._client.read_json()
        if reply is None or not reply.get("ok", False):
            raise RuntimeError(f"task queue result failed: {reply}")

    def release(self) -> None:
        self._client.write_json({"op": "release"})
        reply = self._client.read_json()
        if reply is None or not reply.get("ok", False):
            raise RuntimeError(f"task queue release failed: {reply}")


def summarize_results(results_path: Path) -> dict:
    per_category = {}
    per_suite = {}
    incomplete = []
    total_success = 0
    total_count = 0

    with open(results_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            suite = record.get("suite") or "unknown"
            category = record.get("category") or "unknown"
            success = bool(record.get("success"))
            suite_bucket = per_suite.setdefault(suite, {"total_count": 0, "success_count": 0, "categories": {}})
            cat_bucket = suite_bucket["categories"].setdefault(category, {"total_count": 0, "success_count": 0})
            global_bucket = per_category.setdefault(category, {"total_count": 0, "success_count": 0})
            suite_bucket["total_count"] += 1
            cat_bucket["total_count"] += 1
            global_bucket["total_count"] += 1
            total_count += 1
            if success:
                suite_bucket["success_count"] += 1
                cat_bucket["success_count"] += 1
                global_bucket["success_count"] += 1
                total_success += 1
            if record.get("error"):
                incomplete.append(record)

    def _rate(bucket):
        count = bucket["total_count"]
        return (bucket["success_count"] / count) if count else 0.0

    for bucket in per_category.values():
        bucket["success_rate"] = _rate(bucket)
    for bucket in per_suite.values():
        bucket["success_rate"] = _rate(bucket)
        for cat_bucket in bucket["categories"].values():
            cat_bucket["success_rate"] = _rate(cat_bucket)

    return {
        "per_category": per_category,
        "per_suite": per_suite,
        "total": {
            "total_count": total_count,
            "success_count": total_success,
            "success_rate": (total_success / total_count) if total_count else 0.0,
        },
        "incomplete": incomplete,
    }


def format_summary(summary: dict) -> str:
    lines = []
    per_category = summary["per_category"]
    seen = set()
    for category, label in CATEGORY_LABELS:
        bucket = per_category.get(category, {"total_count": 0, "success_count": 0, "success_rate": 0.0})
        lines.append(
            f"{label}: {bucket['success_count']}/{bucket['total_count']} ({100.0 * bucket.get('success_rate', 0.0):.1f}%)"
        )
        seen.add(category)
    for category, bucket in per_category.items():
        if category in seen:
            continue
        lines.append(
            f"{category}: {bucket['success_count']}/{bucket['total_count']} ({100.0 * bucket.get('success_rate', 0.0):.1f}%)"
        )
    total = summary["total"]
    lines.append(
        f"Total: {total['success_count']}/{total['total_count']} ({100.0 * total['success_rate']:.1f}%)"
    )
    lines.append("per suite:")
    for suite, bucket in summary["per_suite"].items():
        lines.append(
            f"  {suite}: {bucket['success_count']}/{bucket['total_count']} ({100.0 * bucket['success_rate']:.1f}%)"
        )
    if summary["incomplete"]:
        lines.append(f"incomplete episodes: {len(summary['incomplete'])}")
    return "\n".join(lines)
