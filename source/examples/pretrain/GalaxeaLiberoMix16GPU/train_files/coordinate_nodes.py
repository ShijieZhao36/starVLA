#!/usr/bin/env python3
"""Bounded shared-file barriers; stop only this launcher's own child process group."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(value)
    os.replace(tmp, path)


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + '\n')


def failure(args):
    for pattern in ('launch/failure_node*.json',):
        for path in Path(args.run).glob(pattern):
            return f'Node failure: {path}: {path.read_text()}'
    for path in Path(args.launcher).glob('node*/failure.json'):
        return f'Launcher failure: {path}: {path.read_text()}'
    for path in Path(args.launcher).glob('node*/exit_code'):
        if path.read_text().strip() not in ('', '0'):
            return f'Launcher exited unsuccessfully: {path}'
    return None


def stop_owned_child(proc):
    if proc is None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        proc.poll()
        try:
            os.killpg(proc.pid, 0)
        except ProcessLookupError:
            return
        time.sleep(.2)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('wait', 'run', 'mark'))
    parser.add_argument('--run', required=True)
    parser.add_argument('--launcher', required=True)
    parser.add_argument('--node-rank', type=int, choices=(0, 1), required=True)
    parser.add_argument('--timeout', type=float, default=600)
    parser.add_argument('--path', action='append', default=[])
    parser.add_argument('--log')
    parser.add_argument('--value', default='0\n')
    argv = sys.argv[1:]
    split = argv.index('--') if '--' in argv else len(argv)
    args = parser.parse_args(argv[:split])
    command = argv[split + 1:]
    if args.mode == 'mark':
        if len(args.path) != 1:
            parser.error('mark needs exactly one --path')
        atomic_text(args.path[0], args.value)
        return
    child = None
    log = None
    start = time.monotonic()

    def interrupted(sig, frame):
        raise RuntimeError(f'Interrupted by signal {sig}')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        if args.mode == 'run':
            if not command or not args.log:
                raise ValueError('run needs --log and a command after --')
            log = open(args.log, 'x')
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        elif not args.path:
            raise ValueError('wait needs at least one --path')
        while True:
            error = failure(args)
            if error:
                raise RuntimeError(error)
            if args.mode == 'wait' and all(Path(path).is_file() for path in args.path):
                return
            if child is not None and child.poll() is not None:
                if child.returncode:
                    raise RuntimeError(f'Owned child exited {child.returncode}: {args.log}')
                return
            if time.monotonic() - start > args.timeout:
                raise TimeoutError(f'{args.mode} exceeded {args.timeout}s')
            time.sleep(2)
    except BaseException as exc:
        stop_owned_child(child)
        atomic_json(Path(args.launcher) / f'node{args.node_rank}/failure.json',
                    dict(node_rank=args.node_rank, run=args.run, error=str(exc), time=time.time()))
        raise
    finally:
        if log is not None:
            log.close()


if __name__ == '__main__':
    main()
