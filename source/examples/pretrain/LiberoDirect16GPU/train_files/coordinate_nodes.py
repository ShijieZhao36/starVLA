#!/usr/bin/env python3
"""Bounded shared-file barriers and supervision of this launcher's own child group."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temp, path)


def failure(args):
    for p in Path(args.run).glob('launch/failure_node*.json'):
        return f'Node failure: {p}: {p.read_text()}'
    for p in Path(args.launcher).glob('node*/failure.json'):
        return f'Launcher failure: {p}: {p.read_text()}'
    for p in Path(args.launcher).glob('node*/exit_code'):
        if p.read_text().strip() not in ('', '0'):
            return f'Launcher exited unsuccessfully: {p}'
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['wait', 'run'])
    p.add_argument('--run', required=True)
    p.add_argument('--launcher', required=True)
    p.add_argument('--node-rank', type=int, choices=[0, 1], required=True)
    p.add_argument('--timeout', type=float, required=True)
    p.add_argument('--path', action='append', default=[])
    p.add_argument('--log')
    argv = sys.argv[1:]
    separator = argv.index('--') if '--' in argv else len(argv)
    args = p.parse_args(argv[:separator])
    command = argv[separator + 1:]
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
                raise ValueError('run requires --log and a command after --')
            log = open(args.log, 'x')
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        elif not args.path:
            raise ValueError('wait requires at least one --path')
        while True:
            error = failure(args)
            if error:
                raise RuntimeError(error)
            if args.mode == 'wait' and all(Path(x).is_file() for x in args.path):
                return
            if child is not None and child.poll() is not None:
                if child.returncode != 0:
                    raise RuntimeError(f'Owned child exited {child.returncode}: {args.log}')
                return
            if time.monotonic() - start > args.timeout:
                raise TimeoutError(f'{args.mode} exceeded {args.timeout}s')
            time.sleep(2)
    except BaseException as exc:
        stop_owned_child(child)
        atomic_json(Path(args.launcher) / f'node{args.node_rank}/failure.json',
                    {'node_rank': args.node_rank, 'run': args.run, 'error': str(exc), 'time': time.time()})
        raise
    finally:
        if log is not None:
            log.close()


if __name__ == '__main__':
    main()
