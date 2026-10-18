#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Recycled-session soak for the PersonaPlex shared decoder (RFC #7389 §6).

Holds one server (default: cell D at N=8) under back-to-back paced load
sessions for a wall-clock budget (default 600 s). Single sessions cannot run
past the 3000-frame window until W6 lands, so the 10-minute soak recycles
sessions instead: steady-state server behavior (memory trend, Stage 1 decode
step latency drift) is what the RFC asks the soak to demonstrate. GPU memory
is sampled every 10 s; the periodic Mimi stats lines in the server log give
the decode-step latency and graph hit rate over time.

Typical: python soak_personaplex_shared_mimi.py --cell D --sessions 8
"""

import argparse
import csv
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_personaplex_shared_mimi_matrix as bm  # noqa: E402 (sibling module; needs sys.path above)

SAMPLE_INTERVAL_S = 10.0


def sample_gpu_memory(stop: threading.Event, rows: list[tuple[float, float]]) -> None:
    while not stop.wait(SAMPLE_INTERVAL_S):
        try:
            used = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout
            rows.append((time.time(), float(used.strip().splitlines()[0])))
        except Exception:
            continue


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="/root/models")
    p.add_argument("--input-wav", type=Path, default=Path("/root/personaplex-input-pcm.wav"))
    p.add_argument("--cell", choices=sorted(bm.CELLS), default="D")
    p.add_argument("--sessions", type=int, default=8)
    p.add_argument("--budget-s", type=float, default=600.0)
    p.add_argument("--session-frames", type=int, default=1500, help="Frames per recycled session (1500 = 120 s).")
    p.add_argument("--gpu-mem-util-0", type=float, default=0.75)
    p.add_argument("--gpu-mem-util-1", type=float, default=0.15)
    p.add_argument("--port", type=int, default=8099)
    p.add_argument("--server-timeout-s", type=float, default=1800.0)
    p.add_argument("--driver-timeout-s", type=float, default=900.0)
    p.add_argument("--cuda-visible-devices", default="0")
    p.add_argument("--output-dir", default="/root/pplex_soak")
    args = p.parse_args()

    shared, graphs = bm.CELLS[args.cell]
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    log = out / "server.log"
    deploy = REPO / "vllm_omni/deploy" / f"_autogen_pplex_soak_{args.cell.lower()}_n{args.sessions}.yaml"
    bm.write_deploy_overlay(deploy, args.sessions, shared, graphs, args.gpu_mem_util_0, args.gpu_mem_util_1)

    cmd = [
        sys.executable,
        "-m",
        "vllm_omni.entrypoints.cli.main",
        "serve",
        args.model,
        "--omni",
        "--deploy-config",
        str(deploy),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO), env.get("PYTHONPATH")) if p)
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    print(f"=== soak cell {args.cell} sessions={args.sessions} budget={args.budget_s:.0f}s ===", flush=True)
    with open(log, "w") as lf:
        proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
    url = f"http://127.0.0.1:{args.port}"
    samples: list[tuple[float, float]] = []
    stop = threading.Event()
    sampler = threading.Thread(target=sample_gpu_memory, args=(stop, samples), daemon=True)
    try:
        ready_s = bm.wait_ready(url, proc, log, args.server_timeout_s)
        print(f"  server ready in {ready_s:.1f}s", flush=True)
        sampler.start()
        driver_args = argparse.Namespace(
            model=args.model,
            input_wav=args.input_wav,
            max_client_rtf=None,
            expected_input_sha256=None,
            driver_timeout_s=args.driver_timeout_s,
        )
        deadline = time.monotonic() + args.budget_s
        run = 0
        while time.monotonic() < deadline:
            run += 1
            bm.run_driver(
                driver_args,
                f"ws://127.0.0.1:{args.port}/v1/realtime?duplex=1",
                args.sessions,
                min(args.session_frames, max(1, int((deadline - time.monotonic()) / 0.08))),
                out / f"run{run:02d}",
            )
            print(f"  soak run {run} done ({time.monotonic() - (deadline - args.budget_s):.0f}s elapsed)", flush=True)
    finally:
        stop.set()
        sampler.join(timeout=SAMPLE_INTERVAL_S * 2)
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                proc.kill()
            proc.wait(timeout=20)
        bm.wait_port_free(args.port)
        deploy.unlink(missing_ok=True)

    with open(out / "gpu_memory.csv", "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["timestamp_epoch", "gpu_memory_mib"])
        writer.writerows(samples)
    if samples:
        values = [mib for _, mib in samples]
        print(
            f"gpu memory MiB over {len(values)} samples: first={values[0]:.0f} last={values[-1]:.0f} "
            f"mean={sum(values) / len(values):.0f} max={max(values):.0f} drift={values[-1] - values[0]:+.0f}",
            flush=True,
        )
    stats_lines = [line for line in log.read_text(errors="replace").splitlines() if "hit_rate=" in line]
    print(f"periodic Mimi stats lines: {len(stats_lines)}", flush=True)
    for line in stats_lines:
        print(f"  {line.split(' - ', 1)[-1].strip()}", flush=True)
    print(f"soak artifacts: {out}", flush=True)


if __name__ == "__main__":
    main()
