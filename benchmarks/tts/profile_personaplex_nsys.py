#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""nsys profile runs for the PersonaPlex shared-Mimi evidence (RFC #7389 §6).

Profiles one server per (cell, sessions) with CUDA activity + OS runtime
trace. Collection is bracketed deterministically: the server starts under
``nsys launch`` (no collection), the paced load run begins, collection starts
a few seconds into the run, and stops when the run ends — so the capture is
steady-state decode and never weight load. Companion to
``bench_personaplex_shared_mimi_matrix.py``; reuses its overlay and
readiness helpers.

Typical: python profile_personaplex_nsys.py --cells A D --sessions 1 8
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_personaplex_shared_mimi_matrix as bm  # noqa: E402 (sibling module; needs sys.path above)

SETTLE_S = 8.0  # streaming is established before collection starts


def _nsis_ok(cmd: list[str]) -> bool:
    completed = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if completed.returncode != 0:
        print(f"  nsys command failed: {' '.join(cmd)}\n{completed.stdout}{completed.stderr}", flush=True)
        return False
    return True


def profile_cell(args: argparse.Namespace, cell: str, sessions: int, out_dir: Path) -> bool:
    shared, graphs = bm.CELLS[cell]
    tag = f"{cell.lower()}_n{sessions}"
    cell_dir = out_dir / tag
    cell_dir.mkdir(parents=True, exist_ok=True)
    log = cell_dir / "server.log"
    report = cell_dir / f"nsys_{tag}"
    session_id = f"pprof_{tag}"
    deploy = REPO / "vllm_omni/deploy" / f"_autogen_pplex_nsys_{tag}.yaml"
    bm.write_deploy_overlay(deploy, sessions, shared, graphs, args.gpu_mem_util_0, args.gpu_mem_util_1)

    server_cmd = [
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
    nsys_cmd = [
        "nsys",
        "launch",
        f"--session-new={session_id}",
        "-o",
        str(report),
        "--trace=cuda,osrt,nvtx",
        "--sample=none",
        "--cpuctxsw=none",
        "--force-overwrite",
        "true",
        *server_cmd,
    ]
    driver_cmd = [
        sys.executable,
        str(bm.DRIVER),
        "--url",
        f"ws://127.0.0.1:{args.port}/v1/realtime?duplex=1",
        "--model",
        args.model,
        "--input-wav",
        str(args.input_wav),
        "--sessions",
        str(sessions),
        "--load-frames",
        str(args.load_frames),
        "--output-dir",
        str(cell_dir / "run01"),
        "--timeout-s",
        str(args.driver_timeout_s),
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO), env.get("PYTHONPATH")) if p)
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    print(f"\n=== nsys cell {cell} sessions={sessions} ===", flush=True)
    with open(log, "w") as lf:
        proc = subprocess.Popen(nsys_cmd, env=env, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
    url = f"http://127.0.0.1:{args.port}"
    try:
        ready_s = bm.wait_ready(url, proc, log, args.server_timeout_s)
        print(f"  server ready in {ready_s:.1f}s; starting paced load", flush=True)
        driver_log = open(cell_dir / "driver.log", "w")
        driver = subprocess.Popen(driver_cmd, env=env, stdout=driver_log, stderr=subprocess.STDOUT)
        time.sleep(SETTLE_S)
        started = _nsis_ok(["nsys", "start", f"--session={session_id}"])
        driver_rc = driver.wait(timeout=args.driver_timeout_s + 300)
        driver_log.close()
        if started:
            _nsis_ok(["nsys", "stop", f"--session={session_id}"])
        print(f"  driver rc={driver_rc}; report: {report}.nsys-rep", flush=True)
        return started and driver_rc == 0
    finally:
        subprocess.run(["nsys", "quit", f"--session={session_id}"], capture_output=True, timeout=120)
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
        deploy.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="/root/models")
    p.add_argument("--input-wav", type=Path, default=Path("/root/personaplex-input-pcm.wav"))
    p.add_argument("--cells", choices=sorted(bm.CELLS), nargs="+", default=["A", "D"])
    p.add_argument("--sessions", type=int, nargs="+", default=[1, 8])
    p.add_argument("--load-frames", type=int, default=375)
    p.add_argument("--gpu-mem-util-0", type=float, default=0.75)
    p.add_argument("--gpu-mem-util-1", type=float, default=0.15)
    p.add_argument("--port", type=int, default=8099)
    p.add_argument("--server-timeout-s", type=float, default=1800.0)
    p.add_argument("--driver-timeout-s", type=float, default=900.0)
    p.add_argument("--cuda-visible-devices", default="0")
    p.add_argument("--output-dir", default="/root/pplex_nsys")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    ok = True
    for cell in args.cells:
        for sessions in args.sessions:
            ok &= profile_cell(args, cell, sessions, out)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
