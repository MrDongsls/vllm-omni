#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""nsys profile runs for the PersonaPlex shared-Mimi evidence (RFC #7389 §6).

Profiles one server per (cell, sessions) using the documented Omni capture
flow (docs/contributing/profiling.md §6): the server is wrapped in
``nsys profile --capture-range=cudaProfilerApi`` with per-stage
``profiler: cuda`` configs, and the serving API's /start_profile and
/stop_profile bracket exactly one paced load run — model init and idle time
stay outside the trace. ``--cuda-graph-trace=node`` keeps kernels inside
CUDA-graph replays attributed (required for the graphed cells B/D).
Companion to ``bench_personaplex_shared_mimi_matrix.py``.

Typical: python profile_personaplex_nsys.py --cells A D --sessions 1 8
"""

import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_personaplex_shared_mimi_matrix as bm  # noqa: E402 (sibling module; needs sys.path above)


def _profile_api(url: str, action: str) -> bool:
    response = httpx.post(f"{url}/{action}", json={"stages": [0, 1]}, timeout=120.0)
    print(f"  /{action} -> {response.status_code} {response.text[:120]}", flush=True)
    return response.status_code == 200


def profile_cell(args: argparse.Namespace, cell: str, sessions: int, out_dir: Path) -> bool:
    shared, graphs = bm.CELLS[cell]
    tag = f"{cell.lower()}_n{sessions}"
    cell_dir = out_dir / tag
    cell_dir.mkdir(parents=True, exist_ok=True)
    log = cell_dir / "server.log"
    report = cell_dir / f"nsys_{tag}"
    deploy = REPO / "vllm_omni/deploy" / f"_autogen_pplex_nsys_{tag}.yaml"
    bm.write_deploy_overlay(
        deploy, sessions, shared, graphs, args.gpu_mem_util_0, args.gpu_mem_util_1, profiler_cuda=True
    )

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
    # Documented Nsight Systems wrapping (docs/contributing/profiling.md §6):
    # cudaProfilerApi capture range + node-level CUDA-graph tracing.
    nsys_cmd = [
        "nsys",
        "profile",
        "--trace=cuda,nvtx,osrt",
        "--sample=process-tree",
        "--cpuctxsw=process-tree",
        "--trace-fork-before-exec=true",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop-shutdown",
        "--cuda-graph-trace=node",
        "-o",
        str(report),
        "--force-overwrite",
        "true",
        *server_cmd,
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
        started = _profile_api(url, "start_profile")
        driver_args = argparse.Namespace(
            model=args.model,
            input_wav=args.input_wav,
            max_client_rtf=None,
            expected_input_sha256=None,
            driver_timeout_s=args.driver_timeout_s,
        )
        try:
            bm.run_driver(
                driver_args,
                f"ws://127.0.0.1:{args.port}/v1/realtime?duplex=1",
                sessions,
                args.load_frames,
                cell_dir / "run01",
            )
            driver_ok = True
        except Exception as exc:
            print(f"  driver failed: {exc}", flush=True)
            driver_ok = False
        stopped = _profile_api(url, "stop_profile") if started else False
        # The report is only valid when both profile calls succeeded.
        ok = started and stopped and driver_ok
        print(f"  report: {report}.nsys-rep (valid={ok})", flush=True)
        return ok
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            proc.terminate()
        try:
            proc.wait(timeout=90)
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
