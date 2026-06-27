"""Best-effort system and accelerator resource logging."""

from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return output


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str))
        handle.write("\n")
        handle.flush()


def collect_environment(config_path: str | Path | None = None) -> dict[str, Any]:
    """Collect mostly static process and package metadata."""

    torch_info = _torch_info()
    return {
        "timestamp_utc": utc_now(),
        "config_path": None if config_path is None else str(config_path),
        "command": sys.argv,
        "cwd": os.getcwd(),
        "hostname": socket.gethostname(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "platform": platform.platform(),
        },
        "git": _git_info(),
        "torch": torch_info,
        "device": _device_summary(torch_info),
    }


def collect_system_info(output_dir: str | Path | None = None) -> dict[str, Any]:
    """Collect host-level static information and one immediate resource snapshot."""

    return {
        "timestamp_utc": utc_now(),
        "hostname": socket.gethostname(),
        "cpu": _cpu_static_info(),
        "memory": _memory_static_info(),
        "disk": _disk_info(output_dir),
        "gpu": _gpu_static_info(),
    }


def collect_resource_snapshot(
    output_dir: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
    started_at_monotonic: float | None = None,
) -> dict[str, Any]:
    """Collect one best-effort resource snapshot.

    The function is deliberately defensive: unavailable libraries, missing GPU
    drivers, and transient collection errors are encoded in the payload instead
    of being raised to the training process.
    """

    now = time.monotonic()
    snapshot: dict[str, Any] = {
        "timestamp_utc": utc_now(),
        "elapsed_seconds": None if started_at_monotonic is None else now - started_at_monotonic,
        "hostname": socket.gethostname(),
        "metadata": metadata or {},
    }
    snapshot["cpu"] = _cpu_runtime_info()
    snapshot["memory"] = _memory_runtime_info()
    snapshot["gpu"] = _gpu_runtime_info()
    snapshot["torch_cuda"] = _torch_cuda_memory_info()
    snapshot["disk"] = _disk_info(output_dir)
    return snapshot


class ResourceMonitor:
    """Background JSONL resource sampler.

    Sampling failures are written as records in the JSONL file and never raised
    out of the monitor thread.
    """

    def __init__(
        self,
        output_path: str | Path,
        output_dir: str | Path,
        interval_seconds: float = 60.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.output_path = Path(output_path)
        self.output_dir = Path(output_dir)
        self.interval_seconds = max(float(interval_seconds), 1.0)
        self._base_metadata = dict(metadata or {})
        self._context: dict[str, Any] = {}
        self._context_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at_monotonic: float | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._started_at_monotonic = time.monotonic()
        _prime_psutil_cpu_percent()
        self.sample_once(reason="start")
        self._thread = threading.Thread(target=self._run, name="resource-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=self.interval_seconds + 5.0)
        self.sample_once(reason="stop")
        self._thread = None

    def set_context(self, **context: Any) -> None:
        with self._context_lock:
            self._context = {key: value for key, value in context.items() if value is not None}

    def clear_context(self) -> None:
        self.set_context()

    def sample_once(self, *, reason: str = "manual") -> None:
        metadata = self._metadata(reason=reason)
        try:
            record = collect_resource_snapshot(
                self.output_dir,
                metadata=metadata,
                started_at_monotonic=self._started_at_monotonic,
            )
        except Exception as exc:  # pragma: no cover - defensive guard.
            record = {
                "timestamp_utc": utc_now(),
                "elapsed_seconds": None,
                "hostname": socket.gethostname(),
                "metadata": metadata,
                "error": f"{type(exc).__name__}: {exc}",
            }
        append_jsonl(self.output_path, record)

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            self.sample_once(reason="interval")

    def _metadata(self, *, reason: str) -> dict[str, Any]:
        with self._context_lock:
            context = dict(self._context)
        return {**self._base_metadata, **context, "sample_reason": reason}


def _try_import_psutil() -> Any | None:
    try:
        import psutil
    except ModuleNotFoundError:
        return None
    except Exception:
        return None
    return psutil


def _prime_psutil_cpu_percent() -> None:
    psutil = _try_import_psutil()
    if psutil is None:
        return
    try:
        psutil.cpu_percent(interval=None)
    except Exception:
        return


def _cpu_static_info() -> dict[str, Any]:
    psutil = _try_import_psutil()
    info: dict[str, Any] = {
        "logical_count": os.cpu_count(),
        "physical_count": None,
        "frequency_mhz": None,
    }
    if psutil is None:
        info["status"] = "psutil_unavailable"
        return info
    try:
        info["physical_count"] = psutil.cpu_count(logical=False)
        freq = psutil.cpu_freq()
        info["frequency_mhz"] = None if freq is None else freq.current
        info["status"] = "available"
    except Exception as exc:
        info["status"] = "unavailable"
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def _cpu_runtime_info() -> dict[str, Any]:
    psutil = _try_import_psutil()
    if psutil is None:
        return {"status": "psutil_unavailable", "percent": None}
    try:
        return {
            "status": "available",
            "percent": psutil.cpu_percent(interval=None),
            "load_average": os.getloadavg() if hasattr(os, "getloadavg") else None,
        }
    except Exception as exc:
        return {"status": "unavailable", "percent": None, "error": f"{type(exc).__name__}: {exc}"}


def _memory_static_info() -> dict[str, Any]:
    psutil = _try_import_psutil()
    if psutil is None:
        return {"status": "psutil_unavailable", "total_bytes": None}
    try:
        memory = psutil.virtual_memory()
        return {"status": "available", "total_bytes": memory.total}
    except Exception as exc:
        return {"status": "unavailable", "total_bytes": None, "error": f"{type(exc).__name__}: {exc}"}


def _memory_runtime_info() -> dict[str, Any]:
    psutil = _try_import_psutil()
    if psutil is None:
        return {"status": "psutil_unavailable"}
    try:
        process = psutil.Process()
        virtual = psutil.virtual_memory()
        swap = psutil.swap_memory()
        return {
            "status": "available",
            "system_total_bytes": virtual.total,
            "system_available_bytes": virtual.available,
            "system_used_bytes": virtual.used,
            "system_percent": virtual.percent,
            "swap_total_bytes": swap.total,
            "swap_used_bytes": swap.used,
            "swap_percent": swap.percent,
            "process_rss_bytes": process.memory_info().rss,
            "process_vms_bytes": process.memory_info().vms,
        }
    except Exception as exc:
        return {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}


def _disk_info(output_dir: str | Path | None) -> dict[str, Any]:
    if output_dir is None:
        return {"status": "unavailable"}
    output = Path(output_dir)
    try:
        output.mkdir(parents=True, exist_ok=True)
        usage = shutil.disk_usage(output)
        return {
            "status": "available",
            "path": str(output),
            "filesystem_total_bytes": usage.total,
            "filesystem_used_bytes": usage.used,
            "filesystem_free_bytes": usage.free,
            "output_dir_size_bytes": _directory_size(output),
        }
    except Exception as exc:
        return {"status": "unavailable", "path": str(output), "error": f"{type(exc).__name__}: {exc}"}


def _directory_size(path: Path) -> int | None:
    total = 0
    try:
        for item in path.rglob("*"):
            if item.is_file():
                total += item.stat().st_size
    except Exception:
        return None
    return total


def _gpu_static_info() -> dict[str, Any]:
    runtime = _gpu_runtime_info()
    return {
        "backend": runtime.get("backend"),
        "status": runtime.get("status"),
        "device_count": len(runtime.get("devices") or []),
        "devices": [
            {
                "index": device.get("index"),
                "name": device.get("name"),
                "memory_total_mb": device.get("memory_total_mb"),
            }
            for device in runtime.get("devices") or []
        ],
        "error": runtime.get("error"),
    }


def _gpu_runtime_info() -> dict[str, Any]:
    for collector in (_gpu_from_pynvml, _gpu_from_nvidia_smi, _gpu_from_torch_cuda):
        payload = collector()
        if payload.get("status") == "available":
            return payload
    mps_available = _torch_mps_available()
    if mps_available:
        return {
            "status": "unavailable",
            "backend": "mps",
            "devices": [],
            "message": "Apple MPS is available, but utilization and memory metrics are unavailable.",
        }
    return {
        "status": "unavailable",
        "backend": "none",
        "devices": [],
        "message": "No NVIDIA GPU metrics backend is available.",
    }


def _gpu_from_pynvml() -> dict[str, Any]:
    try:
        import pynvml
    except ModuleNotFoundError:
        return {"status": "unavailable", "backend": "pynvml"}
    except Exception as exc:
        return {"status": "unavailable", "backend": "pynvml", "error": f"{type(exc).__name__}: {exc}"}
    try:
        pynvml.nvmlInit()
        devices = []
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            devices.append(
                {
                    "index": index,
                    "name": name,
                    "utilization_percent": utilization.gpu,
                    "memory_utilization_percent": utilization.memory,
                    "memory_used_mb": memory.used / (1024**2),
                    "memory_total_mb": memory.total / (1024**2),
                }
            )
        return {"status": "available", "backend": "pynvml", "devices": devices}
    except Exception as exc:
        return {"status": "unavailable", "backend": "pynvml", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def _gpu_from_nvidia_smi() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except FileNotFoundError:
        return {"status": "unavailable", "backend": "nvidia-smi"}
    except Exception as exc:
        return {
            "status": "unavailable",
            "backend": "nvidia-smi",
            "error": f"{type(exc).__name__}: {exc}",
        }
    if completed.returncode != 0:
        return {
            "status": "unavailable",
            "backend": "nvidia-smi",
            "error": completed.stderr.strip() or f"returncode={completed.returncode}",
        }
    devices = []
    for index, line in enumerate(completed.stdout.splitlines()):
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        devices.append(
            {
                "index": index,
                "name": parts[0],
                "utilization_percent": _to_float(parts[1]),
                "memory_used_mb": _to_float(parts[2]),
                "memory_total_mb": _to_float(parts[3]),
            }
        )
    return {"status": "available", "backend": "nvidia-smi", "devices": devices}


def _gpu_from_torch_cuda() -> dict[str, Any]:
    torch = _try_import_torch()
    if torch is None:
        return {"status": "unavailable", "backend": "torch_cuda"}
    try:
        if not torch.cuda.is_available():
            return {"status": "unavailable", "backend": "torch_cuda"}
        devices = []
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": props.name,
                    "utilization_percent": None,
                    "memory_used_mb": None,
                    "memory_total_mb": props.total_memory / (1024**2),
                }
            )
        return {
            "status": "available",
            "backend": "torch_cuda_memory_only",
            "devices": devices,
            "message": "GPU utilization unavailable; torch CUDA memory stats are logged separately.",
        }
    except Exception as exc:
        return {"status": "unavailable", "backend": "torch_cuda", "error": f"{type(exc).__name__}: {exc}"}


def _torch_cuda_memory_info() -> dict[str, Any]:
    torch = _try_import_torch()
    if torch is None:
        return {"status": "torch_unavailable"}
    try:
        if not torch.cuda.is_available():
            return {"status": "cuda_unavailable"}
        current = torch.cuda.current_device()
        return {
            "status": "available",
            "current_device": current,
            "memory_allocated_bytes": torch.cuda.memory_allocated(current),
            "memory_reserved_bytes": torch.cuda.memory_reserved(current),
            "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(current),
            "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(current),
        }
    except Exception as exc:
        return {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}


def _torch_info() -> dict[str, Any]:
    torch = _try_import_torch()
    if torch is None:
        return {"available": False}
    try:
        return {
            "available": True,
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": getattr(torch.version, "cuda", None),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "cudnn_version": (
                torch.backends.cudnn.version() if hasattr(torch.backends, "cudnn") else None
            ),
            "mps_available": _torch_mps_available(torch),
        }
    except Exception as exc:
        return {"available": True, "error": f"{type(exc).__name__}: {exc}"}


def _try_import_torch() -> Any | None:
    try:
        import torch
    except ModuleNotFoundError:
        return None
    except Exception:
        return None
    return torch


def _torch_mps_available(torch_module: Any | None = None) -> bool:
    torch = torch_module or _try_import_torch()
    if torch is None:
        return False
    try:
        return bool(
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        )
    except Exception:
        return False


def _device_summary(torch_info: dict[str, Any]) -> str:
    if torch_info.get("cuda_available"):
        return "cuda"
    if torch_info.get("mps_available"):
        return "mps"
    return "cpu"


def _git_info() -> dict[str, Any]:
    return {
        "commit": _git(["rev-parse", "HEAD"]),
        "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": _git(["status", "--porcelain"]) not in (None, ""),
    }


def _git(args: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _to_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
