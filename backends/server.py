"""
Base for backends that run as their own OpenAI-compatible HTTP server
(llama-server, mlx_lm.server, vLLM): launch the process, wait until it is
healthy, then forward requests through the existing pass-through provider.

subprocess.Popen rather than asyncio subprocesses on purpose: asyncio's
version needs a Proactor event loop on Windows, which uvicorn does not always
use, and the NVIDIA laptop runs Windows.
"""
import asyncio
import os
import socket
import subprocess
import time
from collections.abc import AsyncIterator
from pathlib import Path

import httpx

from backends.base import Backend, ModelSpec
from providers.openai_compatible import OpenAICompatibleProvider, UpstreamError
from settings import UPSTREAM_TIMEOUT_SECONDS
from utils.logger import get_logger

logger = get_logger(__name__)

HOST = "127.0.0.1"
LOG_DIR = Path("logs") / "backends"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind((HOST, 0))
        return s.getsockname()[1]


def _tail(path: Path, lines: int = 15) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


class ServerBackend(Backend):
    # Polled until it answers 200. llama-server answers 503 while loading.
    health_path = "/health"

    def __init__(self, settings: dict | None = None):
        super().__init__(settings)
        self._proc: subprocess.Popen | None = None
        self._log_file = None
        self._upstream: OpenAICompatibleProvider | None = None
        self._spec: ModelSpec | None = None

    def command(self, spec: ModelSpec, host: str, port: int) -> list[str]:
        raise NotImplementedError

    def upstream_model(self, spec: ModelSpec) -> str:
        """The `model` value the server expects in each request."""
        return spec.name

    async def start(self, spec: ModelSpec) -> None:
        self._spec = spec
        port = spec.port or _free_port()
        cmd = [*self.command(spec, HOST, port), *spec.args]

        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = LOG_DIR / f"{spec.name}.log"
        self._log_file = open(log_path, "ab")
        self._log_file.write(f"\n==== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n".encode())
        self._log_file.flush()

        logger.info("Starting %s for %s on port %d (log: %s)", self.name, spec.name, port, log_path)
        try:
            self._proc = subprocess.Popen(
                cmd, stdout=self._log_file, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                # Own process group on POSIX, so a Ctrl+C aimed at uvicorn does
                # not also kill the backend before stop() can shut it down cleanly.
                start_new_session=os.name == "posix",
            )
        except OSError as e:
            await self.stop()
            raise UpstreamError(502, f"Could not launch {self.name} for '{spec.name}': {e}")

        base = f"http://{HOST}:{port}"
        deadline = time.monotonic() + spec.startup_timeout
        async with httpx.AsyncClient(timeout=5.0) as client:
            while True:
                if self._proc.poll() is not None:
                    code = self._proc.returncode
                    await self.stop()
                    raise UpstreamError(
                        502,
                        f"{self.name} exited with code {code} while loading '{spec.name}'. "
                        f"Last log lines ({log_path}):\n{_tail(log_path)}",
                    )
                try:
                    if (await client.get(base + self.health_path)).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                if time.monotonic() > deadline:
                    await self.stop()
                    raise UpstreamError(
                        504, f"{self.name} did not become healthy within {spec.startup_timeout:.0f}s for '{spec.name}'."
                    )
                await asyncio.sleep(0.5)

        self._upstream = OpenAICompatibleProvider(spec.name, base + "/v1", None, UPSTREAM_TIMEOUT_SECONDS)
        logger.info("%s ready for %s (pid %d)", self.name, spec.name, self._proc.pid)

    async def stop(self) -> None:
        if self._upstream is not None:
            await self._upstream.aclose()
            self._upstream = None
        if self._proc is not None and self._proc.poll() is None:
            logger.info("Stopping %s (pid %d)", self.name, self._proc.pid)
            self._proc.terminate()
            try:
                await asyncio.to_thread(self._proc.wait, 20)
            except subprocess.TimeoutExpired:
                logger.warning("%s did not exit after SIGTERM; killing", self.name)
                self._proc.kill()
                await asyncio.to_thread(self._proc.wait)
        self._proc = None
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    def _body(self, body: dict) -> dict:
        if self._upstream is None or self._proc is None or self._proc.poll() is not None:
            raise UpstreamError(502, f"{self.name} for '{self._spec.name if self._spec else '?'}' is not running.")
        return {**body, "model": self.upstream_model(self._spec)}

    async def chat_completion(self, body: dict) -> dict:
        body = self._body(body)  # before touching _upstream: it raises when not running
        return await self._upstream.chat_completion(body)

    async def open_stream(self, body: dict) -> AsyncIterator[bytes]:
        body = self._body(body)
        return await self._upstream.open_stream(body)
