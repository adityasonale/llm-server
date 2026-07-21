# Matches the interpreter used in development (3.10.11).
# "slim" keeps the base small; the full image adds ~700MB of build tooling
# we do not need because every dependency here ships a manylinux wheel.
FROM python:3.10-slim

# Never write .pyc files, and don't buffer stdout — otherwise logs appear
# only when the buffer flushes, which makes `docker logs` look frozen.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies are copied and installed BEFORE the source. Docker caches each
# layer, so editing a .py file does not invalidate the pip install layer and
# rebuilds stay fast. Copying everything first would reinstall on every edit.
COPY requirements.txt requirements-local.txt ./
RUN pip install --no-cache-dir -r requirements-local.txt

# NOTE: this installs the full local stack, which makes the image several GB.
# It is required because server.py imports providers.local.services at module
# load, so the process cannot start without torch even when only Groq is used.
# Installing just requirements.txt here needs those imports deferred first.
# The torch installed here is the CPU build — no GPU passthrough is configured,
# so the "local" provider will run on CPU (slowly) inside this container.

COPY . .

# Run as a non-root user: a process that does not need root should not have it.
# The logs directory is created and owned before dropping privileges, since the
# rotating file handler in utils/logger.py writes there at startup.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/logs \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Uses stdlib urllib rather than curl, which slim does not ship.
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()"

# 0.0.0.0, not 127.0.0.1 — binding to loopback inside the container would make
# the port unreachable from the host even with -p published.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
