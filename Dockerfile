FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt requirements-local.txt ./

# CPU wheel: the PyPI torch bundles ~3GB of CUDA libs this container cannot use.
# Own layer so a later failure does not re-download it.
RUN pip install --no-cache-dir --timeout 120 --retries 10 \
    torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir --timeout 120 --retries 10 -r requirements-local.txt

COPY . .

RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/logs \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Uses stdlib urllib rather than curl, which slim does not ship.
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
