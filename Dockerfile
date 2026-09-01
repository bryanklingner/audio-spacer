FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY audio_spacer.py server.py ./
COPY static ./static

ENV SPACER_DATA=/data
EXPOSE 8000

# The app already serves /api/health; without a HEALTHCHECK, Docker reports the
# container as "Up" even when uvicorn is wedged, so orchestration and dashboards
# show green for a service that is not answering.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status == 200 else 1)"
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
