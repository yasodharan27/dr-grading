# CPU image of the web demo and the command-line tools. Build from the repository root:
#   docker build -f app/Dockerfile -t drgrade:1.0.0 .
# The model weights are not in the image: mount them read-only at /weights (or set DRGRADE_DOWNLOAD_WEIGHTS=1).

# ---- build stage: a virtual environment with the pinned CPU wheels
FROM python:3.11.9-slim AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0 torchvision==0.29.0
COPY app/requirements-docker.txt /tmp/requirements-docker.txt
RUN pip install -r /tmp/requirements-docker.txt \
 && find /opt/venv -name "__pycache__" -type d -prune -exec rm -rf {} + \
 && find /opt/venv -path "*/tests/*" -name "*.py" -delete

# ---- runtime stage
FROM python:3.11.9-slim
RUN useradd --create-home --uid 1000 app
COPY --from=build /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    HOST=0.0.0.0 PORT=8501 \
    DRGRADE_WEIGHTS_DIR=/weights DRGRADE_OUTPUT_DIR=/tmp/drgrade DRGRADE_DEMO_IMAGES=/app/data/demo_images \
    CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=2 MPLCONFIGDIR=/tmp/matplotlib \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
WORKDIR /app
COPY drgrade/ drgrade/
COPY app/server.py app/page.html app/entrypoint.sh app/
COPY scripts/ scripts/
COPY data/ data/
COPY config.yaml weights_manifest.json pyproject.toml ./
COPY LICENSES/ LICENSES/
RUN chmod 755 app/entrypoint.sh && mkdir -p /weights && chown app:app /weights
USER app
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/' % os.environ.get('PORT', '8501'), timeout=8)"
CMD ["app/entrypoint.sh"]
