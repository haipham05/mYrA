ARG BASE_API_IMAGE=myra-api:latest
FROM ${BASE_API_IMAGE}

WORKDIR /app
COPY apps/api/translation-engine/requirements.txt /opt/myra-translation/requirements.txt
RUN python -m venv /opt/myra-translation/.venv \
    && /opt/myra-translation/.venv/bin/pip --timeout 120 install --require-hashes \
        --no-cache-dir -r /opt/myra-translation/requirements.txt
ENV MYRA_TRANSLATION_ENGINE_PYTHON=/opt/myra-translation/.venv/bin/python \
    MYRA_TRANSLATION_ENGINE_RUNNER=/app/app/services/translation/engine_runner.py \
    MYRA_TRANSLATION_LAYOUT_MODEL=/tmp/.cache/babeldoc/models/doclayout_yolo_docstructbench_imgsz1024.onnx \
    HOME=/tmp

CMD ["python", "-m", "app.translation_worker"]
