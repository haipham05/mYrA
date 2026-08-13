ARG BASE_TRANSLATION_IMAGE=myra-api:latest
FROM ${BASE_TRANSLATION_IMAGE}

WORKDIR /app
ARG REUSE_TRANSLATION_ENV=0
COPY apps/api/translation-engine/requirements.txt /opt/myra-translation/requirements.txt
COPY apps/api/translation-engine/nllb-requirements.txt /opt/myra-translation/nllb-requirements.txt
RUN if [ "$REUSE_TRANSLATION_ENV" = "1" ] && [ -x /opt/myra-translation/.venv/bin/pip ]; then \
        /opt/myra-translation/.venv/bin/pip install --timeout 120 \
            --extra-index-url https://download.pytorch.org/whl/cu126 \
            --require-hashes --no-deps --no-cache-dir \
            -r /opt/myra-translation/nllb-requirements.txt; \
    else \
        python -m venv /opt/myra-translation/.venv \
        && /opt/myra-translation/.venv/bin/pip install --timeout 120 \
            --extra-index-url https://download.pytorch.org/whl/cu126 \
            --require-hashes --no-cache-dir \
            -r /opt/myra-translation/requirements.txt; \
    fi
COPY infrastructure/translation/setup_nllb.py /opt/myra-translation/setup_nllb.py
ENV MYRA_TRANSLATION_ENGINE_PYTHON=/opt/myra-translation/.venv/bin/python \
    MYRA_TRANSLATION_ENGINE_RUNNER=/app/app/services/translation/engine_runner.py \
    MYRA_TRANSLATION_MODEL_PATH=/opt/myra-translation/nllb \
    MYRA_TRANSLATION_LAYOUT_MODEL=/tmp/.cache/babeldoc/models/doclayout_yolo_docstructbench_imgsz1024.onnx \
    HOME=/tmp

CMD ["python", "-m", "app.translation_worker"]
