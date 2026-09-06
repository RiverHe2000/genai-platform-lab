# Gateway image (CPU; the models live behind it). Multi-stage, non-root, health-checked.
FROM python:3.12-slim AS build
WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --upgrade pip \
 && pip wheel --no-cache-dir --no-deps -w /wheels . \
 && pip wheel --no-cache-dir -w /wheels "uvicorn[standard]>=0.30"

FROM python:3.12-slim
RUN useradd --create-home --uid 10001 gateway
WORKDIR /app
COPY --from=build /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels
COPY deploy/gateway.fake.yaml deploy/gateway.vllm.yaml /app/deploy/
USER gateway
ENV LLMGATE_CONFIG_PATH=/app/deploy/gateway.vllm.yaml \
    LLMGATE_HOST=0.0.0.0 \
    LLMGATE_PORT=8080
EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=3s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=2).status == 200 else 1)"
CMD ["sh", "-c", "llmgate serve --config ${LLMGATE_CONFIG_PATH} --host ${LLMGATE_HOST} --port ${LLMGATE_PORT}"]
