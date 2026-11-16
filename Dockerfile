FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml ./
COPY search ./search
COPY discoverylib ./discoverylib
RUN pip install --no-cache-dir .
ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 SEARCH_DATA_DIR=/cache SEARCH_ARTIFACT_DIR=/runs
EXPOSE 8000
CMD ["uvicorn", "search.service:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
