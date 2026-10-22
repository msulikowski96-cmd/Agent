FROM python:3.12-slim
# Narzędzia, z których agent może korzystać przy budowaniu aplikacji
RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl ca-certificates nodejs npm build-essential zip unzip \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . /app
ENV DATA_DIR=/data PORT=8080 PYTHONUNBUFFERED=1
RUN mkdir -p /data
EXPOSE 8080
CMD ["python", "app.py"]
