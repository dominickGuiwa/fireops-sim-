# syntax=docker/dockerfile:1.7

FROM python:3.14-slim AS backend-builder

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

RUN python -m venv "${VIRTUAL_ENV}"
COPY backend /tmp/fireops-backend
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir /tmp/fireops-backend


FROM python:3.14-slim AS backend

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN groupadd --gid 10001 fireops \
    && useradd --uid 10001 --gid fireops --create-home --shell /usr/sbin/nologin fireops

COPY --from=backend-builder /opt/venv /opt/venv
WORKDIR /app

USER fireops
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


FROM node:22-alpine AS frontend-dependencies

WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci


FROM frontend-dependencies AS frontend-builder

ARG VITE_API_URL=http://localhost:8000
ENV VITE_API_URL=${VITE_API_URL}

COPY frontend ./
RUN npm run build


FROM nginxinc/nginx-unprivileged:1.30.2-alpine AS frontend

COPY --from=frontend-builder /app/frontend/dist /usr/share/nginx/html

USER nginx
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget -qO- http://127.0.0.1:8080/ >/dev/null || exit 1
