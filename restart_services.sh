#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${ROOT_DIR}/logs"

BACKEND_PORT="${BACKEND_PORT:-8001}"
FRONTEND_PORT="${FRONTEND_PORT:-7860}"

# 未指定 PYTHON_BIN 时，优先选用已具备 fastapi+uvicorn 的解释器
if [[ -z "${PYTHON_BIN:-}" ]]; then
  for candidate in \
    "${HOME}/miniconda3/bin/python" "${HOME}/anaconda3/bin/python" \
    /root/miniconda3/bin/python /root/miniconda3/envs/llm-universe/bin/python \
    "$(command -v python3 2>/dev/null)" "$(command -v python 2>/dev/null)"; do
    [[ -z "${candidate}" ]] && continue
    if [[ -x "${candidate}" ]] || command -v "${candidate}" >/dev/null 2>&1; then
      if "${candidate}" -c "import fastapi, uvicorn" 2>/dev/null; then
        PYTHON_BIN="${candidate}"
        break
      fi
    fi
  done
fi
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "${LOG_DIR}"

if [[ "${SKIP_PIP_INSTALL:-0}" != "1" ]]; then
  echo "Installing/updating minimal deps (slowapi) with: ${PYTHON_BIN}"
  "${PYTHON_BIN}" -m pip install -q --disable-pip-version-check \
    "slowapi==0.1.9" 2>/dev/null \
    || echo "WARN: pip install failed; continuing with current environment"
fi

kill_pidfile() {
  local f="$1"
  [[ -f "${f}" ]] || return
  local pid
  pid="$(tr -d ' \n' < "${f}" || true)"
  [[ -n "${pid}" ]] || return
  if kill -0 "${pid}" 2>/dev/null; then
    echo "Stopping pid ${pid} (${f})"
    kill "${pid}" 2>/dev/null || true
    sleep 1
    kill -0 "${pid}" 2>/dev/null && kill -9 "${pid}" 2>/dev/null || true
  fi
  rm -f "${f}"
}

stop_port() {
  local port="$1"
  local pids

  if command -v lsof >/dev/null 2>&1; then
    pids="$(lsof -tiTCP:"${port}" -sTCP:LISTEN || true)"
  elif command -v fuser >/dev/null 2>&1; then
    fuser -k "${port}/tcp" >/dev/null 2>&1 || true
    sleep 1
    return
  else
    pids=""
  fi

  if [[ -z "${pids}" ]]; then
    echo "No process is listening on port ${port} (or cannot detect without lsof/fuser)."
    return
  fi

  echo "Stopping process on port ${port}: ${pids}"
  kill ${pids} || true
  sleep 2

  pids="$(lsof -tiTCP:"${port}" -sTCP:LISTEN || true)"
  if [[ -n "${pids}" ]]; then
    echo "Force stopping process on port ${port}: ${pids}"
    kill -9 ${pids} || true
  fi
}

port_is_open() {
  local port="$1"
  if command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:"${port}" -sTCP:LISTEN >/dev/null 2>&1
    return
  fi
  if command -v curl >/dev/null 2>&1; then
    curl -sf --connect-timeout 1 "http://127.0.0.1:${port}/" >/dev/null 2>&1 ||
      curl -sf --connect-timeout 1 "http://127.0.0.1:${port}/openapi.json" >/dev/null 2>&1
    return
  fi
  timeout 0.4 bash -c "echo >/dev/tcp/127.0.0.1/${port}" >/dev/null 2>&1
}

wait_port() {
  # 勿用 {1..45}：macOS /bin/bash 3.2 不展开该语法，会导致只等 1 次就误判失败
  local port="$1"
  local name="$2"
  local i

  for ((i = 1; i <= 45; i++)); do
    if port_is_open "${port}"; then
      echo "${name} is listening on port ${port}."
      return 0
    fi
    sleep 1
  done

  echo "${name} did not start on port ${port}. Check logs under ${LOG_DIR}."
  exit 1
}

echo "Restarting backend and frontend..."
kill_pidfile "${LOG_DIR}/backend.pid"
kill_pidfile "${LOG_DIR}/frontend.pid"
stop_port "${BACKEND_PORT}"
stop_port "${FRONTEND_PORT}"

echo "Starting backend on port ${BACKEND_PORT}..."
(
  cd "${ROOT_DIR}/serve"
  nohup "${PYTHON_BIN}" -m uvicorn api:app --host 0.0.0.0 --port "${BACKEND_PORT}" \
    > "${LOG_DIR}/backend.log" 2>&1 &
  echo $! > "${LOG_DIR}/backend.pid"
)

echo "Starting frontend on port ${FRONTEND_PORT}..."
(
  cd "${ROOT_DIR}"
  API_PORT="${BACKEND_PORT}" FRONTEND_PORT="${FRONTEND_PORT}" \
    GRADIO_SERVER_HOST="${GRADIO_SERVER_HOST:-0.0.0.0}" \
    NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1 \
    nohup "${PYTHON_BIN}" serve/run_gradio.py \
    > "${LOG_DIR}/frontend.log" 2>&1 &
  echo $! > "${LOG_DIR}/frontend.pid"
)

wait_port "${BACKEND_PORT}" "Backend"
wait_port "${FRONTEND_PORT}" "Frontend"

echo "Backend:  http://127.0.0.1:${BACKEND_PORT}"
echo "Frontend: http://127.0.0.1:${FRONTEND_PORT}"
echo "Logs:     ${LOG_DIR}"
