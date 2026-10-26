#!/usr/bin/env bash
# 下载 CUAD v1 数据集（约 38MB）到 data/cuad/。见 tasks/T-004.md「数据准备」。
# data/ 已被 .gitignore 排除，不会进入版本控制。
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
url="https://huggingface.co/datasets/theatticusproject/cuad/resolve/main/CUAD_v1/CUAD_v1.json"

mkdir -p "${root}/data/cuad"
curl -L -o "${root}/data/cuad/CUAD_v1.json" "${url}"
