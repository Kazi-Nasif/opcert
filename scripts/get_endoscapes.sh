#!/usr/bin/env bash
# Endoscapes2023 (CAMMA, IHU Strasbourg), CC BY-NC-SA 4.0, ~6.3 GB zip, public download.
# Murali et al., arXiv:2312.12429. https://github.com/CAMMA-public/Endoscapes
set -euo pipefail
mkdir -p "$(dirname "$0")/../data" && cd "$(dirname "$0")/../data"
curl -L -C - -o endoscapes.zip https://s3.unistra.fr/camma_public/datasets/endoscapes/endoscapes.zip
unzip -q endoscapes.zip && rm endoscapes.zip
