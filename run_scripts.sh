#!/bin/bash
set -euo pipefail

python3 API-downloading.py
python3 'Run&Output.py'

echo "Scripts executed successfully."
