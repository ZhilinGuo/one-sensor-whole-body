#!/bin/bash
# Auxiliary foot-contact prediction from the head IMU alone (0.809 macro-F1 row).
set -euo pipefail
cd "$(dirname "$0")/.."

python eval_contact.py --ckpt imuposer_headfeet_all3 --sensors head --split run \
    --tag contact_head_run
