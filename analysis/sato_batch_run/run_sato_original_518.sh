#!/bin/bash
# Times SATO's unmodified straighten_seg.py on C0518.
# Expects SATO cloned into SATO_repo/ at the repo root, and SATO_inputs/
# prepared by sato_prepare.py. Run from anywhere; the log is written next
# to this script.
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/../.."
PY="${PYTHON:-python}"
LOG="$HERE/sato_original_518_timing.log"

echo "START: $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$LOG"
START_EPOCH=$(date +%s)

"$PY" SATO_repo/straighten/straighten_seg.py \
    --seg_path SATO_inputs/518_mask.nii.gz \
    --centerline_path SATO_inputs/518_centerline.npy \
    --save_straighten_path SATO_inputs/518_straightened_ORIGINAL_unfixed.nii.gz \
    >> "$LOG" 2>&1
STATUS=$?

END_EPOCH=$(date +%s)
echo "END: $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"
echo "EXIT_STATUS: $STATUS" >> "$LOG"
echo "ELAPSED_SECONDS: $((END_EPOCH - START_EPOCH))" >> "$LOG"
