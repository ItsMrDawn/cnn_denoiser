#!/bin/bash
#
# Fetch Microsoft MS-SNSD and synthesize the training set this project expects:
#   dataset/CleanSpeech_training/   clnsp{N}.wav
#   dataset/NoisySpeech_training/   noisy{N}_SNRdb_{snr}_clnsp{N}.wav  (0/10/20/30 dB)
#   dataset/clean_test, dataset/noise_test   (used by search evaluation)
#
# Usage: scripts/data/prepare_dataset.sh [total_hours]   (default 5 => ~380 clean files)
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATASET_DIR="${ROOT_DIR}/dataset"
SNSD_DIR="${DATASET_DIR}/MS-SNSD"
TOTAL_HOURS=${1:-5}
PYTHON="${ROOT_DIR}/.venv/bin/python"

mkdir -p "$DATASET_DIR"

if [ ! -d "$SNSD_DIR" ]; then
    git clone --depth 1 https://github.com/microsoft/MS-SNSD.git "$SNSD_DIR"
fi

if [ ! -d "${SNSD_DIR}/NoisySpeech_training" ]; then
    cat > "${SNSD_DIR}/cnn_denoiser.cfg" <<EOF
[noisy_speech]
sampling_rate: 16000
audioformat: *.wav
audio_length: 10
silence_length: 0.2
total_hours: ${TOTAL_HOURS}
snr_lower: 0
snr_upper: 30
total_snrlevels: 4
noise_dir: None
speech_dir: None
noise_types_excluded: None
EOF
    # The upstream synthesizer passes a float to np.linspace(num=...), which
    # NumPy >= 1.18 rejects, and does not seed its RNG. Patch both at runtime.
    (cd "$SNSD_DIR" && "$PYTHON" -c '
import runpy, sys, numpy as np
_linspace = np.linspace
np.linspace = lambda a, b, n, *args, **kw: _linspace(a, b, int(n), *args, **kw)
np.random.seed(42)
sys.argv = ["noisyspeech_synthesizer.py", "--cfg", "cnn_denoiser.cfg"]
runpy.run_path("noisyspeech_synthesizer.py", run_name="__main__")
')
fi

for d in CleanSpeech_training NoisySpeech_training clean_test noise_test; do
    ln -sfn "MS-SNSD/${d}" "${DATASET_DIR}/${d}"
done

echo "Clean files: $(ls "${DATASET_DIR}/CleanSpeech_training" | wc -l | tr -d ' ')"
echo "Noisy files: $(ls "${DATASET_DIR}/NoisySpeech_training" | wc -l | tr -d ' ')"
