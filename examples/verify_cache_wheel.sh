#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PYTHON=${PYTHON:-python}
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT
"$PYTHON" -m build --wheel --outdir "$OUT/wheel" "$ROOT" >/dev/null
uv venv --python "$PYTHON" "$OUT/env" >/dev/null
uv pip install --python "$OUT/env/bin/python" "$OUT"/wheel/*.whl >/dev/null
cd "$OUT"
export ROOT
"$OUT/env/bin/python" -I - <<'PY'
import importlib
import json
import os
import pathlib
import subprocess
import sys

import numpy as np
from scratchgpt.cache_audit import audit, fixture
from scratchgpt.cache_benchmark import benchmark, load_legacy
from scratchgpt.inference import Inference

root = pathlib.Path(os.environ['ROOT'])
for name in ('scratchgpt.inference', 'scratchgpt.cache_audit', 'scratchgpt.cache_benchmark'):
    module = importlib.import_module(name)
    assert 'site-packages' in module.__file__
    assert not str(module.__file__).startswith(str(root))

def compare(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            compare(a[key], b[key])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            compare(x, y)
    elif type(a) is float:
        np.testing.assert_allclose(a, b, atol=2e-6, rtol=2e-6)
    else:
        assert type(a) is type(b) and a == b

controls = json.loads((root / 'docs/cache-witnesses.json').read_text())
for control in controls['reports']:
    compare(audit(control['input']), {k: v for k, v in control.items() if k != 'fixture'})
pathlib.Path('input.json').write_text(json.dumps(fixture()))
cli_fixture = json.loads(subprocess.check_output([sys.executable, '-I', '-m', 'scratchgpt.cache_audit', '--fixture']))
assert cli_fixture == fixture()
cli_report = json.loads(subprocess.check_output([sys.executable, '-I', '-m', 'scratchgpt.cache_audit', 'input.json']))
compare(cli_report, audit(fixture()))
model, sha = load_legacy(root / 'checkpoints/final.npz')
assert sha == '3de6a1626e4489a2e0bc867fc58fe66d323ac6c527bb3af7be662b49510a817e'
snapshot = Inference(model)
session = snapshot.session()
stream = np.random.default_rng(31).integers(model.vocab_size, size=(1, 80))
for end in range(1, 81):
    got = session.advance(stream[:, end-1:end])[:, -1]
    original = model(stream[:, max(0, end-model.block_size):end]).data[:, -1]
    np.testing.assert_allclose(got, original, atol=1e-5, rtol=0)
assert session.counts['window_rebuilds'] == 16
# This checks the installed benchmark CLI without trying to reproduce timings.
cli_benchmark = json.loads(subprocess.check_output([
    sys.executable, '-I', '-m', 'scratchgpt.cache_benchmark', '--checkpoint', str(root / 'checkpoints/final.npz'),
    '--steps', '3', '--repeats', '1', '--warmups', '0', '--prompts', '4', '64']))
assert cli_benchmark['checkpoint_sha256'] == sha
assert len(cli_benchmark['cases']) == 2
for case in cli_benchmark['cases']:
    assert case['arms']['numpy_cached']['max_logit_delta'] < 1e-5
print(json.dumps(dict(installed_modules=3, full_native_controls=controls['count'],
                     all_native_coordinates=True, audit_cli_routes=2, benchmark_cli=True,
                     shipped_checkpoint_steps=80, window_rebuilds=16)))
PY
