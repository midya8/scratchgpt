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
"$OUT/env/bin/python" - <<'PY'
import importlib,json,os,pathlib,tempfile,subprocess,sys
import numpy as np
from scratchgpt.checkpoint import Training
root=pathlib.Path(os.environ['ROOT'])
for name in ('scratchgpt','scratchgpt.checkpoint','scratchgpt.replay','scratchgpt.train','scratchgpt.tensor'):
    module=importlib.import_module(name)
    assert 'site-packages' in module.__file__ and not str(module.__file__).startswith(str(root))
d=json.loads((root/'docs/replay.json').read_text());text=d['text']*d['repeats']
a=Training(text,d['config'],d['options'])
for _ in range(12):a.update()
for split in range(13):
    b=Training(text,d['config'],d['options'])
    for _ in range(split):b.update()
    b.save('state.npz');b=Training.load('state.npz',text,d['config'],d['options'])
    for _ in range(12-split):b.update()
    assert b.digest()==a.digest()
pathlib.Path('text.txt').write_text(text)
cmd=[sys.executable,'-m','scratchgpt.train','--data','text.txt','--out','run','--steps','4','--block','6','--batch','3','--n-layer','1','--n-head','2','--d-model','8','--lr','.003','--warmup','5','--clip','.4','--eval-every','3','--eval-batches','2','--seed','7']
subprocess.run(cmd,check=True,capture_output=True)
cmd[cmd.index('--steps')+1]='8';subprocess.run(cmd,check=True,capture_output=True)
c=Training.load('run/ckpt.npz',text,d['config'],d['options']);assert c.digest()==a.digest()
# Original shipped inference checkpoint stays supported, outside the checkout.
result=subprocess.run([sys.executable,'-m','scratchgpt.sample','--ckpt',str(root/'checkpoints/final.npz'),'--data',str(root/'data/tinyshakespeare.txt'),'--prompt','JULIET:','--n','16','--top-k','20'],check=True,capture_output=True,text=True)
assert result.stdout.startswith('JULIET:')
print(json.dumps(dict(installed_modules=5,boundaries=13,separate_process_resume=True,original_shipped_sample=True)))
PY
