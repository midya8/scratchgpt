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
import importlib,json,os,pathlib,subprocess,sys
import numpy as np
from scratchgpt.remat_audit import audit,fixture
from scratchgpt.remat import forward
from scratchgpt.model import GPT
from scratchgpt.tensor import cross_entropy
root=pathlib.Path(os.environ['ROOT'])
for name in ('scratchgpt.remat','scratchgpt.remat_audit','scratchgpt.model','scratchgpt.tensor'):
    mod=importlib.import_module(name)
    assert 'site-packages' in mod.__file__ and not str(mod.__file__).startswith(str(root))
d=json.loads((root/'docs/remat-witnesses.json').read_text())
fields=0
for r in d['reports']:
    fresh=audit(r['input'])
    for arm in ('eager','rematerialized'):
        np.testing.assert_allclose(fresh[arm]['logits'],r[arm]['logits'],atol=2e-6,rtol=2e-6)
        assert fresh[arm]['retained']==r[arm]['retained'] and fresh[arm]['events']==r[arm]['events']
        for a,b in zip(fresh[arm]['parameters'],r[arm]['parameters']):
            assert a['name']==b['name'] and a['shape']==b['shape']
            for k in ('gradient','updated'):
                np.testing.assert_allclose(a[k],b[k],atol=2e-6,rtol=2e-6);fields+=len(a[k])
pathlib.Path('input.json').write_text(json.dumps(fixture()))
example=json.loads(subprocess.check_output([sys.executable,'-I','-m','scratchgpt.remat_audit','--fixture']))
assert example==fixture()
cli=json.loads(subprocess.check_output([sys.executable,'-I','-m','scratchgpt.remat_audit','input.json']))
assert cli==audit(fixture())
with np.load(root/'checkpoints/final.npz',allow_pickle=False) as z:
    cfg=json.loads(str(z['config']));m=GPT(**cfg,rng=np.random.default_rng(0))
    for i,p in enumerate(m.parameters()):p.data[...]=z[f'p{i}']
x=np.array([[1,2,3,4]]);y=np.array([[2,3,4,5]])
refs=None
for group in (0,1,2):
    for p in m.parameters():p.grad=None
    logits=forward(m,x,group);cross_entropy(logits,y).backward()
    grads=[p.grad.copy() for p in m.parameters()]
    if refs is None:refs=grads
    else:
        for a,b in zip(refs,grads):np.testing.assert_allclose(a,b,atol=1e-7,rtol=2e-5)
print(json.dumps(dict(installed_modules=4,native_witnesses=len(d['reports']),parameter_numeric_fields=fields,both_cli_paths=True,original_trained_coordinates=m.n_params,trained_groupings=2)))
PY
