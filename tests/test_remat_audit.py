import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scratchgpt.remat_audit import audit,fixture,parse


def test_all_checked_in_native_witnesses_rebuild_every_field():
    import runpy
    build=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/build_remat.py'))['build']
    old=json.loads((Path(__file__).resolve().parents[1]/'docs/remat-witnesses.json').read_text())
    fresh=build()
    def compare(a,b):
        if type(a) in (int,float):
            assert abs(a-b)<=2e-6*max(1,abs(a),abs(b))
        elif isinstance(a,list):
            assert len(a)==len(b)
            for x,y in zip(a,b):compare(x,y)
        elif isinstance(a,dict):
            assert set(a)==set(b)
            for k in a:compare(a[k],b[k])
        else:assert a==b
    compare(fresh,old)


@pytest.mark.parametrize('change',[
 lambda o:o.update(schema='bad'),lambda o:o.update(extra=1),lambda o:o.update(group_size=True),
 lambda o:o.update(group_size=-1),lambda o:o.update(learning_rate=float('nan')),
 lambda o:o.update(tokens=[[True]]),lambda o:o.update(tokens=[[100]]),lambda o:o.update(targets=[[1]]),
 lambda o:o['config'].update(d_model=10),lambda o:o['weights'].pop(),lambda o:o['weights'].reverse(),
 lambda o:o['weights'][0]['values'].__setitem__(0,float('inf')),
 lambda o:o['weights'][0]['shape'].__setitem__(0,True),lambda o:o['weights'][0]['values'].pop(),
])
def test_bad_schema(change):
    o=fixture();change(o)
    with pytest.raises((ValueError,TypeError)):parse(o)


def test_cli_example_and_both_error_routes(tmp_path):
    result=subprocess.run([sys.executable,'-m','scratchgpt.remat_audit','--fixture'],capture_output=True,text=True)
    assert result.returncode==0
    o=json.loads(result.stdout);p=tmp_path/'input.json';p.write_text(json.dumps(o))
    r=subprocess.run([sys.executable,'-m','scratchgpt.remat_audit',str(p)],capture_output=True,text=True)
    assert r.returncode==0 and json.loads(r.stdout)==audit(o)
    p.write_text('{"schema":NaN}')
    r=subprocess.run([sys.executable,'-m','scratchgpt.remat_audit',str(p)],capture_output=True,text=True)
    assert r.returncode==2 and 'nonfinite' in r.stderr
    r=subprocess.run([sys.executable,'-m','scratchgpt.remat_audit'],capture_output=True,text=True)
    assert r.returncode==2 and 'provide input' in r.stderr
