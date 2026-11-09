import copy
import json
import os
import zipfile

import numpy as np
import pytest

from scratchgpt.checkpoint import Training, canonical, strict_json
from scratchgpt.data import get_batch
from scratchgpt.model import GPT
from scratchgpt.optim import Adam
from scratchgpt.tensor import cross_entropy
from scratchgpt.train import clip_grad_norm

TEXT = 'abcde abc ab cde\n'*100
CFG = dict(vocab_size=len(set(TEXT)),block_size=4,n_layer=1,n_head=2,d_model=8)
OPT = dict(batch=3,lr=.003,warmup=5,clip=.4,eval_every=3,eval_batches=2,seed=7)


def make():
    return Training(TEXT,CFG,OPT)


def rewrite(path, mutate):
    with np.load(path,allow_pickle=False) as z:
        arrays={k:z[k].copy() for k in z.files}
    m=json.loads(str(arrays['metadata']))
    mutate(m,arrays)
    arrays['metadata']=np.array(canonical(m))
    np.savez(path,**arrays)


@pytest.mark.parametrize('split',range(13))
def test_every_resume_boundary(tmp_path,split):
    full=make()
    for _ in range(12): full.update()
    a=make()
    for _ in range(split): a.update()
    # Arbitrary terminal previews do not consume evaluation randomness/history.
    digest=a.digest()
    assert a.evaluate()==a.evaluate()
    assert a.digest()==digest
    a.save(tmp_path/'x.npz')
    b=Training.load(tmp_path/'x.npz',TEXT,CFG,OPT)
    assert b.digest()==a.digest()
    for _ in range(12-split): b.update()
    assert b.digest()==full.digest()
    for p,q in zip(full.model.parameters(),b.model.parameters()):
        assert np.array_equal(p.data,q.data)


def test_original_updates_are_retained():
    a=make()
    r=np.random.default_rng(OPT['seed'])
    model=GPT(rng=r,dropout=0.,**CFG)
    opt=Adam(model.parameters(),lr=OPT['lr'])
    for step in range(8):
        x,y=get_batch(a.train_data,CFG['block_size'],OPT['batch'],r)
        loss=cross_entropy(model(x),y);opt.zero_grad();loss.backward()
        clip_grad_norm(model.parameters(),OPT['clip'])
        if step<OPT['warmup']: opt.lr=OPT['lr']*(step+1)/OPT['warmup']
        opt.step();a.update()
        assert float(loss.data)==a.last_loss
        for p,q in zip(model.parameters(),a.model.parameters()):
            assert np.array_equal(p.data,q.data)


@pytest.mark.parametrize('key',['seed','batch','lr','warmup','clip','eval_every','eval_batches'])
def test_changed_options_refused(tmp_path,key):
    a=make();a.save(tmp_path/'x.npz')
    o=dict(OPT);o[key]=o[key]*2 if key=='lr' else o[key]+1
    with pytest.raises(ValueError,match='options'): Training.load(tmp_path/'x.npz',TEXT,CFG,o)


def test_same_vocab_different_data_refused(tmp_path):
    a=make();a.save(tmp_path/'x.npz')
    with pytest.raises(ValueError,match='data'): Training.load(tmp_path/'x.npz',TEXT[::-1],CFG,OPT)


@pytest.mark.parametrize('case', ['schema','environment','names','clock','boolstep','history','loss',
                                  'rng','rnginc','rngcached','adam','shape','dtype','nan','negativev','extra','missing'])
def test_corrupt_state_refused(tmp_path,case):
    a=make()
    for _ in range(3):a.update()
    p=tmp_path/'x.npz';a.save(p)
    def mutate(m,z):
        if case=='schema':m['schema']='other'
        if case=='environment':m['environment']['numpy']='other'
        if case=='names':m['names'][0]='wrong'
        if case=='clock':m['opt_t']=2
        if case=='boolstep':m['step']=True
        if case=='history':m['history'][0][0]=2
        if case=='loss':m['last_loss']=-1
        if case=='rng':m['rng']['bit_generator']='MT19937'
        if case=='rnginc':m['rng']['state']['inc']=2
        if case=='rngcached':m['rng']['has_uint32']=2
        if case=='adam':m['adam']['lr']=.9
        if case=='shape':z['p0']=np.zeros(1,dtype=np.float32)
        if case=='dtype':z['p0']=z['p0'].astype(np.float64)
        if case=='nan':z['p0'][0,0]=np.nan
        if case=='negativev':z['v0'][0,0]=-1
        if case=='extra':z['surprise']=np.array(1)
        if case=='missing':del z['p0']
    rewrite(p,mutate)
    with pytest.raises(ValueError):Training.load(p,TEXT,CFG,OPT)


def test_legacy_not_silently_resumed(tmp_path):
    p=tmp_path/'x.npz';np.savez(p,config=json.dumps(CFG),p0=np.zeros(1))
    with pytest.raises(ValueError,match='legacy'):Training.load(p,TEXT,CFG,OPT)


def test_pickle_refused(tmp_path):
    p=tmp_path/'x.npz';np.savez(p,metadata=np.array({'danger':1},dtype=object))
    with pytest.raises(ValueError):Training.load(p,TEXT,CFG,OPT)


def test_duplicate_archive_member_refused(tmp_path):
    p=tmp_path/'x.npz';make().save(p)
    with zipfile.ZipFile(p,'a') as z:
        with pytest.warns(UserWarning):z.writestr('metadata.npy',b'bad')
    with pytest.raises(ValueError,match='duplicate'):Training.load(p,TEXT,CFG,OPT)


def test_atomic_replace_failure_keeps_previous(tmp_path,monkeypatch):
    p=tmp_path/'x.npz';a=make();a.save(p);original=p.read_bytes();a.update()
    def fail(*args):raise OSError('simulated replace failure')
    monkeypatch.setattr(os,'replace',fail)
    with pytest.raises(OSError):a.save(p)
    assert p.read_bytes()==original
    assert list(tmp_path.iterdir())==[p]


def test_atomic_serialization_failure_keeps_previous(tmp_path,monkeypatch):
    p=tmp_path/'x.npz';a=make();a.save(p);original=p.read_bytes()
    def fail(*args,**kwargs):raise OSError('simulated save failure')
    monkeypatch.setattr(np,'savez',fail)
    with pytest.raises(OSError):a.save(p)
    assert p.read_bytes()==original
    assert list(tmp_path.iterdir())==[p]


def test_strict_json():
    with pytest.raises(ValueError):strict_json('{"x":1,"x":2}')
    with pytest.raises(ValueError):strict_json('{"x":NaN}')


def test_missing_eval_rng_changes_future_history(tmp_path):
    a=make()
    for _ in range(4):a.update()
    a.save(tmp_path/'x.npz');b=Training.load(tmp_path/'x.npz',TEXT,CFG,OPT)
    b.eval_rng=np.random.default_rng(OPT['seed']+1)
    for _ in range(2):a.update();b.update()
    assert a.history!=b.history
    assert all(np.array_equal(p.data,q.data) for p,q in zip(a.model.parameters(),b.model.parameters()))


def test_legacy_rng_restart_changes_updates(tmp_path):
    a=make()
    for _ in range(4):a.update()
    a.save(tmp_path/'x.npz');b=Training.load(tmp_path/'x.npz',TEXT,CFG,OPT)
    b.rng=make().rng
    a.update();b.update()
    assert any(not np.array_equal(p.data,q.data) for p,q in zip(a.model.parameters(),b.model.parameters()))


def test_nonfinite_save_does_not_replace(tmp_path):
    p=tmp_path/'x.npz';a=make();a.save(p);before=p.read_bytes()
    a.model.parameters()[0].data[0,0]=np.nan
    with pytest.raises(ValueError):a.save(p)
    assert p.read_bytes()==before
