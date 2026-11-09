import os
import subprocess
import sys
import numpy as np
from scratchgpt.checkpoint import Training
from test_checkpoint import TEXT,CFG,OPT


def test_separate_process_continuation_and_sampling(tmp_path):
    data=tmp_path/'text.txt';data.write_text(TEXT)
    def train(out,steps):
        cmd=[sys.executable,'-m','scratchgpt.train','--data',str(data),'--out',str(out),
             '--steps',str(steps),'--block','4','--n-layer','1','--n-head','2','--d-model','8',
             '--batch','3','--lr','.003','--warmup','5','--clip','.4','--eval-every','3',
             '--eval-batches','2','--seed','7']
        return subprocess.run(cmd,text=True,capture_output=True,check=True)
    train(tmp_path/'full',12);train(tmp_path/'split',4);train(tmp_path/'split',8)
    a=Training.load(tmp_path/'full/ckpt.npz',TEXT,CFG,OPT)
    b=Training.load(tmp_path/'split/ckpt.npz',TEXT,CFG,OPT)
    assert a.digest()==b.digest()
    got=subprocess.run([sys.executable,'-m','scratchgpt.sample','--ckpt',str(tmp_path/'split/final.npz'),
                        '--data',str(data),'--prompt','abc','--n','5','--top-k','3'],text=True,capture_output=True,check=True)
    assert got.stdout.startswith('abc') and len(got.stdout.strip())>=7
    # Changed dataset is a hard failure, and does not overwrite the checkpoint.
    before=(tmp_path/'split/ckpt.npz').read_bytes()
    data.write_text(TEXT[::-1])
    try:train(tmp_path/'split',1)
    except subprocess.CalledProcessError as e:
        assert e.returncode==2 and 'data mismatch' in e.stderr
    else:raise AssertionError('mismatched data accepted')
    assert (tmp_path/'split/ckpt.npz').read_bytes()==before
