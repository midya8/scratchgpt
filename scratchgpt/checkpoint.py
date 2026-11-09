"""Validated, atomic, step-boundary continuation. No pickle or implicit warm start."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import platform
import tempfile
import zipfile
import warnings
from pathlib import Path

import numpy as np

from .data import CharVocab, get_batch, train_val_split
from .model import GPT
from .optim import Adam
from .tensor import cross_entropy

SCHEMA = 'scratchgpt.continuation.v1'
THREADS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')


def integer(x, lo, hi, name):
    if type(x) is not int or not lo <= x <= hi:
        raise ValueError(f'{name} must be an integer in [{lo}, {hi}]')
    return x


def finite(x, lo, hi, name):
    if type(x) not in (int, float) or not np.isfinite(x) or not lo <= x <= hi:
        raise ValueError(f'{name} must be finite in [{lo}, {hi}]')
    return float(x)


def fields(obj, expected, name):
    if type(obj) is not dict or set(obj) != set(expected):
        raise ValueError(f'{name}: expected fields {sorted(expected)}')


def config_validate(c):
    fields(c, ('vocab_size', 'block_size', 'n_layer', 'n_head', 'd_model'), 'config')
    for k, a, b in [('vocab_size', 2, 512), ('block_size', 1, 512),
                    ('n_layer', 1, 8), ('n_head', 1, 16), ('d_model', 2, 256)]:
        integer(c[k], a, b, k)
    if c['d_model'] % c['n_head']:
        raise ValueError('d_model must be divisible by n_head')
    d, l = c['d_model'], c['n_layer']
    if c['vocab_size'] * d * 2 + c['block_size'] * d + l * (12*d*d+4*d) + 2*d > 7_000_000:
        raise ValueError('model exceeds checkpoint parameter bound')


def options_validate(o):
    fields(o, ('batch', 'lr', 'warmup', 'clip', 'eval_every', 'eval_batches', 'seed'), 'options')
    for k, a, b in [('batch', 1, 128), ('warmup', 0, 10**9),
                    ('eval_every', 1, 10**9), ('eval_batches', 1, 1000),
                    ('seed', 0, 2**63-1)]:
        integer(o[k], a, b, k)
    finite(o['lr'], 1e-12, 1, 'lr')
    finite(o['clip'], 0, 1e6, 'clip')


def environment():
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()) as s:
        warnings.simplefilter('ignore', UserWarning)
        np.show_config()
    return dict(python=platform.python_version(), numpy=np.__version__,
                platform=platform.platform(), machine=platform.machine(),
                numpy_build=s.getvalue(), threads={k: os.environ.get(k) for k in THREADS})


def named_parameters(model):
    """Stable names in the original GPT.parameters order, never sorted."""
    out = [('tok_emb.w', model.tok_emb.w), ('pos_emb.w', model.pos_emb.w),
           ('ln_f.w', model.ln_f.w), ('ln_f.b', model.ln_f.b), ('head.w', model.head.w)]
    for i, b in enumerate(model.blocks):
        for name, p in [('ln1.w', b.ln1.w), ('ln1.b', b.ln1.b),
                        ('attn.qkv.w', b.attn.qkv.w), ('attn.proj.w', b.attn.proj.w),
                        ('ln2.w', b.ln2.w), ('ln2.b', b.ln2.b),
                        ('mlp.fc.w', b.mlp.fc.w), ('mlp.proj.w', b.mlp.proj.w)]:
            out.append((f'blocks.{i}.{name}', p))
    if [id(p) for _, p in out] != [id(p) for p in model.parameters()]:
        raise ValueError('parameter order changed; a schema migration is required')
    return out


def rng_validate(s):
    fields(s, ('bit_generator', 'state', 'has_uint32', 'uinteger'), 'RNG')
    if s['bit_generator'] != 'PCG64':
        raise ValueError('only PCG64 is supported')
    fields(s['state'], ('state', 'inc'), 'PCG64 state')
    integer(s['state']['state'], 0, 2**128-1, 'PCG64 state')
    integer(s['state']['inc'], 1, 2**128-1, 'PCG64 increment')
    if not s['state']['inc'] % 2:
        raise ValueError('PCG64 increment must be odd')
    integer(s['has_uint32'], 0, 1, 'has_uint32')
    integer(s['uinteger'], 0, 2**32-1, 'uinteger')


def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False)


def strict_json(s):
    def pairs(items):
        d = {}
        for k, v in items:
            if k in d:
                raise ValueError('duplicate JSON key')
            d[k] = v
        return d
    def reject(x):
        raise ValueError('nonfinite JSON constant')
    return json.loads(s, object_pairs_hook=pairs, parse_constant=reject)


class Training:
    """A checkpoint boundary is after update and scheduled evaluation, before next batch."""
    def __init__(self, text, config, options):
        config_validate(config)
        options_validate(options)
        self.config, self.options = dict(config), dict(options)
        self.vocab = CharVocab(text)
        if len(self.vocab) != config['vocab_size']:
            raise ValueError('vocabulary size does not match config')
        self.data_identity = dict(sha256=hashlib.sha256(text.encode('utf-8')).hexdigest(),
                                  characters=len(text), vocabulary=self.vocab.itos,
                                  split='contiguous-last-floor-10-percent')
        self.train_data, self.val_data = train_val_split(self.vocab.encode(text))
        if min(len(self.train_data), len(self.val_data)) <= config['block_size'] + 1:
            raise ValueError('both splits need more than block_size+1 characters')
        self.rng = np.random.Generator(np.random.PCG64(options['seed']))
        self.eval_rng = np.random.Generator(np.random.PCG64(options['seed']+1))
        self.model = GPT(rng=self.rng, dropout=0.0, **config)
        self.names = [n for n, _ in named_parameters(self.model)]
        self.opt = Adam(self.model.parameters(), lr=options['lr'])
        self.step, self.history, self.last_loss = 0, [], None
        self.env = environment()

    def evaluate(self, advance=False):
        # Terminal previews must not alter future scheduled evaluation or checkpoint state.
        r = self.eval_rng
        if not advance:
            r = np.random.Generator(np.random.PCG64())
            r.bit_generator.state = self.eval_rng.bit_generator.state
        values = []
        for _ in range(self.options['eval_batches']):
            x, y = get_batch(self.val_data, self.config['block_size'], self.options['batch'], r)
            values.append(float(cross_entropy(self.model(x), y).data))
        return float(np.mean(values))

    def update(self, witness=False):
        o = self.options
        x, y = get_batch(self.train_data, self.config['block_size'], o['batch'], self.rng)
        loss = cross_entropy(self.model(x), y)
        self.opt.zero_grad()
        loss.backward()
        params = self.model.parameters()
        norm = float(np.sqrt(sum(float((p.grad*p.grad).sum()) for p in params)))
        scale = o['clip']/norm if o['clip'] and norm > o['clip'] else 1.0
        for p in params:
            if scale != 1.0:
                p.grad = p.grad * scale
        # Explicit every-step schedule also works across the warmup boundary.
        self.opt.lr = o['lr'] * min((self.step+1)/o['warmup'], 1) if o['warmup'] else o['lr']
        before = None
        if witness:
            before = [dict(name=n, p=p.data.copy(), g=p.grad.copy(),
                           m=m.copy(), v=v.copy()) for (n,p),m,v in
                      zip(named_parameters(self.model), self.opt.m, self.opt.v)]
        self.opt.step()
        self.step += 1
        self.last_loss = float(loss.data)
        if self.step % o['eval_every'] == 0:
            self.history.append([self.step, self.last_loss, self.evaluate(advance=True)])
        return dict(step=self.step, loss=self.last_loss, grad_norm=norm,
                    clip_scale=scale, lr=self.opt.lr, x=x.tolist(), y=y.tolist(), before=before)

    def metadata(self):
        return dict(schema=SCHEMA, config=self.config, options=self.options,
                    data=self.data_identity, environment=self.env, names=self.names,
                    step=self.step, opt_t=self.opt.t, history=self.history, last_loss=self.last_loss,
                    rng=self.rng.bit_generator.state, eval_rng=self.eval_rng.bit_generator.state,
                    adam=dict(b1=self.opt.b1, b2=self.opt.b2, eps=self.opt.eps, lr=self.opt.lr))

    def arrays(self):
        arrays = dict(metadata=np.array(canonical(self.metadata())))
        for i, p in enumerate(self.model.parameters()):
            arrays[f'p{i}'], arrays[f'm{i}'], arrays[f'v{i}'] = p.data, self.opt.m[i], self.opt.v[i]
        return arrays

    def save(self, path):
        # Same-directory temp + replace: failed writes do not truncate a valid checkpoint.
        target = Path(path)
        arrays = self.arrays()
        for key, value in arrays.items():
            if key != 'metadata' and (value.dtype != np.dtype('float32') or not np.isfinite(value).all() or (key.startswith('v') and (value < 0).any())):
                raise ValueError('refusing to save nonfinite/invalid training state')
        fd, name = tempfile.mkstemp(prefix=target.name+'.', suffix='.tmp', dir=target.parent)
        try:
            with os.fdopen(fd, 'wb') as f:
                np.savez(f, **arrays)
                f.flush()
                os.fsync(f.fileno())
            os.replace(name, target)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    @classmethod
    def load(cls, path, text, config, options):
        # Validate the complete artifact into a fresh object before exposing any mutation.
        if Path(path).stat().st_size > 100_000_000:
            raise ValueError('checkpoint exceeds 100 MB limit')
        with zipfile.ZipFile(path) as archive:
            info = archive.infolist()
            if len(info) > 250 or len({i.filename for i in info}) != len(info):
                raise ValueError('duplicate/excess archive members')
            if sum(i.file_size for i in info) > 100_000_000:
                raise ValueError('uncompressed checkpoint exceeds limit')
        with np.load(path, allow_pickle=False) as z:
            if 'metadata' not in z.files:
                raise ValueError('legacy checkpoint is weights-only/unsafe for exact resume; use sample CLI or explicit warm start, not resume')
            md = z['metadata']
            if md.shape != () or md.dtype.kind != 'U' or len(str(md)) > 1_000_000:
                raise ValueError('invalid metadata scalar')
            m = strict_json(str(md))
            fields(m, ('schema','config','options','data','environment','names','step',
                       'opt_t','history','last_loss','rng','eval_rng','adam'), 'metadata')
            if m['schema'] != SCHEMA:
                raise ValueError('unsupported checkpoint schema')
            out = cls(text, config, options)
            for key, expected in [('config',out.config), ('options',out.options),
                                  ('data',out.data_identity), ('environment',out.env), ('names',out.names)]:
                # Canonical encoding distinguishes bool from numeric settings.
                if canonical(m[key]) != canonical(expected):
                    raise ValueError(f'checkpoint {key} mismatch')
            n = len(out.names)
            expected = {'metadata'} | {f'{k}{i}' for k in 'pmv' for i in range(n)}
            if set(z.files) != expected:
                raise ValueError('missing or extra parameter/buffer arrays')
            step = integer(m['step'], 0, 10**9, 'step')
            if type(m['opt_t']) is not int or m['opt_t'] != step:
                raise ValueError('Adam clock != step')
            if step == 0:
                if m['last_loss'] is not None:
                    raise ValueError('step zero has no last loss')
            else:
                finite(m['last_loss'], 0, 1e12, 'last loss')
            if type(m['history']) is not list or len(m['history']) != step//options['eval_every']:
                raise ValueError('evaluation history length mismatch')
            for i, row in enumerate(m['history']):
                if type(row) is not list or len(row) != 3 or type(row[0]) is not int or row[0] != (i+1)*options['eval_every']:
                    raise ValueError('invalid history step')
                finite(row[1], 0, 1e12, 'train loss'); finite(row[2], 0, 1e12, 'val loss')
            for key in ('rng','eval_rng'):
                rng_validate(m[key])
            want_lr = options['lr'] * min(step/options['warmup'],1) if options['warmup'] and step else options['lr']
            want_adam = dict(b1=0.9,b2=0.999,eps=1e-8,lr=want_lr)
            if canonical(m['adam']) != canonical(want_adam):
                raise ValueError('Adam settings/schedule mismatch')
            for i, p in enumerate(out.model.parameters()):
                for prefix, target in [('p',p.data),('m',out.opt.m[i]),('v',out.opt.v[i])]:
                    a = z[f'{prefix}{i}']
                    if a.shape != target.shape or a.dtype != np.dtype('float32') or not np.isfinite(a).all():
                        raise ValueError(f'invalid {prefix}{i} shape/dtype/values')
                    if prefix == 'v' and (a < 0).any():
                        raise ValueError('negative Adam second moment')
                    target[...] = a
            out.step = out.opt.t = step
            out.history, out.last_loss = m['history'], m['last_loss']
            out.opt.lr = m['adam']['lr']
            out.rng.bit_generator.state = m['rng']
            out.eval_rng.bit_generator.state = m['eval_rng']
            return out

    def digest(self):
        h = hashlib.sha256(canonical(self.metadata()).encode())
        for p, m, v in zip(self.model.parameters(), self.opt.m, self.opt.v):
            for a in (p.data,m,v):
                h.update(a.tobytes(order='C'))
        return h.hexdigest()
