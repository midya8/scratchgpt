"""Independent float64 forward and scalar Adam witnesses, not a second trainer."""
import math

import numpy as np


def forward(weights, config, tokens):
    """No Tensor, GPT, nn or optimizer calls. Names bind the supplied weights."""
    w={k:np.asarray(v,dtype=np.float64) for k,v in weights.items()}
    x=w['tok_emb.w'][tokens]+w['pos_emb.w'][np.arange(len(tokens))]
    d,h=config['d_model'],config['n_head'];dh=d//h
    def ln(x,prefix):
        mean=x.mean(axis=-1,keepdims=True)
        return (x-mean)/np.sqrt(((x-mean)**2).mean(axis=-1,keepdims=True)+1e-5)*w[prefix+'.w']+w[prefix+'.b']
    for l in range(config['n_layer']):
        p=f'blocks.{l}.'
        qkv=(ln(x,p+'ln1')@w[p+'attn.qkv.w']).reshape(len(tokens),3,h,dh)
        q,k,v=[qkv[:,i].transpose(1,0,2) for i in range(3)]
        scores=q@k.transpose(0,2,1)/math.sqrt(dh)
        scores[:,np.triu_indices(len(tokens),1)[0],np.triu_indices(len(tokens),1)[1]]=-np.inf
        att=np.exp(scores-scores.max(axis=-1,keepdims=True));att/=att.sum(axis=-1,keepdims=True)
        x=x+(att@v).transpose(1,0,2).reshape(len(tokens),d)@w[p+'attn.proj.w']
        z=ln(x,p+'ln2')@w[p+'mlp.fc.w']
        z=.5*z*(1+np.tanh(math.sqrt(2/math.pi)*(z+.044715*z**3)))
        x=x+z@w[p+'mlp.proj.w']
    return ln(x,'ln_f')@w['head.w']


def adam_scalar(p,g,m,v,t,lr,b1=.9,b2=.999,eps=1e-8):
    """Math reference in float64, deliberately not the float32 implementation."""
    m=b1*m+(1-b1)*g
    v=b2*v+(1-b2)*g*g
    return p-lr*(m/(1-b1**t))/(math.sqrt(v/(1-b2**t))+eps),m,v
