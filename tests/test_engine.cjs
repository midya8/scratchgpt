'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),R=require('../docs/engine.js'),d=require('../docs/replay.json');
function close(a,b,tol=2e-7){assert.ok(Number.isFinite(a)&&Math.abs(a-b)<=tol,`${a} vs ${b}`);}
test('all scalar Adam states and independent complete forward/loss references',()=>{let count=0,logits=0;
 for(const r of d.records){for(const i of r.before)for(let k=0;k<i.p.length;k++){let got=R.adam(i.p[k],i.g[k],i.m[k],i.v[k],r.step,r.lr);for(const f of ['p','m','v']){close(got[f],i.after[f][k]);count++;}}
 let loss=0;for(let b=0;b<3;b++){let l=R.forward(R.weights(r),d.config,r.x[b]);for(let t=0;t<6;t++)for(let v=0;v<d.config.vocab_size;v++){close(l[t][v],r.logits[b][t][v],2e-5);logits++;}loss+=R.nll(R.forward(R.weights(r,'before'),d.config,r.x[b]),r.y[b])/3;}close(loss,r.loss,1e-6);}
 assert.equal(count,38016);assert.equal(logits,2592);console.log({adam_values:count,forward_values:logits});
});
test('causality, input and optimizer domain checks',()=>{let r=d.records[5],w=R.weights(r),x=r.x[0].slice(),y=x.slice();y[5]=(y[5]+1)%12;assert.deepEqual(R.forward(w,d.config,x).slice(0,5),R.forward(w,d.config,y).slice(0,5));for(const text of ['', 'toolong!', 'z'])assert.throws(()=>R.encode(text,d.vocabulary,6));for(const v of [-1,0,1.5,NaN])assert.throws(()=>R.adam(1,2,0,0,v,.01));assert.throws(()=>R.adam(1,2,0,-1,2,.01));assert.throws(()=>R.adam(1,2,0,0,2,0));assert.throws(()=>R.adam(1,2,0,0,2,.01,1));assert.throws(()=>R.forward(w,d.config,[12]));});
