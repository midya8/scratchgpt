'use strict';
// Independent scalar float64 reverse-mode engine. No native Tensor/NumPy calls.
// Original GPT operations are rebuilt here; scalar graph counts are NOT native
// Tensor.data logical-element counts and neither metric is allocator/RSS memory.
(function(root){
class N{constructor(v,prev=[],back=()=>{}){this.v=v;this.g=0;this.prev=prev;this.back=back;}}
const node=x=>x instanceof N?x:new N(x);
function add(a,b){a=node(a);b=node(b);const z=new N(a.v+b.v,[a,b],()=>{a.g+=z.g;b.g+=z.g;});return z;}
function mul(a,b){a=node(a);b=node(b);const z=new N(a.v*b.v,[a,b],()=>{a.g+=z.g*b.v;b.g+=z.g*a.v;});return z;}
function power(a,p){const z=new N(a.v**p,[a],()=>{a.g+=z.g*p*a.v**(p-1);});return z;}
function exp(a){const z=new N(Math.exp(a.v),[a],()=>{a.g+=z.g*z.v;});return z;}
function log(a){const z=new N(Math.log(a.v),[a],()=>{a.g+=z.g/a.v;});return z;}
function tanh(a){const z=new N(Math.tanh(a.v),[a],()=>{a.g+=z.g*(1-z.v*z.v);});return z;}
const sum=xs=>xs.reduce((a,b)=>add(a,b),new N(0));
function topo(roots){const seen=new Set(),order=[],stack=roots.map(n=>[n,false]);while(stack.length){const [n,done]=stack.pop();if(done)order.push(n);else if(!seen.has(n)){seen.add(n);stack.push([n,true]);for(let i=n.prev.length-1;i>=0;i--)stack.push([n.prev[i],false]);}}return order;}
const flat=x=>Array.isArray(x)?x.flat(Infinity):[x];
function backward(n){n.g=1;for(const q of topo([n]).reverse())q.back();}
function graph(n,exclude){const ex=new Set(exclude);return topo([n]).filter(q=>!ex.has(q)).length;}
const softmax=xs=>{const mx=Math.max(...xs.map(x=>x.v)),es=xs.map(x=>exp(add(x,-mx))),den=power(sum(es),-1);return es.map(e=>mul(e,den));};
function ln(xs,w,b){const mean=mul(sum(xs),1/xs.length),xc=xs.map(x=>add(x,mul(mean,-1))),inv=power(add(mul(sum(xc.map(x=>mul(x,x))),1/xs.length),1e-5),-.5);return xc.map((x,i)=>add(mul(mul(x,inv),w[i]),b[i]));}
function linear(x,w){return w[0].map((_,j)=>sum(x.map((v,i)=>mul(v,w[i][j]))));}
function gelu(x){return mul(mul(x,.5),add(tanh(mul(add(x,mul(power(x,3),.044715)),Math.sqrt(2/Math.PI))),1));}
function reshape(v,shape){if(shape.length===1)return v.slice();const stride=shape.slice(1).reduce((a,b)=>a*b,1);return Array.from({length:shape[0]},(_,i)=>reshape(v.slice(i*stride,(i+1)*stride),shape.slice(1)));}
function weights(rows,copyNodes=false){return Object.fromEntries(rows.map(r=>[r.name,reshape(r.values.map(v=>new N(v instanceof N?v.v:v)),r.shape)]));}
function block(x,w,l,c){const pre='blocks.'+l+'.',d=c.d_model,h=c.n_head,dh=d/h;
 const norm=x.map(row=>row.map(xs=>ln(xs,w[pre+'ln1.w'],w[pre+'ln1.b']))),qkv=norm.map(row=>row.map(xs=>linear(xs,w[pre+'attn.qkv.w'])));
 const out=x.map((row,b)=>row.map((xs,t)=>{let heads=[];for(let head=0;head<h;head++){const q=qkv[b][t].slice(head*dh,(head+1)*dh),scores=[];for(let j=0;j<=t;j++)scores.push(mul(sum(q.map((v,k)=>mul(v,qkv[b][j][d+head*dh+k]))),dh**-.5));const p=softmax(scores);for(let k=0;k<dh;k++)heads.push(sum(p.map((v,j)=>mul(v,qkv[b][j][2*d+head*dh+k]))));}const projected=linear(heads,w[pre+'attn.proj.w']);return xs.map((v,i)=>add(v,projected[i]));}));
 return out.map(row=>row.map(xs=>{const z=ln(xs,w[pre+'ln2.w'],w[pre+'ln2.b']),ff=linear(linear(z,w[pre+'mlp.fc.w']).map(gelu),w[pre+'mlp.proj.w']);return xs.map((v,i)=>add(v,ff[i]));}));
}
function region(x,w,layers,c,events){const names=Object.keys(w).filter(n=>layers.some(l=>n.startsWith('blocks.'+l+'.'))),params=names.flatMap(n=>flat(w[n])),inputs=flat(x),all=[...inputs,...params];
 function run(){const xx=x.map(row=>row.map(v=>v.map(q=>new N(q.v)))),ww={};for(const n of names)ww[n]=reshape(flat(w[n]).map(q=>new N(q.v)),shapeOf(w[n]));let y=xx;for(const l of layers)y=block(y,ww,l,c);return {xx,ww,y};}
 let local=run();const ys=flat(local.y),privateParams=new Set(Object.values(local.ww).flatMap(flat)),count=topo(ys).filter(n=>!privateParams.has(n)).length;
 events.push({region:layers[0],blocks:layers.length,phase:'forward',scalar_nodes:count});
 const values=ys.map(n=>n.v);local=null;
 let proxies;const anchor=new N(0,all,()=>{const now=run(),nodes=flat(now.y);if(nodes.some((n,i)=>n.v!==values[i]))throw Error('rematerialized output changed');events.push({region:layers[0],blocks:layers.length,phase:'recompute',scalar_nodes:count});for(let i=0;i<nodes.length;i++)nodes[i].g+=proxies[i].g;for(const n of topo(nodes).reverse())n.back();flat(now.xx).forEach((n,i)=>{inputs[i].g+=n.g;});for(const name of names)flat(now.ww[name]).forEach((n,i)=>{flat(w[name])[i].g+=n.g;});});
 proxies=values.map(v=>new N(v,[anchor]));return reshape(proxies,shapeOf(x));
}
function shapeOf(x){return Array.isArray(x)?[x.length,...shapeOf(x[0])]:[];}
function validate(o){
 const keys=(v,ks)=>v&&typeof v==='object'&&!Array.isArray(v)&&Object.keys(v).sort().join('|')===ks.slice().sort().join('|');
 const integer=(x,a,b)=>Number.isInteger(x)&&x>=a&&x<=b,finite=(x,a,b)=>typeof x==='number'&&Number.isFinite(x)&&x>=a&&x<=b;
 if(!keys(o,['schema','config','group_size','learning_rate','tokens','targets','weights'])||o.schema!=='scratchgpt.remat.input.v1')throw Error('Unsupported input schema or fields');
 const c=o.config;if(!keys(c,['vocab_size','block_size','n_layer','n_head','d_model'])||!integer(c.vocab_size,2,12)||!integer(c.block_size,1,6)||!integer(c.n_layer,1,3)||!integer(c.n_head,1,16)||!integer(c.d_model,2,8)||c.d_model%c.n_head)throw Error('Invalid bounded model config');
 if(!integer(o.group_size,0,c.n_layer)||!finite(o.learning_rate,0,.1))throw Error('Invalid group size or learning rate');
 for(const k of ['tokens','targets']){const a=o[k];if(!Array.isArray(a)||a.length<1||a.length>2||!Array.isArray(a[0])||!integer(a[0].length,1,c.block_size)||a.some(r=>!Array.isArray(r)||r.length!==a[0].length||r.some(v=>!integer(v,0,c.vocab_size-1))))throw Error('Invalid '+k);}
 if(JSON.stringify(shapeOf(o.tokens))!==JSON.stringify(shapeOf(o.targets)))throw Error('tokens/targets shape mismatch');
 const d=c.d_model,expected=[['tok_emb.w',[c.vocab_size,d]],['pos_emb.w',[c.block_size,d]],['ln_f.w',[d]],['ln_f.b',[d]],['head.w',[d,c.vocab_size]]];
 for(let l=0;l<c.n_layer;l++)for(const [n,s]of [['ln1.w',[d]],['ln1.b',[d]],['attn.qkv.w',[d,3*d]],['attn.proj.w',[d,d]],['ln2.w',[d]],['ln2.b',[d]],['mlp.fc.w',[d,4*d]],['mlp.proj.w',[4*d,d]]])expected.push(['blocks.'+l+'.'+n,s]);
 if(!Array.isArray(o.weights)||o.weights.length!==expected.length)throw Error('Complete named weight list required');
 o.weights.forEach((r,i)=>{const [name,shape]=expected[i];if(!keys(r,['name','shape','values'])||r.name!==name||JSON.stringify(r.shape)!==JSON.stringify(shape)||!Array.isArray(r.values)||r.values.length!==shape.reduce((a,b)=>a*b,1)||r.values.some(v=>!finite(v,-10,10)))throw Error('Weight name, shape, order or value mismatch');});return c;
}
function arm(o,group){const c=o.config,w=weights(o.weights),params=Object.values(w).flatMap(flat),events=[];
 let x=o.tokens.map(row=>row.map((v,t)=>w['tok_emb.w'][v].map((q,i)=>add(q,w['pos_emb.w'][t][i]))));
 if(group){for(let l=0;l<c.n_layer;l+=group)x=region(x,w,Array.from({length:Math.min(group,c.n_layer-l)},(_,i)=>l+i),c,events);}else for(let l=0;l<c.n_layer;l++)x=block(x,w,l,c);
 const logits=x.map(row=>row.map(xs=>linear(ln(xs,w['ln_f.w'],w['ln_f.b']),w['head.w']))),terms=[];
 logits.forEach((row,b)=>row.forEach((xs,t)=>{const max=Math.max(...xs.map(q=>q.v)),lse=add(log(sum(xs.map(q=>exp(add(q,-max))))),max);terms.push(add(lse,mul(xs[o.targets[b][t]],-1)));}));
 const loss=mul(sum(terms),1/terms.length),retained=graph(loss,params);backward(loss);
 return {loss:loss.v,logits:logits.map(row=>row.map(xs=>xs.map(q=>q.v))),scalar_retained_nodes:retained,events,parameters:o.weights.map(r=>{const ns=flat(w[r.name]);return {name:r.name,shape:r.shape.slice(),gradient:ns.map(q=>q.g),updated:ns.map(q=>q.v-o.learning_rate*q.g)};})};
}
function audit(o){validate(o);const eager=arm(o,0),rematerialized=arm(o,o.group_size);let gd=0,ud=0;for(let i=0;i<eager.parameters.length;i++)for(let j=0;j<eager.parameters[i].gradient.length;j++){gd=Math.max(gd,Math.abs(eager.parameters[i].gradient[j]-rematerialized.parameters[i].gradient[j]));ud=Math.max(ud,Math.abs(eager.parameters[i].updated[j]-rematerialized.parameters[i].updated[j]));}return {schema:'scratchgpt.remat.browser.v1',input:o,eager,rematerialized,comparison:{max_gradient_delta:gd,max_update_delta:ud,parameter_coordinates:o.weights.reduce((a,b)=>a+b.values.length,0),scalar_nodes_removed:eager.scalar_retained_nodes-rematerialized.scalar_retained_nodes},scope:'Independent float64 scalar graph and actual transformer backward/recomputation. Scalar node counts differ from native logical Tensor.data elements. Neither is RSS/allocator memory. No dropout, GPU, speed or new training claims.'};}
const api={audit,validate};if(typeof module!=='undefined')module.exports=api;else root.Remat=api;
})(typeof globalThis!=='undefined'?globalThis:this);
