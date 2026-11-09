/* Scalar float64 forward and Adam reference. No pretrained service or ML runtime. */
(function(root){
'use strict';
function finite(x,name){if(typeof x!=='number'||!Number.isFinite(x))throw Error(name+' must be finite');return x;}
function adam(p,g,m,v,t,lr,b1=.9,b2=.999,eps=1e-8){
 [p,g,m,v,lr,b1,b2,eps].forEach((x,i)=>finite(x,'argument '+i));
 if(!Number.isInteger(t)||t<1||v<0||lr<=0||b1<0||b1>=1||b2<0||b2>=1||eps<=0)throw Error('Invalid Adam domain');
 const mn=b1*m+(1-b1)*g,vn=b2*v+(1-b2)*g*g;
 const mh=mn/(1-b1**t),vh=vn/(1-b2**t),delta=lr*mh/(Math.sqrt(vh)+eps);
 return {p:p-delta,m:mn,v:vn,mhat:mh,vhat:vh,delta};
}
function matrix(flat,rows,cols){return Array.from({length:rows},(_,i)=>flat.slice(i*cols,(i+1)*cols));}
function mm(x,w){return x.map(row=>w[0].map((_,j)=>row.reduce((s,v,k)=>s+v*w[k][j],0)));}
function add(a,b){return a.map((row,i)=>row.map((v,j)=>v+b[i][j]));}
function norm(x,w,b){return x.map(row=>{let mu=row.reduce((s,v)=>s+v,0)/row.length;let variance=row.reduce((s,v)=>s+(v-mu)**2,0)/row.length;return row.map((v,i)=>(v-mu)/Math.sqrt(variance+1e-5)*w[i]+b[i]);});}
function softmax(x){let max=Math.max(...x);let e=x.map(v=>Math.exp(v-max));let s=e.reduce((a,b)=>a+b,0);return e.map(v=>v/s);}
function weights(record,stage='after'){
 let out={};for(const i of record.before){let f=stage==='after'?i.after.p:i.p;out[i.name]=i.shape.length===1?f.slice():matrix(f,i.shape[0],i.shape[1]);}return out;
}
function forward(w,c,tokens){
 if(!Array.isArray(tokens)||tokens.length<1||tokens.length>c.block_size||tokens.some(i=>!Number.isInteger(i)||i<0||i>=c.vocab_size))throw Error('Invalid tokens');
 let x=tokens.map((v,i)=>w['tok_emb.w'][v].map((q,j)=>q+w['pos_emb.w'][i][j]));let D=c.d_model,H=c.n_head,dh=D/H,T=tokens.length;
 const ln=(x,p)=>norm(x,w[p+'.w'],w[p+'.b']);
 for(let l=0;l<c.n_layer;l++){
  let p='blocks.'+l+'.',qkv=mm(ln(x,p+'ln1'),w[p+'attn.qkv.w']);
  let result=Array.from({length:T},()=>Array(D).fill(0));
  for(let h=0;h<H;h++)for(let i=0;i<T;i++){
   let scores=Array.from({length:i+1},(_,j)=>{let sum=0;for(let k=0;k<dh;k++)sum+=qkv[i][h*dh+k]*qkv[j][D+h*dh+k];return sum/Math.sqrt(dh);});
   let att=softmax(scores);
   for(let k=0;k<dh;k++)for(let j=0;j<=i;j++)result[i][h*dh+k]+=att[j]*qkv[j][2*D+h*dh+k];
  }
  x=add(x,mm(result,w[p+'attn.proj.w']));
  let z=mm(ln(x,p+'ln2'),w[p+'mlp.fc.w']).map(row=>row.map(v=>.5*v*(1+Math.tanh(Math.sqrt(2/Math.PI)*(v+.044715*v**3)))));
  x=add(x,mm(z,w[p+'mlp.proj.w']));
 }
 return mm(ln(x,'ln_f'),w['head.w']);
}
function nll(logits,targets){return logits.reduce((s,row,i)=>{let max=Math.max(...row);return s+Math.log(row.reduce((a,v)=>a+Math.exp(v-max),0))-(row[targets[i]]-max);},0)/targets.length;}
function encode(text,vocabulary,block){if(!text.length||Array.from(text).length>block)throw Error('Use 1 to '+block+' characters');let ids=Array.from(text).map(c=>vocabulary.indexOf(c));if(ids.some(i=>i<0))throw Error('Character not in the synthetic vocabulary');return ids;}
root.Replay={adam,weights,forward,softmax,nll,encode};if(typeof module!=='undefined')module.exports=root.Replay;
})(typeof globalThis!=='undefined'?globalThis:this);
