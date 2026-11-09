'use strict';
// Chrome creates DevToolsActivePort before HTTP discovery is necessarily serving.
// Only this pre-test readiness step retries. Actual inference assertions never do.
async function waitForDebugger(url,{fetchFn=globalThis.fetch,sleep=ms=>new Promise(r=>setTimeout(r,ms)),alive=()=>true,attempts=1200,timeoutMs=30000}={}){
 let last;const deadline=Date.now()+timeoutMs;
 for(let i=0;i<attempts&&alive()&&Date.now()<deadline;i++){
  try{const r=await fetchFn(url,{signal:AbortSignal.timeout(1000)});if(!r.ok)throw Error('debugger HTTP '+r.status);const tabs=await r.json();const page=tabs.find(t=>t.type==='page'&&typeof t.webSocketDebuggerUrl==='string');if(page)return page;last=Error('debugger has no page target');}catch(e){last=e;}
  if(i+1<attempts)await sleep(25);
 }
 throw new Error('Chrome debugger endpoint did not become ready: '+(last?.message||'browser exited'),{cause:last});
}
module.exports={waitForDebugger};
