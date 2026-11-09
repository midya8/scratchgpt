'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path'),{spawn,execFileSync}=require('node:child_process'),{waitForDebugger}=require('./chrome_support.js');
const delay=ms=>new Promise(r=>setTimeout(r,ms));
function compare(a,b,p='root'){assert.equal(typeof a,typeof b,p);if(typeof a==='number')assert.ok(Number.isFinite(b)&&Math.abs(a-b)<=1e-11*Math.max(1,Math.abs(a),Math.abs(b)),p);else if(a===null||typeof a!=='object')assert.equal(a,b,p);else{assert.deepEqual(Object.keys(a).sort(),Object.keys(b).sort(),p);for(const k of Object.keys(a))compare(a[k],b[k],p+'.'+k);}}
test('actual Chrome complete training replay, coordinate tables, independent inference and exports',{timeout:180000},async()=>{
 const server=require('node:http').createServer((req,res)=>{try{const file=path.join(__dirname,'../docs',req.url==='/'?'index.html':req.url);const types={'.html':'text/html','.js':'application/javascript','.css':'text/css','.json':'application/json'};res.setHeader('Content-Type',types[path.extname(file)]||'text/plain');res.end(fs.readFileSync(file));}catch(e){res.statusCode=404;res.end('not found');}});await new Promise(r=>server.listen(0,'127.0.0.1',r));
 const profile=fs.mkdtempSync(path.join(os.tmpdir(),'scratchgpt-audit-')),browser=spawn(process.env.CHROME_BIN||'google-chrome',['--headless','--no-sandbox','--disable-gpu','--disable-extensions','--disable-dev-shm-usage','--remote-debugging-port=0',`--user-data-dir=${profile}`,'about:blank'],{stdio:['ignore','ignore','pipe']});let ws,startupError,diagnostics='';browser.on('error',e=>startupError=e);browser.stderr.on('data',d=>diagnostics=(diagnostics+d).slice(-4000));
 try{for(let i=0;i<1200&&!startupError&&browser.exitCode===null&&!fs.existsSync(path.join(profile,'DevToolsActivePort'));i++)await delay(25);assert.ok(fs.existsSync(path.join(profile,'DevToolsActivePort')),startupError?.message||diagnostics||'Chrome unavailable');const port=fs.readFileSync(path.join(profile,'DevToolsActivePort'),'utf8').split('\n')[0],tab=await waitForDebugger(`http://127.0.0.1:${port}/json/list`,{alive:()=>browser.exitCode===null&&!startupError});ws=new WebSocket(tab.webSocketDebuggerUrl);await new Promise((res,rej)=>{ws.onopen=res;ws.onerror=rej;});let seq=0;const pending=new Map();ws.onmessage=e=>{const m=JSON.parse(e.data);if(pending.has(m.id)){const {res,rej}=pending.get(m.id);pending.delete(m.id);m.error?rej(Error(JSON.stringify(m.error))):res(m.result);}};const call=(method,params={})=>new Promise((res,rej)=>{const id=++seq;pending.set(id,{res,rej});ws.send(JSON.stringify({id,method,params}));});const js=async expression=>{const r=await call('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;};
 await call('Page.enable');await call('Page.navigate',{url:process.env.SCRATCH_URL||`http://127.0.0.1:${server.address().port}/`});
 for(let i=0;i<500;i++){if(await js('Boolean(window.replayReady)'))break;await delay(20);}assert.ok(await js('window.replayReady'));

 const d=require('../docs/replay.json'),R=require('../docs/engine.js'),fmt=x=>Number(x).toPrecision(8);
 const table=async id=>js(`Array.from(document.querySelectorAll('#'+${JSON.stringify(id)}+' tbody tr'),tr=>Array.from(tr.cells,c=>c.textContent))`);
 const set=async(id,value,event='change')=>js(`document.getElementById(${JSON.stringify(id)}).value=${JSON.stringify(String(value))};document.getElementById(${JSON.stringify(id)}).dispatchEvent(new Event(${JSON.stringify(event)}))`);
 assert.deepEqual(await table('boundaries'),d.boundaries.map(b=>[String(b.split),'SAME',fmt(b.legacy_weight_delta),fmt(b.missing_moment_delta),b.eval_only_history_equal?'same':'changed']));
 let cells=0,views=0,inferences=0;
 for(let ri=0;ri<d.records.length;ri++){
  const r=d.records[ri];await set('step',ri);
  for(let pi=0;pi<r.before.length;pi++){
   const item=r.before[pi];await set('parameter',pi);
   for(let page=0;page*32<item.p.length;page++){
    const start=page*32,end=Math.min(start+32,item.p.length),expected=[];
    for(let k=start;k<end;k++)expected.push([String(k),...[item.p[k],item.g[k],item.m[k],item.v[k],item.after.p[k],item.after.m[k],item.after.v[k]].map(fmt)]);
    assert.deepEqual(await table('coordinates'),expected);cells+=expected.length*8;
    if(end<item.p.length)await js('document.getElementById("next").click()');
   }
   for(const k of [0,Math.floor(item.p.length/2),item.p.length-1]){
    await set('coordinate',k,'input');const vals=await js('Array.from(document.querySelectorAll("#coordinate-values dd"),e=>e.textContent)');
    assert.deepEqual(vals,[item.name,item.shape.join(' x '),String(k),...[item.p[k],item.g[k],item.m[k],item.v[k],item.after.p[k],item.after.m[k],item.after.v[k]].map(fmt)]);
    const got=R.adam(item.p[k],item.g[k],item.m[k],item.v[k],r.step,r.lr);
    assert.equal(await js('document.getElementById("scalar-result").textContent'),"p' = "+fmt(got.p)+"; m' = "+fmt(got.m)+"; v' = "+fmt(got.v)+"; delta = "+fmt(got.delta));views++;
   }
  }
  for(let b=0;b<3;b++)for(const stage of ['before','after']){
   await set('batch',b);await set('stage',stage);await js('document.getElementById("use-batch").click()');
   let l=R.forward(R.weights(r,stage),d.config,r.x[b]),prob=R.softmax(l.at(-1));
   assert.deepEqual(await table('logits'),d.vocabulary.map((c,i)=>[JSON.stringify(c),fmt(l.at(-1)[i]),fmt(prob[i])]));
   assert.deepEqual(await table('all-logits'),r.x[b].map((t,i)=>[String(i),JSON.stringify(d.vocabulary[t]),...l[i].map(fmt)]));inferences++;
  }
 }
 await js('document.getElementById("g").value="0.25";document.getElementById("g").dispatchEvent(new Event("input"))');assert.equal(await js('document.getElementById("scalar-result").className'),'');
 for(const [id,bad] of [['g',''],['lr','0'],['b1','1'],['b2','-1'],['eps','0'],['coordinate','-1'],['coordinate','']]){await set('parameter',0);await set(id,bad,'input');assert.ok(await js('document.getElementById("scalar-result").textContent.length>0'));}
 for(const bad of ['', 'z','abcdefghi']){await set('prompt',bad,'input');assert.equal(await js('document.getElementById("inference-output").hidden'),true);assert.equal(await js('document.getElementById("download-inference").disabled'),true);assert.deepEqual(await table('logits'),[]);}
 await set('step',4);await set('prompt','red fo','input');await js('document.getElementById("prompt").focus()');await call('Input.dispatchKeyEvent',{type:'keyDown',key:'Enter',code:'Enter',windowsVirtualKeyCode:13});await call('Input.dispatchKeyEvent',{type:'keyUp',key:'Enter',code:'Enter',windowsVirtualKeyCode:13});assert.equal(await js('document.getElementById("inference-output").hidden'),false);
 await js('HTMLAnchorElement.prototype.click=function(){window.downloaded={name:this.download,url:this.href}};URL.revokeObjectURL=function(){}');
 await js('document.getElementById("download-update").click()');const exported=await js('fetch(downloaded.url).then(r=>r.json())');assert.deepEqual(exported.record,d.records[4]);
 await js('document.getElementById("download-inference").click()');const inf=await js('fetch(downloaded.url).then(r=>r.json())');assert.equal(inf.text,'red fo');assert.deepEqual(inf.logits,R.forward(R.weights(d.records[4],'after'),d.config,R.encode('red fo',d.vocabulary,6)));
 for(const width of [320,390,1280]){await call('Emulation.setDeviceMetricsOverride',{width,height:1000,deviceScaleFactor:1,mobile:width<700});assert.ok(await js('document.documentElement.scrollWidth<=innerWidth'),'overflow '+width);}
 console.log(JSON.stringify({coordinate_table_cells:cells,coordinate_views:views,inference_views:inferences,exports:2,widths:[320,390,1280]}));
 if(process.env.SCRATCH_SCREENSHOT){const width=Number(process.env.SCRATCH_WIDTH||1280);await call('Emulation.setDeviceMetricsOverride',{width,height:1200,deviceScaleFactor:1,mobile:width<700});await set('step',4);await set('parameter',6);await set('prompt','red fo','input');await js('document.activeElement.blur();window.scrollTo(0,0)');if(process.env.SCRATCH_EXPAND)await js('document.querySelectorAll("details").forEach(e=>e.open=true)');if(process.env.SCRATCH_RIGHT)await js('document.querySelectorAll(".table-wrap").forEach(e=>e.scrollLeft=e.scrollWidth)');const m=await call('Page.getLayoutMetrics'),image=await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:true,clip:{x:0,y:0,width,height:m.cssContentSize.height,scale:1}});fs.writeFileSync(process.env.SCRATCH_SCREENSHOT,Buffer.from(image.data,'base64'));}
 }finally{server.close();if(ws)ws.close();if(browser.pid&&browser.exitCode===null){browser.kill('SIGTERM');await new Promise(r=>browser.once('exit',r));}await fs.promises.rm(profile,{recursive:true,force:true,maxRetries:10,retryDelay:200});}
});
