import vm from 'node:vm';import{readFileSync}from'node:fs';import assert from'node:assert/strict';
const elements=new Map();const storage=new Map();
const document={querySelector:s=>{if(!elements.has(s))elements.set(s,{innerHTML:'',textContent:'',classList:{add(){},remove(){}}});return elements.get(s)},addEventListener(){}};
const context=vm.createContext({document,location:{hash:''},window:{addEventListener(){}},localStorage:{getItem:k=>storage.get(k),setItem:(k,v)=>storage.set(k,v)},console,Date,JSON,setTimeout,Number,Math,Object,String,Array,confirm:()=>true});
vm.runInContext(readFileSync('curriculum.js','utf8')+'\n'+readFileSync('labs.js','utf8')+'\n'+readFileSync('app.js','utf8'),context);
vm.runInContext(`
if(weeks.length!==24||weeks.reduce((n,w)=>n+w.tasks.length,0)!==120)throw Error('Invalid curriculum');
for(const p of navigation){location.hash='#'+p[0];filter='All';render();if(!$('#content').innerHTML.length)throw Error('Empty view '+p[0])}
for(const l of labs){location.hash='#labs/'+l.id;render();if(!$('#content').innerHTML.includes(l.name))throw Error('Lab view '+l.id)}
if(!weeks.every(w=>w.concepts.length>=3))throw Error('Missing concepts');
for(const id of Object.values(weekLab))if(!labs.some(l=>l.id===id))throw Error('Bad week lab '+id);
for(const w of weeks)if(!resources[w.resource])throw Error('Missing resource '+w.resource);
{let w=0,b=0;for(let i=0;i<300;i++)({w,b}=gdStep(w,b,0.05));if(Math.abs(w-2.5)>0.4||gdLoss(w,b)>0.2)throw Error('GD did not converge '+w);let d=0,e=0;for(let i=0;i<60;i++)({w:d,b:e}=gdStep(d,e,0.6));if(Number.isFinite(gdLoss(d,e))&&gdLoss(d,e)<1e4)throw Error('GD should diverge at lr 0.6')}
{const t='the lowest the newest the widest',m=bpeTrain(t,30),enc=bpeEncode(t,m);if(enc.join('').replace(/·/g,' ')!==t)throw Error('BPE roundtrip');if(enc.length>=[...t].length)throw Error('BPE did not compress');if(bpeEncode('xyz',m).join('')!=='xyz')throw Error('BPE unknown word')}
{const L=samplingTokens.map(t=>t[1]);const g=sampleDist(L,0,10,1).final;if(g[0]!==1||g.filter(Boolean).length!==1)throw Error('Greedy');const k=sampleDist(L,1,3,1).final;if(k.filter(Boolean).length!==3||Math.abs(k.reduce((a,b)=>a+b)-1)>1e-9)throw Error('top-k');const p=sampleDist(L,1,10,0.5).final;if(p.filter(Boolean).length>=10)throw Error('top-p')}
{if(retrieve('android main thread blocked')[0].title!=='Android ANR')throw Error('Retrieval rank');if(retrieve('memory during generation')[0].title!=='KV cache')throw Error('Retrieval KV')}
if(agentSteps.length!==4||agentSteps.at(-1).stop!=='end_turn')throw Error('Agent steps');
state.done['1-0']=true;save();if(total()!==1)throw Error('Progress count');
if(!validImport(JSON.parse(JSON.stringify(state))))throw Error('Roundtrip validation');
if(validImport({version:1}))throw Error('Invalid backup accepted');
state.sessions.push({date:'2026-11-17',week:'<script>',minutes:1,text:'bad'});if(validImport(state))throw Error('Invalid session accepted');
`,context);
assert.ok(storage.size===1);console.log('PASS: eight views, five labs (GD, BPE, sampling, retrieval, agent logic), 24 weeks / 120 tasks with concepts, progress, backup roundtrip and invalid-import rejection.');
