import vm from 'node:vm';import{readFileSync}from'node:fs';import assert from'node:assert/strict';
const elements=new Map();const storage=new Map();
const document={querySelector:s=>{if(!elements.has(s))elements.set(s,{innerHTML:'',textContent:'',classList:{add(){},remove(){}}});return elements.get(s)},addEventListener(){}};
const context=vm.createContext({document,location:{hash:''},window:{addEventListener(){}},localStorage:{getItem:k=>storage.get(k),setItem:(k,v)=>storage.set(k,v)},console,Date,JSON,setTimeout,Number,Math,Object,String,Array,confirm:()=>true});
vm.runInContext(readFileSync('curriculum.js','utf8')+'\n'+readFileSync('app.js','utf8'),context);
vm.runInContext(`
if(weeks.length!==24||weeks.reduce((n,w)=>n+w.tasks.length,0)!==120)throw Error('Invalid curriculum');
for(const p of navigation){location.hash='#'+p[0];filter='All';render();if(!$('#content').innerHTML.length)throw Error('Empty view '+p[0])}
state.done['1-0']=true;save();if(total()!==1)throw Error('Progress count');
if(!validImport(JSON.parse(JSON.stringify(state))))throw Error('Roundtrip validation');
if(validImport({version:1}))throw Error('Invalid backup accepted');
state.sessions.push({date:'2026-11-17',week:'<script>',minutes:1,text:'bad'});if(validImport(state))throw Error('Invalid session accepted');
`,context);
assert.ok(storage.size===1);console.log('PASS: seven views, 24 weeks / 120 tasks, progress calculation, backup roundtrip and invalid-import rejection.');
