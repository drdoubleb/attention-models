#!/usr/bin/env node
/** Validate the shipped single-file tutorial and its educational calculations.
 * No dependencies. Run: node tests/test_guide.js */
const fs=require('fs'),path=require('path'),vm=require('vm'),assert=require('assert/strict');
const html=fs.readFileSync(path.join(__dirname,'..','index.html'),'utf8');
const scripts=[...html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)];
for(const [i,m] of scripts.entries())new vm.Script(m[1],{filename:'inline-script-'+i});
const guide=html.match(/<script id="guide-core">([\s\S]*?)<\/script>/)[1];
// Evaluate definitions without booting the UI. Storage denial must be recoverable.
const code=guide.slice(0,guide.indexOf("window.addEventListener('hashchange'"));
const context=vm.createContext({localStorage:{getItem(){throw new Error('disabled');}}});
vm.runInContext(code,context);
const api=vm.runInContext('({GUIDE_LESSONS,GUIDE_LAB_MAP,GUIDE_LAB_CONTEXT,guideSoftmax,guideAttention,guideTopP,guideLoss,guideContext,guideChecked})',context);
const near=(a,b)=>assert.ok(Math.abs(a-b)<1e-10,`${a} != ${b}`);
const sum=xs=>xs.reduce((a,b)=>a+b,0);
const {guideSoftmax:softmax,guideAttention:attention,guideTopP:topP}=api;
assert.equal(api.guideChecked.size,0);
near(sum(softmax([1000,1001,1002])),1);
const stable=softmax([1000,1001,1002]);softmax([0,1,2]).forEach((v,i)=>near(v,stable[i]));
near(softmax([0,0])[0],.5);
assert.deepEqual(Array.from(softmax([2,2,1],0)),[1,0,0]);
assert.throws(()=>softmax([1,2],-1));
assert.equal(softmax([0,-Infinity])[1],0);
assert.ok(softmax([0,2],2)[1]<softmax([0,2],.5)[1]);
const masked=attention([0,0,100],[[2,0],[0,4],[999,999]],1,true);
near(masked.blended[0],1);near(masked.blended[1],2);assert.equal(masked.weights[2],0);near(sum(masked.weights),1);
const self=attention([0,100],[[2,3],[99,99]],0,true);near(self.blended[0],2);near(self.blended[1],3);
assert.ok(attention([0,0,1],[[2,0],[0,4],[9,9]],1,false).weights[2]>0);
const filtered=topP([.6,.3,.1],.8);near(filtered[0],2/3);near(filtered[1],1/3);assert.equal(filtered[2],0);near(sum(filtered),1);
topP([.6,.3,.1],1).forEach((p,i)=>near(p,[.6,.3,.1][i]));
near(api.guideLoss(.5),Math.log(2));assert.ok(api.guideLoss(.8)<api.guideLoss(.1));
assert.deepEqual(Array.from(api.guideContext(['the','cat','sat','on','the','mat'],3)),['on','the','mat']);
const lessons=api.GUIDE_LESSONS,ids=new Set(lessons.map(l=>l.id));assert.equal(lessons.length,14);assert.equal(ids.size,lessons.length);
for(const lesson of lessons){
  for(const field of ['id','group','label','title','intro','body','takeaway','question','activity'])assert.ok(lesson[field],`${lesson.id}: missing ${field}`);
  assert.equal(lesson.choices.length,lesson.feedback.length);assert.ok(lesson.answer>=0&&lesson.answer<lesson.choices.length);
  assert.ok(lesson.sources.length);for(const [,url] of lesson.sources)assert.ok(url.startsWith('https://'));
  if(lesson.lab!==undefined)assert.ok(lesson.lab>=0&&lesson.lab<15);
  assert.ok(guide.includes("kind==='"+lesson.activity+"'"),'Missing activity '+lesson.activity);
}
assert.equal(api.GUIDE_LAB_MAP.length,15);assert.equal(api.GUIDE_LAB_CONTEXT.length,15);for(const id of api.GUIDE_LAB_MAP)assert.ok(ids.has(id));
assert.ok(html.includes('id="app" hidden'));assert.ok(html.includes('id="guide-main"'));
assert.ok(!html.includes('None of the five lines changes.'));assert.ok(!html.includes('title:"Read its mind"'));
console.log(`Passed: ${scripts.length} scripts parse; 14 complete lessons; all lab links; softmax, mask, weighted values, temperature, top-p, context, loss, and unavailable storage.`);
