#!/usr/bin/env node
// Dependency-free documentation validation; no GitHub, archive or inference writes.
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import assert from 'node:assert/strict';
const root=path.resolve(process.argv[2]??path.dirname(fileURLToPath(import.meta.url)));
const read=p=>fs.readFileSync(path.join(root,p),'utf8');
const manifest=JSON.parse(read('manifest.json'));
const corpus=JSON.parse(read('corpus-20.json'));
const fail=(ok,msg)=>assert.ok(ok,msg);
const unique=(values,what)=>fail(new Set(values).size===values.length,`Duplicate ${what}`);
const ids=new Map(manifest.issues.map(i=>[i.id,i]));
unique(manifest.issues.map(i=>i.id),'issue IDs');
unique(manifest.issues.map(i=>i.order),'orders');
unique(manifest.labels.map(l=>l.name),'labels');
fail(manifest.schema_version===1,'Unknown schema');
fail(manifest.repository==='Enucatl/zanzara-archive','Unexpected repository');
fail(manifest.project.visibility==='PRIVATE','Project must be private');
fail(manifest.project.milestones.length===0,'No milestones');
assert.deepEqual(manifest.project.fields.Status,['Backlog','Blocked','Ready','In progress','In review','Done']);
const phases=['P0','P1','P1R','P2','P3','P4','P5','P6','P7','P8'];
assert.deepEqual(manifest.project.fields.Phase,phases);
assert.deepEqual(manifest.project.fields.Executor,['Luna','Human']);
fail(manifest.project.views.length===4,'Four project views required');
const expectedCounts={P0:6,P1:10,P1R:17,P2:9,P3:7,P4:6,P5:6,P6:4,P7:4,P8:4};
for(const [id,count] of Object.entries(expectedCounts)){
 const parent=ids.get(id);
 fail(parent?.parent===null && parent.executor==='Human' && parent.kind==='Phase',`Invalid phase ${id}`);
 for(let n=1;n<=count;n++)fail(ids.has(`${id}-${String(n).padStart(2,'0')}`),`Missing handoff child ${id}-${n}`);
 const children=manifest.issues.filter(i=>i.parent===id).map(i=>i.id);
 assert.deepEqual([...parent.children].sort(),[...children].sort(),`Child listing ${id}`);
 children.filter(child=>ids.get(child).release_blocker!==false).forEach(child=>fail(parent.blocked_by.includes(child),`${id} not blocked by child ${child}`));
}
for(const id of ['P1-H01','P1R-H01','P3-H01','P5-H01','P1-06','P2-07','P3-06','P6-02','P7-02','P7-03','P8-02']){
 fail(ids.get(id)?.executor==='Human' && ids.get(id)?.kind==='Operator',`Required human task ${id}`);
}
const headings=['Outcome and requirement','Design and interfaces','Bounded steps','Inputs, outputs and failure behavior','Exclusions','Acceptance checklist','Verification commands','Evidence and documentation','Stop conditions and completion rule'];
for(const i of manifest.issues){
 fail(Number.isInteger(i.order)&&i.order>=0,`Invalid order ${i.id}`);
 fail(manifest.project.fields.Phase.includes(i.phase),`Invalid phase ${i.id}`);
 fail(manifest.project.fields.Kind.includes(i.kind),`Invalid kind ${i.id}`);
 fail(manifest.project.fields.Executor.includes(i.executor),`Invalid executor ${i.id}`);
 fail(manifest.project.fields.Priority.includes(i.priority),`Invalid priority ${i.id}`);
 fail(i.body===`issues/${i.id}.md`,`Unexpected body path ${i.id}`);
 fail(i.title.startsWith(`[${i.id}] `),`Title missing ID ${i.id}`);
 unique(i.blocked_by,`blockers ${i.id}`);
 for(const dep of i.blocked_by)fail(ids.has(dep)&&dep!==i.id,`Bad dependency ${i.id} -> ${dep}`);
 if(i.parent){
  fail(ids.get(i.parent)?.kind==='Phase' && i.parent===i.phase,`Bad parent ${i.id}`);
  fail(!i.blocked_by.includes(i.parent),`${i.id} depends on own parent`);
 }
 const previousPhase={P1:'P0',P1R:'P0',P2:'P1',P3:'P2',P4:'P3',P5:'P4',P6:'P5',P7:'P6',P8:'P7'}[i.phase];
 if(previousPhase)fail(i.blocked_by.includes(previousPhase),`Missing previous phase blocker ${i.id}`);
 if(i.release_blocker!==undefined)fail(typeof i.release_blocker==='boolean',`Invalid release blocker ${i.id}`);
 const body=read(i.body);
 fail((body.match(/<!-- zanzara-plan:/g)??[]).length===1,`Marker count ${i.id}`);
 for(const text of [`<!-- zanzara-plan:${i.id} -->`,`# ${i.title}`,`- Executor: ${i.executor}`,`- Kind: ${i.kind}`,`- Parent: ${i.parent??'none'}`,`- Blocked by: ${i.blocked_by.join(', ')||'none'}`])fail(body.includes(text),`Missing metadata ${text}`);
 headings.forEach(h=>fail(body.includes(`## ${h}\n`),`Missing ${h} in ${i.id}`));
 fail((body.match(/^- \[ \]/gm)??[]).length>=3,`Insufficient acceptance checklist ${i.id}`);
 fail(!/\bTODO\b|\bTBD\b|lorem ipsum/i.test(body),`Unfinished body ${i.id}`);
 fail(i.initial_status===(i.blocked_by.length?'Blocked':'Ready'),`Incorrect initial state ${i.id}`);
 i.labels.forEach(l=>fail(manifest.labels.some(x=>x.name===l),`Unknown label ${l}`));
 fail(read('BACKLOG.md').includes(`](${i.body})`),`Missing backlog link ${i.id}`);
}
// Includes parent -> children release edges, so a child -> own-parent error forms a real cycle.
const visiting=new Set(),done=new Set();
function visit(id){fail(!visiting.has(id),`Dependency cycle at ${id}`);if(done.has(id))return;visiting.add(id);ids.get(id).blocked_by.forEach(visit);visiting.delete(id);done.add(id);}
ids.forEach((_,id)=>visit(id));
unique(manifest.requirements.map(r=>r.id),'requirements');
for(const r of manifest.requirements){fail(r.requirement&&r.issues.length,`Empty mapping ${r.id}`);r.issues.forEach(id=>fail(ids.has(id),`Unknown mapping ${r.id} -> ${id}`));}
const expectedDates=['260910','260907','260724','260723','260722','260721','260720','260717','260716','260715','260714','260713','260710','260709','260708','260707','260706','260703','260702','260701'];
assert.deepEqual(corpus.episodes.map(e=>e.relative_filename),expectedDates.map(d=>`${d}-lazanzara.opus`),'Frozen corpus filenames/order changed');
fail(corpus.golden_episode==='260910-lazanzara.opus','Golden episode changed');
unique(corpus.episodes.map(e=>e.sha256),'corpus hashes');
for(const e of corpus.episodes){
 fail(/^[a-f0-9]{64}$/.test(e.sha256),`Bad source hash ${e.relative_filename}`);
 fail(e.episode_date===`20${e.relative_filename.slice(0,2)}-${e.relative_filename.slice(2,4)}-${e.relative_filename.slice(4,6)}`,`Bad source date ${e.relative_filename}`);
 for(const key of ['duration_ms','sample_rate_hz','channels','size_bytes'])fail(Number.isInteger(e[key])&&e[key]>0,`Bad ${key}`);
 fail(e.codec==='opus'&&e.channels===1&&e.sample_rate_hz===48000,'Unexpected inspected audio metadata');
}
fail(corpus.episodes[0].sha256==='06de18da0691a19738bcda30dace2d5be87c8be651e9bd9ed8e56b5828f64536','Golden hash changed');
// Validate Markdown links outside fences, including section anchors; never fetch external URLs.
const slug=s=>s.toLowerCase().replace(/[^\p{L}\p{N}\s_-]/gu,'').replace(/ /g,'-');
function unfence(s){return s.replace(/^(`{3,}|~{3,}).*\n[\s\S]*?^\1\s*$/gm,'');}
function markdownFiles(dir){return fs.readdirSync(dir,{withFileTypes:true}).flatMap(e=>e.isDirectory()?markdownFiles(path.join(dir,e.name)):e.name.endsWith('.md')?[path.join(dir,e.name)]:[]);}
for(const file of markdownFiles(root)){
 const content=unfence(fs.readFileSync(file,'utf8'));
 for(const match of content.matchAll(/\[[^\]\n]*\]\(([^)\n]+)\)/g)){
  const target=match[1];if(/^(https?:|mailto:)/.test(target))continue;
  const [relative,anchor]=target.split('#');const absolute=relative?path.resolve(path.dirname(file),relative):file;
  fail(fs.existsSync(absolute),`Broken link ${path.relative(root,file)} -> ${target}`);
  if(anchor){const anchors=[...unfence(fs.readFileSync(absolute,'utf8')).matchAll(/^#{1,6} (.+)$/gm)].map(m=>slug(m[1]));fail(anchors.includes(anchor),`Broken anchor ${path.relative(root,file)} -> ${target}`);}
 }
}
const expectedFiles=new Set(manifest.issues.map(i=>path.basename(i.body)));
assert.deepEqual(fs.readdirSync(path.join(root,'issues')).filter(f=>f.endsWith('.md')).sort(),[...expectedFiles].sort(),'Orphan/missing issue bodies');
console.log(`Validated ${ids.size} issues, ${manifest.issues.reduce((n,i)=>n+i.blocked_by.length,0)} blocker edges, ${manifest.requirements.length} requirement mappings, 20 frozen episodes and all local Markdown links.`);
