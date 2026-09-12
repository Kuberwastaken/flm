import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { FLM } from '../web/model.js';
import { BrowserBaseline } from '../web/baseline-model.js';
import catalog from '../web/model-catalog.json' with {type:'json'};

const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const difference = (a,b) => { assert.equal(a.length,b.length); return Math.max(...a.map((v,i)=>Math.abs(v-b[i]))); };
test('automatic default uses only the shared BabyLM FLM validation panel', () => {
  assert.equal(Object.values(catalog.models).filter(x=>!x.preview).length,18);
  assert.equal(Object.values(catalog.models).filter(x=>x.preview).length,4);
  const eligible=Object.entries(catalog.models).filter(([,v])=>v.study==='babylm' && v.architecture==='flm');
  assert.equal(catalog.default_flm,eligible.sort((a,b)=>a[1].validation_bpb-b[1].validation_bpb)[0][0]);
  for(const [path,hash] of Object.entries(catalog.inputs)) assert.equal(sha(readFileSync(new URL('../'+path,import.meta.url))),hash);
});
for(const [id,item] of Object.entries(catalog.models)) test(`${id}: package hashes and full-logit Python parity across the context window`, t => {
  const base=new URL(`../public/models/${item.path}/`,import.meta.url), read=name=>readFileSync(new URL(name,base));
  assert.equal(sha(read('model.json')),item.manifest_sha256);
  const config=JSON.parse(read('model.json')), raw=read('weights.bin'), truth=JSON.parse(read('parity.json'));
  assert.equal(sha(raw),config.weights_sha256);
  assert.equal(sha(read('tokenizer.json')),config.browser_tokenizer_sha256);
  if(config.anatomy_sha256) assert.equal(sha(read('anatomy.json')),config.anatomy_sha256);
  const binary=raw.buffer.slice(raw.byteOffset,raw.byteOffset+raw.byteLength);
  const m=item.architecture==='flm'?new FLM(config,binary):new BrowserBaseline(config,binary);
  let max=0;
  for(let i=0;i<truth.tokens.length;i++) {
    m.step(truth.tokens[i]); const expected=truth.cases.find(x=>x.length===i+1);
    if(expected) {
      const error=difference(m.logits,expected.logits); max=Math.max(max,error);
      assert.ok(error<.0005,`step ${i+1}: maximum logit error ${error}`);
      if(expected.h) { assert.ok(difference(m.h,expected.h)<.00005); assert.ok(difference(m.slow,expected.slow)<.00005); }
    }
  }
  t.diagnostic(`Maximum absolute logit error: ${max}`);
  if(item.architecture!=='flm') { assert.equal(m.h.length,0); assert.equal(m.slow.length,0); }
  if(item.architecture==='transformer') assert.ok(m.cache.every(x=>x.length===95));
  m.reset(); const first=m.step(truth.tokens[0]).slice();
  m.step(123); m.reset(); assert.deepEqual(m.step(truth.tokens[0]),first);
});
