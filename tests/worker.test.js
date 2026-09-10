import test from 'node:test';
import assert from 'node:assert/strict';
import { Worker } from 'node:worker_threads';
import { once } from 'node:events';
import { readFileSync } from 'node:fs';
import { TextCodec } from '../web/text-codec.js';
import { MODEL_PACKAGES } from '../web/packages.js';

for (const selection of ['ami', 'wikitext', 'babylm']) test(`${selection}: worker generates, scores, learns and cancels with true token/byte accounting`, async t => {
  const worker = new Worker(new URL('./helpers/worker-harness.mjs', import.meta.url));
  t.after(() => worker.terminate());
  await once(worker, 'message'); let sequence = 0;
  const run = (type, data = {}, onMessage = null) => new Promise((resolve, reject) => {
    const id = ++sequence, messages = [];
    const timer = setTimeout(() => { worker.off('message', listener); reject(new Error(`Timed out in ${type}`)); }, 30000);
    const listener = message => {
      if (message.id !== id) return;
      messages.push(message); onMessage?.(message);
      if (message.type === 'idle') {
        clearTimeout(timer); worker.off('message', listener);
        const error = messages.find(x => x.type === 'error');
        if (error) reject(new Error(error.message)); else resolve(messages);
      }
    };
    worker.on('message', listener); worker.postMessage({type, id, ...data});
  });
  const loaded = await run('load', {model: selection});
  const config = loaded.find(x => x.type === 'ready').config;
  const tokenizer = MODEL_PACKAGES[selection].lexical ? JSON.parse(readFileSync(new URL(`../public/models/${MODEL_PACKAGES[selection].path}/tokenizer.json`, import.meta.url))) : null;
  const codec = new TextCodec(config, tokenizer);
  const settings = {prompt: 'The history of science', limit: 16, seed: 42, temperature: .8, topK: 40};
  const a = (await run('generate', settings)).filter(x => x.type === 'generation').at(-1);
  const b = (await run('generate', settings)).filter(x => x.type === 'generation').at(-1);
  assert.equal(a.text, b.text); assert.deepEqual(a.h, b.h);
  assert.ok(a.tokens > 0 && a.tokens <= 16); assert.ok(a.bytes >= a.tokens);
  assert.ok(a.top.every(x => typeof x.label === 'string' && Number.isFinite(x.probability)));
  const text = 'The little bird lives in a quiet garden.\n', probe = 'A bird in the garden.';
  const learned = (await run('learn', {text, probe, epochs: 1, rate: .03})).find(x => x.type === 'learned');
  assert.equal(learned.done, codec.encode(text).length);
  assert.ok([learned.trainingBpb, learned.before, learned.after].every(Number.isFinite));
  await run('clear');
  const c = (await run('generate', settings)).filter(x => x.type === 'generation').at(-1);
  assert.equal(a.text, c.text); assert.deepEqual(a.h, c.h);
  const state = (await run('controls', {recurrence: true, adaptation: true, disabled: [3], prompt: settings.prompt})).find(x => x.type === 'state');
  assert.equal(state.h[3], 0); assert.equal(state.slow[3], 0);
  const cancelled = await run('generate', {...settings, prompt: 'The history of science. '.repeat(200)}, message => {
    if (message.type === 'priming') worker.postMessage({type: 'stop'});
  });
  assert.equal(cancelled.at(-1).cancelled, true);
  assert.equal(cancelled.some(x => x.type === 'generation'), false);
});
