import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { ByteBPE } from '../web/tokenizer.js';
const specification = JSON.parse(readFileSync(new URL('../data/tokenizers/wikitext2-4096/browser-tokenizer.json', import.meta.url)));
const fixture = JSON.parse(readFileSync(new URL('./fixtures/tokenizer-parity.json', import.meta.url)));
test('browser BPE matches the independent Rust/Python tokenizer on Unicode and whitespace cases', () => {
  const tokenizer = new ByteBPE(specification);
  assert.equal(fixture.tokenizer_sha256, specification.tokenizer_sha256);
  for (const { text, tokens } of fixture.cases) {
    assert.deepEqual(tokenizer.encode(text), tokens, JSON.stringify(text));
    assert.equal(tokenizer.decode(tokens), text);
    assert.equal(tokenizer.decode(tokenizer.encode(text, true)), text);
  }
});
test('literal boundary-marker text is encoded as text and cache cannot mutate previous outputs', () => {
  const tokenizer = new ByteBPE(specification), tokens = tokenizer.encode('[BOS] <|eos|>');
  assert.ok(tokens.every(x => x >= 2)); tokens[0] = 0;
  assert.ok(tokenizer.encode('[BOS] <|eos|>').every(x => x >= 2));
});
