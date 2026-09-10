import test from 'node:test';
import assert from 'node:assert/strict';
import { validateConversations, conversationForModel, contextFor } from '../web/storage.js';

const conversation = () => ({ id: 'one', title: 'A meeting', mode: 'dialogue', messages: [
  { role: 'user', text: 'hello' }, { role: 'model', text: 'hi' }, { role: 'user', text: 'what next?' }] });
test('dialogue context preserves turn labels and completion preserves literal text', () => {
  const c = conversation();
  assert.equal(contextFor(c), 'a: hello\nb: hi\na: what next?\nb:');
  c.mode = 'completion'; assert.equal(contextFor(c), 'hellohiwhat next?');
});
test('conversation imports reject ambiguous identities, unknown roles and oversized archives', () => {
  assert.equal(validateConversations([conversation()]).length, 1);
  assert.throws(() => validateConversations([conversation(), conversation()]));
  const c = conversation(); c.messages[0].role = 'system'; assert.throws(() => validateConversations([c]));
  c.messages[0] = { role: 'user', text: 'a'.repeat(2000001) }; assert.throws(() => validateConversations([c]));
});
test('model selection never falls back to an incompatible or legacy conversation', () => {
  const legacy = conversation(), lexical = { ...conversation(), id: 'wiki', modelPackage: 'wikitext', mode: 'completion' };
  assert.equal(conversationForModel([legacy, lexical], 'wikitext', 'one'), lexical);
  assert.equal(conversationForModel([legacy, lexical], 'ami', 'wiki'), legacy);
  assert.equal(conversationForModel([legacy], 'wikitext'), undefined);
  assert.equal(conversationForModel([], 'ami'), undefined);
  assert.throws(() => validateConversations([{ ...lexical, mode: 'dialogue' }]));
});

test('BabyLM conversations retain their own model identity through archive validation and selection', () => {
  const baby = { ...conversation(), id: 'baby', modelPackage: 'babylm', mode: 'completion' };
  const wiki = { ...baby, id: 'wiki', modelPackage: 'wikitext' };
  assert.deepEqual(validateConversations([baby, wiki]), [baby, wiki]);
  assert.equal(conversationForModel([baby, wiki], 'babylm', 'wiki'), baby);
  assert.equal(conversationForModel([wiki], 'babylm'), undefined);
  assert.throws(() => validateConversations([{ ...baby, mode: 'dialogue' }]));
  assert.throws(() => validateConversations([{ ...baby, modelPackage: 'unknown' }]));
});
