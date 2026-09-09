import test from 'node:test';
import assert from 'node:assert/strict';
import { validateConversations, contextFor } from '../web/storage.js';

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
