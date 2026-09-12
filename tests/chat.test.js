import test from 'node:test';
import assert from 'node:assert/strict';
import { chatContext, validateConversations, contextFor } from '../web/storage.js';
import { chatOutput, repetitionLoop } from '../web/chat-format.js';

test('chat encodes explicit roles and optional system context without altering the archive',()=>{
  const conversation={id:'one',title:'Example',mode:'chat',modelPackage:'babylm-100m-flm-s43',systemPrompt:'Be brief.',messages:[{role:'user',text:'Hello'},{role:'model',text:'Hi'},{role:'user',text:'Why?'}]};
  const saved=structuredClone(conversation);
  assert.equal(contextFor(conversation),'System: Be brief.\n\nUser: Hello\nAssistant: Hi\nUser: Why?\nAssistant:');
  assert.deepEqual(validateConversations([conversation]),[saved]);
  assert.deepEqual(conversation,saved);
  assert.throws(()=>validateConversations([{...conversation,systemPrompt:'a'.repeat(2001)}]));
});
test('UTF-8 context trimming keeps system context and recent complete turns',()=>{
  const messages=[{role:'user',text:'古'.repeat(30)},{role:'model',text:'Old reply'},{role:'user',text:'New question'}];
  const result=chatContext({systemPrompt:'Brief',messages},80);
  assert.equal(result.prompt,'System: Brief\n\nUser: New question\nAssistant:');
  assert.equal(result.dropped,2); assert.ok(result.bytes<=80); assert.equal(messages.length,3);
  assert.throws(()=>chatContext({systemPrompt:'Long',messages:[{role:'user',text:'古'.repeat(30)}]},80));
});
test('streaming chat hides complete and partial next-turn markers while retaining raw text separately',()=>{
  for(const marker of ['\nUser:','\nSystem:','\nAssistant:']) {
    for(let i=1;i<marker.length;i++) assert.deepEqual(chatOutput('Answer'+marker.slice(0,i)),{text:'Answer',stopped:false});
    assert.deepEqual(chatOutput('Answer'+marker+' invented turn'),{text:'Answer',stopped:true});
  }
  assert.deepEqual(chatOutput('Hello\nUse this'),{text:'Hello\nUse this',stopped:false});
  assert.deepEqual(chatOutput('Hello\nUs',true),{text:'Hello\nUs',stopped:false});
  assert.deepEqual(chatOutput(''),{text:'',stopped:false});
});

test('chat repetition detection catches noisy letter/digit loops but keeps ordinary text', () => {
  assert.ok(repetitionLoop('B'.repeat(9) + 'A' + 'B'.repeat(14)));
  assert.ok(repetitionLoop('Here is a reply. ' + 'again '.repeat(8)));
  assert.ok(repetitionLoop('12'.repeat(11) + '39' + '12'.repeat(12)));
  for (const text of ['Hello. What would you like to test?', 'The answer is 11111111111.', '01234567890123456789012345678901234567890123456789', 'A:\tYes.\nB:\tI see.']) assert.equal(repetitionLoop(text), null);
});
test('follow-up input excludes looped assistant replies, preserving users and the full archive', () => {
  const conversation = {mode:'chat', systemPrompt:'Be brief.', messages:[
    {role:'user',text:'What is the meaning of life'},
    {role:'model',text:'B'.repeat(100)},
    {role:'user',text:'test'},
    {role:'model',text:'Partial response',rawText:'Partial response'+'12'.repeat(30),stopReason:'repetition'},
    {role:'user',text:'so true bro; '+'B'.repeat(20)}]};
  const saved = structuredClone(conversation), input = chatContext(conversation);
  assert.equal(input.excludedRepetitiveReplies,2);
  assert.equal(input.prompt,'System: Be brief.\n\nUser: What is the meaning of life\nUser: test\nUser: so true bro; '+'B'.repeat(20)+'\nAssistant:');
  assert.deepEqual(conversation,saved);
  const raw = chatContext({...conversation,loopProtection:false});
  assert.equal(raw.excludedRepetitiveReplies,0); assert.ok(raw.prompt.includes('Assistant: '+'B'.repeat(100)));
});
