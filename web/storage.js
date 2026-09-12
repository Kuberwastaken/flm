import { repetitionLoop } from './chat-format.js';
import { MODEL_PACKAGES } from './packages.js';
export const STORAGE_KEY = 'chatflm-conversations-v1';
export const ADAPTER_KEY = 'chatflm-adapter-v1';

export function conversationForModel(conversations, model, preferredId) {
  const matching = conversations.filter(item => (item.modelPackage || 'ami') === model);
  return matching.find(item => item.id === preferredId) || matching[0];
}

export function validateConversations(value) {
  if (!Array.isArray(value) || value.length > 100) throw new Error('Import must contain at most 100 conversations.');
  let size = 0;
  const ids = new Set();
  for (const item of value) {
    if (typeof item?.id !== 'string' || ids.has(item.id) || typeof item.title !== 'string' || item.title.length > 120 ||
        !['chat', 'dialogue', 'completion'].includes(item.mode) || !Array.isArray(item.messages) || item.messages.length > 200 ||
        (item.loopProtection !== undefined && typeof item.loopProtection !== 'boolean') ||
        (item.systemPrompt !== undefined && (typeof item.systemPrompt !== 'string' || item.systemPrompt.length > 2000)) ||
        (item.modelPackage !== undefined && !Object.hasOwn(MODEL_PACKAGES, item.modelPackage)) ||
        (MODEL_PACKAGES[item.modelPackage]?.lexical && item.mode === 'dialogue'))
      throw new Error('Invalid conversation file.');
    ids.add(item.id);
    size += item.systemPrompt?.length || 0;
    for (const message of item.messages) {
      if (!['user', 'model'].includes(message?.role) || typeof message.text !== 'string') throw new Error('Invalid message.');
      size += message.text.length;
    }
  }
  if (size > 2000000) throw new Error('Conversation archive is too large.');
  return value;
}

export function contextFor(conversation) {
  if (conversation.mode === 'chat') return chatContext(conversation).prompt;
  if (conversation.mode === 'completion') return conversation.messages.map(x => x.text).join('');
  return conversation.messages.map(x => `${x.role === 'user' ? 'a' : 'b'}: ${x.text.trim()}`).join('\n') + '\nb:';
}

export function chatContext(conversation, byteLimit = 16000) {
  const encode = new TextEncoder();
  let excludedRepetitiveReplies = 0;
  const turns = conversation.messages.filter(message => {
    const exclude = conversation.loopProtection !== false && message.role === 'model' &&
      (message.stopReason === 'repetition' || repetitionLoop(message.text));
    if (exclude) excludedRepetitiveReplies++;
    return !exclude;
  });
  const system = conversation.systemPrompt?.trim();
  const build = () => (system ? `System: ${system}\n\n` : '') +
    turns.map(x => `${x.role === 'user' ? 'User' : 'Assistant'}: ${x.text.trim()}`).join('\n') + '\nAssistant:';
  let prompt = build(), dropped = 0;
  while (encode.encode(prompt).length > byteLimit && turns.length > 1) {
    turns.shift(); dropped++;
    while (turns.length > 1 && turns[0].role !== 'user') { turns.shift(); dropped++; }
    prompt = build();
  }
  if (encode.encode(prompt).length > byteLimit) throw new Error('The system prompt and latest message exceed 16,000 UTF-8 bytes. Shorten them to send.');
  return { prompt, dropped, excludedRepetitiveReplies, bytes: encode.encode(prompt).length };
}

export function download(name, content, type = 'application/json') {
  const link = document.createElement('a');
  const url = URL.createObjectURL(new Blob([content], { type }));
  link.href = url; link.download = name; link.hidden = true;
  document.body.append(link); link.click(); link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 60000);
}
