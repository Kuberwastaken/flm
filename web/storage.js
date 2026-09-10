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
        !['dialogue', 'completion'].includes(item.mode) || !Array.isArray(item.messages) || item.messages.length > 200 ||
        (item.modelPackage !== undefined && !['ami', 'wikitext'].includes(item.modelPackage)) ||
        (item.modelPackage === 'wikitext' && item.mode !== 'completion'))
      throw new Error('Invalid conversation file.');
    ids.add(item.id);
    for (const message of item.messages) {
      if (!['user', 'model'].includes(message?.role) || typeof message.text !== 'string') throw new Error('Invalid message.');
      size += message.text.length;
    }
  }
  if (size > 2000000) throw new Error('Conversation archive is too large.');
  return value;
}

export function contextFor(conversation) {
  if (conversation.mode === 'completion') return conversation.messages.map(x => x.text).join('');
  return conversation.messages.map(x => `${x.role === 'user' ? 'a' : 'b'}: ${x.text.trim()}`).join('\n') + '\nb:';
}

export function download(name, content, type = 'application/json') {
  const link = document.createElement('a');
  const url = URL.createObjectURL(new Blob([content], { type }));
  link.href = url; link.download = name; link.hidden = true;
  document.body.append(link); link.click(); link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 60000);
}
