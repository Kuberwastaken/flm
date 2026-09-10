import './style.css';
import { loadResearch } from './research.js';
import { BrainView, FlyView } from './views.js';
import { STORAGE_KEY, ADAPTER_KEY, validateConversations, conversationForModel, contextFor, download } from './storage.js';

const $ = id => document.getElementById(id);
const selectedModel = new URLSearchParams(location.search).get('model') === 'ami' ? 'ami' : 'wikitext';
const isLexical = selectedModel === 'wikitext';
const adapterKey = () => `${ADAPTER_KEY}-${config.weights_sha256}`;
const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
let ready = false, busy = false, operation = '', sequence = 0, activeId = 0, researchLoaded = false;
let config, brain, fly, anatomy, lastState, selected = 0, disabled = new Set(), lastPrompt = '';
let conversations = [], currentId, pendingMessage, pendingElement, trace = [], storageBlocked = false;

function notice(message) { $('notice').textContent = message; $('notice').hidden = !message; }
function store() {
  if (storageBlocked) return;
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations)); }
  catch { notice('Browser storage is unavailable or full. You can keep using ChatFLM and export your conversations.'); }
}
function current() { return conversations.find(x => x.id === currentId); }
function setBusy(value) {
  busy = value;
  document.querySelectorAll('[data-idle]').forEach(element => { element.disabled = !ready || value; });
  for (const id of ['generate', 'rename-chat', 'delete-chat']) $(id).disabled ||= !current();
  $('stop').disabled = !value || operation !== 'generate';
  $('stop-learning').disabled = !value || operation !== 'learn';
  $('prompt').readOnly = value;
  $('learning-text').readOnly = value; $('probe-text').readOnly = value;
}
function run(type, data = {}) {
  if (busy) return false;
  operation = type; activeId = ++sequence; setBusy(true); notice('');
  worker.postMessage({ type, id: activeId, ...data }); return true;
}
function choices() {
  $('conversation').replaceChildren(...conversations.map(item => {
    const option = document.createElement('option'); option.value = item.id; option.textContent = item.title; return option;
  }));
  if (!current()) $('conversation').prepend(new Option('Choose a saved conversation', ''));
  $('conversation').value = currentId;
}
function makeMessage(message) {
  const article = document.createElement('article'); article.className = `message ${message.role}`;
  const header = document.createElement('div'); header.className = 'message-header';
  const title = document.createElement('strong'); title.textContent = message.role === 'user' ? 'You' :
    `FLM${message.dataset ? ` · ${message.dataset}` : ''}${message.checkpointStep ? ` · checkpoint ${message.checkpointStep.toLocaleString()}` : ' · earlier session'}`;
  const copy = document.createElement('button'); copy.textContent = 'Copy'; copy.type = 'button'; copy.setAttribute('aria-label', `Copy ${message.role === 'user' ? 'your' : 'FLM'} message`);
  copy.onclick = async () => { try { await navigator.clipboard.writeText(message.text); copy.textContent = 'Copied'; setTimeout(() => copy.textContent = 'Copy', 1600); } catch { notice('Clipboard unavailable. Select the message text to copy it.'); } };
  header.append(title, copy); const body = document.createElement('div'); body.className = 'message-text'; body.textContent = message.text;
  article.append(header, body); $('messages').append(article); return article;
}
function renderConversation() {
  const conversation = current();
  if (!conversation) {
    choices(); $('messages').replaceChildren();
    const message = document.createElement('p'); message.className = 'empty';
    message.textContent = 'The archive holds 100 conversations. Export or delete a saved conversation to make room for this model.';
    $('messages').append(message); setBusy(busy); return;
  }
  choices(); $('mode').value = conversation.mode; $('messages').replaceChildren();
  $('prompt-label').textContent = conversation.mode === 'dialogue' ? 'Your turn' : 'Text to continue';
  $('capability').textContent = isLexical
    ? 'Trained from scratch on WikiText-2. It completes written passages; it is not an instruction-following assistant. Each token contains one or more UTF-8 bytes.'
    : conversation.mode === 'dialogue'
    ? 'Trained on meeting transcripts. It continues dialogue; it has not been trained to follow instructions or give reliable answers.'
    : 'A next-byte predictor trained on meeting transcripts. Continue a passage and inspect the probabilities as it writes.';
  if (conversation.messages.length) conversation.messages.forEach(makeMessage);
  else {
    const empty = document.createElement('div'); empty.className = 'empty';
    const p = document.createElement('p'); p.textContent = isLexical ? 'Start a passage about history, nature or everyday life. Watch the next-token distribution and the recurrent state as it continues.' : 'Start with the kind of language it has seen: people making plans, discussing ideas and taking turns.'; empty.append(p);
    for (const text of (isLexical ? ['The history of science', 'In the summer, the village', 'The small animal moved through'] : ['what should we make together?', 'i think the design should be simple because', 'a: shall we start the meeting?\nb:'])) {
      const button = document.createElement('button'); button.textContent = text; button.onclick = () => { $('prompt').value = text; $('prompt').focus(); }; empty.append(button);
    }
    $('messages').append(empty);
  }
}
function newConversation(mode = $('mode').value || 'dialogue') {
  if (busy) return;
  if (conversations.length >= 100) { notice('The local archive holds 100 conversations. Export or delete some before starting another.'); return; }
  const item = { id: crypto.randomUUID(), title: 'New conversation', mode, modelPackage: selectedModel, messages: [] };
  conversations.unshift(item); currentId = item.id; $('prompt').value = ''; lastPrompt = ''; store(); renderConversation();
}
try {
  const saved = localStorage.getItem(STORAGE_KEY);
  if (saved) conversations = validateConversations(JSON.parse(saved));
} catch { storageBlocked = true; notice('Saved conversations could not be read. Existing storage is preserved; this session will not overwrite it. Export new conversations to keep them.'); }
const requestedConversation = new URLSearchParams(location.search).get('conversation');
const matchingConversation = conversationForModel(conversations, selectedModel, requestedConversation);
if (matchingConversation) { currentId = matchingConversation.id; renderConversation(); }
else if (conversations.length < 100) newConversation(isLexical ? 'completion' : 'dialogue');
else { currentId = ''; renderConversation(); }
$('model').value = selectedModel;
$('mode').querySelector('[value="dialogue"]').disabled = isLexical;
$('prompt').placeholder = isLexical ? 'The history of science' : 'What should we make together?';
function switchModel(value, conversationId = null) {
  const url = new URL(location.href); url.searchParams.set('model', value);
  if (conversationId) url.searchParams.set('conversation', conversationId); else url.searchParams.delete('conversation');
  location.assign(url.href);
}
$('model').onchange = () => switchModel($('model').value);

function showPage() {
  const page = ['chat', 'learn', 'research'].includes(location.hash.slice(1)) ? location.hash.slice(1) : 'chat';
  if (page === 'research' && !researchLoaded) { researchLoaded = true; loadResearch(); }
  $('experiment').hidden = page === 'research'; $('research-page').hidden = page !== 'research';
  $('chat-page').hidden = page !== 'chat'; $('learn-page').hidden = page !== 'learn';
  document.querySelectorAll('[data-page]').forEach(link => {
    if (link.dataset.page === page) link.setAttribute('aria-current', 'page'); else link.removeAttribute('aria-current');
  });
  document.title = `${page === 'chat' ? 'ChatFLM' : page === 'learn' ? 'Local learning' : 'Research'} — Fly Language Model`;
  requestAnimationFrame(() => { brain?.resize(); fly?.resize(); });
}
window.addEventListener('hashchange', showPage); showPage();
$('new-chat').onclick = () => newConversation();
$('conversation').onchange = () => {
  const target = conversations.find(x => x.id === $('conversation').value);
  if (!target) return;
  if ((target.modelPackage || 'ami') !== selectedModel) { switchModel(target.modelPackage || 'ami', target.id); return; }
  currentId = target.id; lastPrompt = ''; renderConversation();
};
$('mode').onchange = () => {
  const mode = $('mode').value;
  if (!current() || current().messages.length) newConversation(mode);
  else { current().mode = mode; store(); renderConversation(); }
};
$('rename-chat').onclick = () => { $('conversation-name').value = current().title; $('rename-dialog').showModal(); };
$('rename-dialog').addEventListener('close', () => { if ($('rename-dialog').returnValue === 'save') { current().title = $('conversation-name').value.trim() || 'Untitled'; store(); choices(); } });
$('delete-chat').onclick = () => $('delete-dialog').showModal();
$('delete-dialog').addEventListener('close', () => {
  if ($('delete-dialog').returnValue !== 'delete') return;
  conversations = conversations.filter(x => x.id !== currentId);
  const next = conversationForModel(conversations, selectedModel);
  if (!next) newConversation(isLexical ? 'completion' : 'dialogue');
  else { currentId = next.id; lastPrompt = ''; store(); renderConversation(); }
});
function showExport(filename, text) {
  $('export-dialog').dataset.filename = filename;
  $('export-json').value = text;
  $('export-status').textContent = `${filename} · ${new TextEncoder().encode(text).length.toLocaleString()} UTF-8 bytes`;
  $('export-dialog').showModal();
}
$('download-export').onclick = () => download($('export-dialog').dataset.filename, $('export-json').value);
$('copy-export').onclick = async () => {
  try { await navigator.clipboard.writeText($('export-json').value); $('export-status').textContent = 'JSON copied. Save it in a .json file to import later.'; }
  catch { $('export-json').focus(); $('export-json').select(); $('export-status').textContent = 'Use Ctrl / ⌘ + C to copy the selected JSON.'; }
};
$('export-dialog').addEventListener('close', () => { $('export-json').value = ''; });
$('export-chats').onclick = () => showExport('chatflm-conversations.json', JSON.stringify(conversations, null, 2));
async function importFile(input, limit, callback) {
  const file = input.files?.[0]; if (!file) return;
  try { if (file.size > limit) throw new Error('This file is too large.'); await callback(await file.text()); }
  catch (error) { notice(error.message); }
  finally { input.value = ''; }
}
$('import-chats').onchange = () => importFile($('import-chats'), 3000000, text => {
  const imported = validateConversations(JSON.parse(text));
  const merged = [...conversations];
  for (const item of imported) merged.push({ ...item, id: crypto.randomUUID() });
  conversations = validateConversations(merged);
  const next = conversationForModel(conversations, selectedModel, conversations.at(-1)?.id);
  if (next) currentId = next.id;
  lastPrompt = '';
  store(); renderConversation(); notice(`Imported ${imported.length} conversations.`);
});

$('composer').onsubmit = event => {
  event.preventDefault(); if (!ready || busy) return;
  const text = $('prompt').value.trim(); if (!text) return;
  const conversation = current();
  if (!conversation || (conversation.modelPackage || 'ami') !== selectedModel) { notice('Select a conversation for this model before continuing.'); return; }
  if (conversation.messages.length >= 198) { notice('This conversation is full. Start a new one to continue.'); return; }
  const message = { role: 'user', text }; conversation.messages.push(message);
  const prompt = contextFor(conversation);
  if (new TextEncoder().encode(prompt).length > 16000) { conversation.messages.pop(); notice('This context exceeds 16,000 UTF-8 bytes. Start a new conversation or shorten your turn.'); return; }
  const temperature = Number($('temperature').value), topK = Number($('top-k').value), limit = Number($('limit').value), seed = Number($('seed').value);
  if (!(temperature >= 0.05 && temperature <= 2 && Number.isInteger(topK) && topK >= 1 && topK <= config.vocabulary && Number.isInteger(limit) && limit >= 1 && limit <= 1600 && Number.isInteger(seed) && seed >= 0 && seed <= 4294967295)) {
    conversation.messages.pop(); notice(`Check the generation settings: temperature 0.05–2, top-k 1–${config.vocabulary}, maximum tokens 1–1,600 and a nonnegative 32-bit integer seed.`); return;
  }
  if (conversation.title === 'New conversation') conversation.title = text.replace(/\s+/g, ' ').slice(0, 55);
  renderConversation(); pendingMessage = { role: 'model', text: '', checkpointStep: config.checkpoint_step,
    modelHash: config.weights_sha256, dataset: config.dataset || 'AMI', createdAt: new Date().toISOString(), settings: {
      seed, temperature, topK, limit, limitUnit: 'tokens', adaptation: $('adapter-enabled').checked,
      recurrence: $('recurrence').checked, disabled: [...disabled] } }; conversation.messages.push(pendingMessage);
  pendingElement = makeMessage(pendingMessage); pendingElement.classList.add('pending');
  $('prompt').value = ''; trace = []; lastPrompt = prompt; store();
  run('generate', { prompt, temperature, topK, limit, seed, observe: $('observe').checked });
};
$('prompt').onkeydown = event => { if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) { event.preventDefault(); $('composer').requestSubmit(); } };
for (const id of ['stop', 'stop-learning']) $(id).onclick = () => { worker.postMessage({ type: 'stop' }); $(id).disabled = true; };

function selectNeuron(index) {
  selected = index; $('neuron').value = String(index); brain?.select(index); neuronInfo();
}
function neuronInfo() {
  if (!anatomy) return;
  const sign = anatomy.source_sign[selected];
  $('neuron-info').textContent = `ID ${anatomy.body_ids[selected]} · ${anatomy.cell_types[selected] || 'untyped'} · ${sign > 0 ? 'positive' : sign < 0 ? 'negative' : 'zero'} source sign` +
    (lastState ? ` · fast ${lastState.h[selected].toFixed(4)} · slow ${lastState.slow[selected].toFixed(4)}` : '') +
    (anatomy.positions[selected] ? '' : ' · soma position unavailable');
  $('mute-neuron').textContent = disabled.has(selected) ? 'Restore' : 'Silence';
}
function byteName(token) {
  if (token === 32) return 'space'; if (token === 10) return '↵'; if (token === 9) return 'tab';
  return token >= 33 && token <= 126 ? String.fromCharCode(token) : `0x${token.toString(16).padStart(2, '0')}`;
}
function updateState(state) {
  lastState = state; brain?.update(state, $('state-mode').value); fly?.pose(state, $('body-response').checked); neuronInfo();
  $('probabilities').replaceChildren(...state.top.map(item => {
    const li = document.createElement('li'), code = document.createElement('code'), score = document.createElement('span');
    code.textContent = item.label ?? byteName(item.token); code.title = `Token ${item.token}`; score.textContent = `${(100 * item.probability).toFixed(1)}%`; li.append(code, score); return li;
  }));
  trace.push(state.meanActivity); if (trace.length > 80) trace.shift();
  $('activity-path').setAttribute('d', trace.map((v, i) => `${i ? 'L' : 'M'}${i * 260 / Math.max(1, trace.length - 1)} ${75 - v * 74}`).join(' '));
  $('activity-value').textContent = `${state.meanActivity.toFixed(3)} · ${trace.length} observed updates · scale 0–1`;
}
function controls() {
  run('controls', { recurrence: $('recurrence').checked, adaptation: $('adapter-enabled').checked, disabled: [...disabled], prompt: lastPrompt });
  $('intervention-note').textContent = `${disabled.size} silenced neuron${disabled.size === 1 ? '' : 's'}. Controls replay the same input from zero state. This is an engineered intervention, not a biological experiment.`;
}
$('neuron').onchange = () => selectNeuron(Number($('neuron').value));
$('mute-neuron').onclick = () => { disabled.has(selected) ? disabled.delete(selected) : disabled.add(selected); neuronInfo(); controls(); };
$('recurrence').onchange = controls; $('adapter-enabled').onchange = controls;
$('reset-interventions').onclick = () => { disabled.clear(); $('recurrence').checked = true; $('adapter-enabled').checked = true; neuronInfo(); controls(); };
$('state-mode').onchange = () => { if (lastState) brain?.update(lastState, $('state-mode').value); };
$('context-points').onchange = () => brain?.context($('context-points').checked);
$('body-response').onchange = () => fly?.pose(lastState, $('body-response').checked);
$('brain-home').onclick = () => brain?.home(); $('brain-left').onclick = () => brain?.rotate(-0.3); $('brain-right').onclick = () => brain?.rotate(0.3);
$('fly-home').onclick = () => fly?.home();

$('learning-file').onchange = () => importFile($('learning-file'), 6000, text => { $('learning-text').value = text; });
$('learning-form').onsubmit = event => {
  event.preventDefault(); if (!ready || busy) return;
  const text = $('learning-text').value, probe = $('probe-text').value;
  if (!text.trim()) { notice('Add some text to learn from.'); return; }
  if (probe.trim() && text.includes(probe.trim())) { notice('The probe appears inside the training text. Use separate text to measure transfer.'); return; }
  $('adapter-enabled').checked = true; $('learning-result').hidden = true; $('learning-progress').hidden = false; $('learning-progress').value = 0;
  $('learning-status').textContent = 'Starting local updates…'; trace = [];
  run('learn', { text, probe, epochs: Number($('epochs').value), rate: Number($('learning-rate').value) });
};
$('save-learning').onclick = () => run('export', { destination: 'storage' });
$('export-learning').onclick = () => run('export', { destination: 'file' });
$('restore-learning').onclick = () => {
  try { const text = localStorage.getItem(adapterKey()) || localStorage.getItem(ADAPTER_KEY); if (!text) throw new Error('No learning is saved for this checkpoint.'); run('import', { adapter: JSON.parse(text) }); }
  catch (error) { notice(error.message); }
};
$('import-learning').onchange = () => importFile($('import-learning'), 12000000, text => run('import', { adapter: JSON.parse(text) }));
$('clear-learning').onclick = () => {
  run('clear'); try { localStorage.removeItem(adapterKey()); localStorage.removeItem(ADAPTER_KEY); } catch { notice('Session learning cleared; saved browser storage could not be accessed.'); }
  $('learning-result').hidden = true; $('learning-status').textContent = 'Baseline restored.';
};

async function loadViews() {
  try {
    brain = new BrainView($('brain-view'), selectNeuron); anatomy = await brain.load(config); $('brain-loading').hidden = true;
    $('neuron').replaceChildren(...anatomy.body_ids.map((id, i) => {
      const option = document.createElement('option'); option.value = String(i); option.textContent = `${anatomy.cell_types[i] || 'untyped'} · ${id}`; return option;
    }));
    const missing = anatomy.positions.filter(x => !x).length;
    $('anatomy-note').textContent = `${config.neurons.toLocaleString()} modeled neurons from ${config.source_neurons.toLocaleString()} in the source graph; ${missing} without a soma position. Gray points are anatomical context only.`;
    selectNeuron(0); if (lastState) brain.update(lastState, $('state-mode').value);
  } catch (error) { $('brain-loading').textContent = `3D brain unavailable: ${error.message} Text generation still works.`; }
  try { fly = new FlyView($('fly-view')); await fly.load(); $('fly-loading').hidden = true; if (lastState) fly.pose(lastState, $('body-response').checked); }
  catch (error) { $('fly-loading').textContent = `3D body unavailable: ${error.message}`; }
}
document.addEventListener('viewerror', event => notice(event.detail));
worker.onmessage = ({ data }) => {
  if (data.id !== activeId) return;
  if (data.type === 'loading') $('model-status').textContent = data.message;
  if (data.type === 'ready') {
    ready = true; config = data.config; $('model-status').textContent = `FLM · ${config.dataset || 'AMI'} · local`;
    $('top-k').max = config.vocabulary;
    $('checkpoint-link').href = `/${config.package_path}/model.json`;
    $('generation-stats').textContent = `Checkpoint ${config.checkpoint_step.toLocaleString()} · ready`;
    $('release-detail').textContent = `${config.trained_parameters.toLocaleString()} trained parameters · ${config.neurons.toLocaleString()} neurons · ${config.retained_edges.toLocaleString()} edges · checkpoint ${config.checkpoint_step.toLocaleString()} · ${(config.weights_bytes / 1000000).toFixed(2)} MB browser weights. The full anatomical graph is not the compact model.`;
    loadViews();
  }
  if (data.type === 'priming') $('generation-stats').textContent = `Reading context: ${data.done.toLocaleString()} / ${data.total.toLocaleString()} tokens`;
  if (data.type === 'state') updateState(data);
  if (data.type === 'generation') {
    if (pendingMessage) {
      pendingMessage.text = data.text; pendingElement.querySelector('.message-text').textContent = data.text;
      const view = $('messages'); if (view.scrollHeight - view.scrollTop - view.clientHeight < 140) view.scrollTop = view.scrollHeight;
    }
    $('generation-stats').textContent = `${data.tokens} tokens · ${data.bytes} bytes · ${(data.tokens / Math.max(.001, data.seconds)).toFixed(1)} tokens/s${$('observe').checked ? ' including playback delay' : ''}`; updateState(data);
  }
  if (data.type === 'learning' || data.type === 'learned') {
    $('learning-progress').value = data.done / data.total;
    $('learning-status').textContent = `${data.done.toLocaleString()} / ${data.total.toLocaleString()} token updates`;
    updateState(data);
    if (data.type === 'learned') {
      const result = $('learning-result'); result.hidden = false;
      result.textContent = `Online training loss: ${data.trainingBpb.toFixed(3)} bits/byte. ` +
        (data.before !== null && data.after !== null ? `Separate probe: ${data.before.toFixed(3)} → ${data.after.toFixed(3)} bits/byte (${data.after < data.before ? 'improved' : 'did not improve'}). Lower is better. ` : 'No separate probe was supplied; this does not measure generalization. ') +
        'These are local measurements on your text, not the release benchmark.';
    }
  }
  if (data.type === 'adapter') {
    if (data.destination === 'file') showExport('flm-local-learning.json', JSON.stringify(data.adapter));
    else try { localStorage.setItem(adapterKey(), JSON.stringify(data.adapter)); notice('Learning saved for this checkpoint. Use Load saved after reopening.'); }
    catch { notice('Browser storage is full or unavailable. Export learning to a file instead.'); }
  }
  if (data.type === 'notice') notice(data.message);
  if (data.type === 'error') notice(data.message);
  if (data.type === 'idle') {
    if (operation === 'generate') {
      pendingElement?.classList.remove('pending');
      if (pendingMessage && !pendingMessage.text) { const conversation = current(); conversation.messages = conversation.messages.filter(x => x !== pendingMessage); pendingElement?.remove(); }
      store(); pendingMessage = null; pendingElement = null;
      if (data.cancelled) $('generation-stats').textContent += ' · stopped';
    }
    if (operation === 'learn' && data.cancelled) $('learning-status').textContent += ' · stopped; completed updates retained';
    setBusy(false);
    if (!ready) { $('model-status').textContent = 'Model unavailable'; notice(`${$('notice').textContent} Reload the page to retry.`); }
  }
};
worker.onerror = event => { ready = false; setBusy(false); $('model-status').textContent = 'Worker unavailable'; notice(`The model worker stopped: ${event.message || 'unknown error'}. Reload to restart.`); };
setBusy(false); run('load', {model: selectedModel});
