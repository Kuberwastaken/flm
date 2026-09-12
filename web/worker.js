import { FLM, random, sample, softmax } from './model.js';
import { TextCodec } from './text-codec.js';
import { MODEL_PACKAGES } from './packages.js';
import { chatOutput } from './chat-format.js';
import { BrowserBaseline } from './baseline-model.js';

let model, codec, active = null;
const encoder = new TextEncoder();
const pause = () => new Promise(resolve => setTimeout(resolve, 0));
const send = (type, data = {}) => postMessage({ type, id: active?.id, ...data });

function snapshot() {
  const probabilities = softmax(model.logits);
  const top = Array.from(probabilities, (probability, token) => ({ token, probability }))
    .filter(x => x.token !== codec.bos).sort((a, b) => b.probability - a.probability).slice(0, 6)
    .map(x => ({...x, label: codec.label(x.token)}));
  return { h: model.h.slice(), slow: model.slow.slice(), top,
    meanActivity: model.n ? model.h.reduce((sum, x) => sum + Math.abs(x), 0) / model.n : 0 };
}

async function prime(text) {
  const bytes = encoder.encode(text);
  if (bytes.length > 16000) throw new Error('Context is limited to 16,000 UTF-8 bytes. Start a new conversation or shorten the text.');
  const tokens = codec.encode(text);
  model.reset(); model.step(codec.bos);
  for (let i = 0; i < tokens.length; i++) {
    if (active.cancelled) return false;
    model.step(tokens[i]);
    if (i % 64 === 63) { send('priming', { done: i + 1, total: tokens.length }); await pause(); }
  }
  return true;
}

async function generate(message) {
  if (!await prime(message.prompt)) return;
  const rng = random(message.seed), decoder = new TextDecoder();
  const limit = Math.max(1, Math.min(1600, Number(message.limit) || 400));
  const started = performance.now(); let text = '', count = 0, bytes = 0;
  send('state', snapshot());
  while (count < limit && !active.cancelled) {
    const token = sample(model.logits, rng, {...message, allowed: codec.allowed});
    if (token === codec.eos) break;
    const piece = codec.bytes(token); bytes += piece.length;
    text += decoder.decode(piece, { stream: true });
    model.step(token); count++;
    if (message.chat && chatOutput(text).stopped) break;
    // Send the actual recurrent state for every token, with time to paint in observation mode.
    send('generation', { text: message.chat ? chatOutput(text).text : text, bytes, tokens: count, seconds: (performance.now() - started) / 1000, ...snapshot() });
    if (message.observe) await new Promise(resolve => setTimeout(resolve, 60));
    else await pause();
  }
  text += decoder.decode();
  send('generation', { text: message.chat ? chatOutput(text, true).text : text, rawText: text,
    turnStopped: message.chat && chatOutput(text, true).stopped,
    bytes, tokens: count, seconds: (performance.now() - started) / 1000, ...snapshot() });
}

async function score(text) {
  const bytes = encoder.encode(text), tokens = codec.encode(text); model.reset(); model.step(codec.bos); let nll = 0;
  for (let i = 0; i < tokens.length; i++) {
    if (active.cancelled) return null;
    nll -= Math.log(Math.max(softmax(model.logits)[tokens[i]], 1e-30));
    model.step(tokens[i]);
    if (i % 64 === 63) await pause();
  }
  return nll / Math.max(bytes.length, 1) / Math.LN2;
}

async function learn(message) {
  const bytes = encoder.encode(message.text), probeBytes = encoder.encode(message.probe || '');
  if (!bytes.length || bytes.length > 6000 || probeBytes.length > 2000)
    throw new Error('Use 1–6,000 training bytes and at most 2,000 separate probe bytes.');
  const epochs = Number(message.epochs), rate = Number(message.rate);
  if (!Number.isInteger(epochs) || epochs < 1 || epochs > 5 || !Number.isFinite(rate) || rate <= 0 || rate > 0.2)
    throw new Error('Invalid learning settings.');
  model.adaptationEnabled = true;
  const before = message.probe ? await score(message.probe) : null;
  if (active.cancelled) return;
  const tokens = codec.encode(message.text); let count = 0, nll = 0, scoredBytes = 0;
  for (let epoch = 0; epoch < epochs; epoch++) {
    model.reset(); model.step(codec.bos);
    for (const token of tokens) {
      if (active.cancelled) return;
      nll += model.learn(token, rate); model.step(token); count++; scoredBytes += codec.bytes(token).length;
      if (count % 32 === 0) {
        send('learning', { done: count, total: tokens.length * epochs, trainingBpb: nll / scoredBytes / Math.LN2, ...snapshot() });
        await pause();
      }
    }
  }
  const after = message.probe ? await score(message.probe) : null;
  send('learned', { done: count, total: tokens.length * epochs, trainingBpb: nll / scoredBytes / Math.LN2, before, after, ...snapshot() });
}

async function load(message) {
  if (!Object.hasOwn(MODEL_PACKAGES, message.model)) throw new Error('Unknown model package.');
  const packageName = MODEL_PACKAGES[message.model].path;
  const base = `${import.meta.env?.BASE_URL ?? '/'}models/${packageName}/`;
  const configResponse = await fetch(`${base}model.json`);
  if (!configResponse.ok) throw new Error(`Model manifest unavailable (${configResponse.status}).`);
  const manifest = await configResponse.arrayBuffer();
  const expected = MODEL_PACKAGES[message.model].manifest_sha256;
  if (expected) {
    const hash = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', manifest)), x => x.toString(16).padStart(2, '0')).join('');
    if (hash !== expected) throw new Error('Model manifest checksum mismatch. Reload to fetch a consistent release.');
  }
  const config = JSON.parse(new TextDecoder().decode(manifest));
  send('loading', { message: `Downloading ${(config.weights_bytes / 1000000).toFixed(2)} MB checkpoint…` });
  const response = await fetch(`${base}weights.bin`);
  if (!response.ok) throw new Error(`Model download failed (${response.status}).`);
  const binary = await response.arrayBuffer();
  const digest = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', binary)), x => x.toString(16).padStart(2, '0')).join('');
  if (digest !== config.weights_sha256) throw new Error('Model checksum mismatch. Reload to fetch a consistent release.');
  let tokenizer = null;
  if (['flm-browser-v2','flm-baseline-browser-v1'].includes(config.format)) {
    const response = await fetch(`${base}tokenizer.json`);
    if (!response.ok) throw new Error('Tokenizer download failed.');
    const payload = await response.arrayBuffer();
    const hash = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', payload)), x => x.toString(16).padStart(2, '0')).join('');
    if (hash !== config.browser_tokenizer_sha256) throw new Error('Tokenizer checksum mismatch.');
    tokenizer = JSON.parse(new TextDecoder().decode(payload));
  }
  const loaded = config.format === 'flm-baseline-browser-v1' ? new BrowserBaseline(config, binary) : new FLM(config, binary), loadedCodec = new TextCodec(config, tokenizer);
  model = loaded; codec = loadedCodec; config.package_path = `models/${packageName}`; send('ready', { config });
}

self.onmessage = async ({ data: message }) => {
  if (message.type === 'stop') { if (active) active.cancelled = true; return; }
  if (active) { postMessage({ type: 'error', id: message.id, message: 'Wait for the current operation to finish.' }); return; }
  active = { id: message.id, cancelled: false };
  try {
    if (message.type === 'load') await load(message);
    else {
      if (!model) throw new Error('The model is still loading.');
      if (message.type === 'generate') await generate(message);
      else if (message.type === 'learn') await learn(message);
      else if (message.type === 'export') send('adapter', { adapter: model.exportLearning(), destination: message.destination });
      else if (message.type === 'import') { model.importLearning(message.adapter); send('notice', { message: 'Learning restored for this checkpoint.' }); }
      else if (message.type === 'clear') { model.clearLearning(); send('notice', { message: 'Local learning cleared. Original readout restored.' }); }
      else if (message.type === 'controls') {
        model.recurrenceEnabled = Boolean(message.recurrence);
        model.adaptationEnabled = Boolean(message.adaptation);
        model.disabled.fill(0);
        for (const index of message.disabled || []) {
          if (Number.isInteger(index) && index >= 0 && index < model.n) model.disabled[index] = 1;
        }
        if (message.prompt && await prime(message.prompt)) send('state', snapshot());
      } else throw new Error('Unknown operation.');
    }
  } catch (error) { send('error', { message: error.message }); }
  finally { send('idle', { cancelled: active.cancelled }); active = null; }
};
