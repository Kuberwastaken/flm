import { loadFinalStudy } from './final-study.js';
import { loadBabyLMStudy } from './babylm-study.js';
import { loadWiringStudy } from './wiring-study.js';
import { loadLanguageTopology } from './language-topology.js';
import { loadLanguageCore } from './language-core.js';
import { loadFeedbackStudy } from './closed-loop.js';
import { loadChoiceReplay } from './choice-replay.js';
import { loadFoodReplay } from './food-replay.js';
const names = {flm: 'FLM', gru: 'GRU', transformer: 'Transformer'};
const colors = {flm: '#a74c20', gru: '#497569', transformer: '#666277'};
const $ = id => document.getElementById(id);
const svgElement = (name, attributes, text = '') => {
  const node = document.createElementNS('http://www.w3.org/2000/svg', name);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, String(value));
  node.textContent = text; return node;
};
async function get(path) {
  const response = await fetch(`${import.meta.env.BASE_URL}research/${path}`);
  if (!response.ok) throw new Error(`Research artifact unavailable (${response.status}).`);
  return response.json();
}

export async function loadResearch() {
  void loadFinalStudy();
  void loadBabyLMStudy();
  void loadWiringStudy();
  void loadLanguageTopology();
  void loadLanguageCore();
  void loadFeedbackStudy();
  void loadChoiceReplay();
  loadFoodReplay();
  try {
    const [report, samples] = await Promise.all([get('validation.json'), get('samples-index.json')]);
    const runs = report.runs.filter(x => names[x.variant]);
    const seeds = [...new Set(runs.map(x => x.seed))].sort();
    $('comparison-seed').replaceChildren(...seeds.map(seed => new Option(String(seed), String(seed))));
    function draw() {
      const chosen = runs.filter(x => x.seed === Number($('comparison-seed').value));
      const common = chosen.length === 3 ? chosen[0].validation.map(p => p.step).filter(step => step > 0 && chosen.every(r => r.validation.some(p => p.step === step))) : [];
      $('comparison-step').replaceChildren(...common.map(step => new Option(step.toLocaleString(), String(step))));
      $('comparison-step').value = String(common.at(-1));
      $('comparison-step').disabled = !common.length;
      const chart = $('validation-chart'); chart.replaceChildren();
      chart.append(svgElement('title', {}, 'Validation loss against matched training updates; lower is better.'));
      const x = step => 64 + step / 6000 * 670, y = value => 266 - (value - 1.5) / 2.1 * 230;
      for (const value of [1.5, 2, 2.5, 3, 3.5]) {
        chart.append(svgElement('path', {d: `M64 ${y(value)}H734`, stroke:'#ddd7cd', fill:'none'}));
        chart.append(svgElement('text', {x:52, y:y(value)+4, 'text-anchor':'end'}, value.toFixed(1)));
      }
      for (const step of [0, 1500, 3000, 4500, 6000]) chart.append(svgElement('text', {x:x(step), y:288, 'text-anchor':'middle'}, step.toLocaleString()));
      chart.append(svgElement('text', {x:399, y:313, 'text-anchor':'middle'}, 'Training updates · 1,536 input tokens per update'));
      chart.append(svgElement('text', {x:64, y:18}, 'Validation bits / UTF-8 byte'));
      for (const run of chosen) chart.append(svgElement('path', {d:run.validation.map((p, i) => `${i ? 'L' : 'M'}${x(p.step)} ${y(p.bits_per_byte)}`).join(' '), stroke:colors[run.variant], 'stroke-width':2.5, fill:'none'}));
      $('comparison-progress').textContent = chosen.map(r => `${names[r.variant]}: ${r.validation.at(-1).step.toLocaleString()} / 6,000 updates${r.complete ? ' · complete' : ' · in progress'}`).join(' — ');
      table();
    }
    function table() {
      const step = $('comparison-step').value ? Number($('comparison-step').value) : null, seed = Number($('comparison-seed').value);
      $('comparison-rows').replaceChildren(...Object.keys(names).map(variant => {
        const run = runs.find(x => x.seed === seed && x.variant === variant);
        const row = document.createElement('tr'), point = run?.validation.find(x => x.step === step);
        const parameters = run?.parameters ?? runs.find(x => x.variant === variant)?.parameters;
        for (const [i, value] of [names[variant], parameters?.toLocaleString() ?? 'Pending', point?.bits_per_byte.toFixed(3) ?? 'Pending', point?.token_perplexity.toFixed(1) ?? 'Pending'].entries()) {
          const cell = document.createElement(i ? 'td' : 'th'); cell.textContent = value; row.append(cell);
        }
        return row;
      }));
    }
    $('comparison-seed').onchange = draw; $('comparison-step').onchange = table; draw();
    $('sample-checkpoint').replaceChildren(...samples.map(x => new Option(`${x.step.toLocaleString()} updates · seed ${x.seed}`, x.file)));
    $('sample-checkpoint').value = samples.at(-1)?.file ?? '';
    let passages;
    async function loadSamples() {
      if (!$('sample-checkpoint').value) return;
      try {
        const data = await get($('sample-checkpoint').value); passages = data.models;
        $('sample-prompt').replaceChildren(...passages[0].passages.map((x, i) => new Option(x.prompt, String(i))));
        $('sample-settings').textContent = `Unedited continuations · temperature ${data.settings.temperature} · top-k ${data.settings.top_k} · sampling seed ${data.settings.seed} · up to ${data.settings.maximum_tokens} tokens. Examples show language patterns and failures; they are not factual answers.`;
        $('sample-download').href = `${import.meta.env.BASE_URL}research/${$('sample-checkpoint').value}`; showSamples();
      } catch (error) { $('sample-settings').textContent = error.message; }
    }
    function showSamples() {
      $('sample-passages').replaceChildren(...passages.map(model => {
        const article = document.createElement('article'), title = document.createElement('h3'), text = document.createElement('p');
        title.textContent = names[model.variant]; text.textContent = model.passages[Number($('sample-prompt').value)].continuation;
        article.append(title, text); return article;
      }));
    }
    $('sample-prompt').onchange = showSamples; $('sample-checkpoint').onchange = loadSamples; await loadSamples();
  } catch (error) { $('comparison-progress').textContent = error.message; }
}
