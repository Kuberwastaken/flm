const names = { flm: 'FLM', gru: 'GRU', transformer: 'Transformer' };
const colors = { flm: '#a74c20', gru: '#497569', transformer: '#666277' };
const $ = id => document.getElementById(id);
const number = value => value.toLocaleString();
let chartObserver;
function row(values) {
  const tr = document.createElement('tr');
  values.forEach((value, index) => {
    const cell = document.createElement(index ? 'td' : 'th'); cell.textContent = value;
    if (!index) cell.scope = 'row'; tr.append(cell);
  });
  return tr;
}
function svg(name, attributes, text = '') {
  const element = document.createElementNS('http://www.w3.org/2000/svg', name);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, String(value));
  element.textContent = text; return element;
}

export async function loadBabyLMStudy() {
  try {
    const response = await fetch(`${import.meta.env.BASE_URL}research/babylm-validation.json`);
    if (!response.ok) throw new Error(`BabyLM snapshot unavailable (${response.status}).`);
    const report = await response.json();
    $('babylm-snapshot').textContent = `${report.runs.filter(run => run.complete).length} of ${report.registered_runs} runs complete. Snapshot: ${new Date(report.snapshot_utc).toLocaleString()}. The table uses validation measurements; final test scores will follow the completed comparison.`;
    const choose = () => report.runs.filter(run => run.scale === $('babylm-scale').value && run.seed === Number($('babylm-seed').value));
    function table() {
      const chosen = choose(), step = $('babylm-step').value ? Number($('babylm-step').value) : null;
      const component = $('babylm-component').value;
      $('babylm-comparison-rows').replaceChildren(...Object.entries(names).map(([variant, label]) => {
        const run = chosen.find(run => run.variant === variant);
        const point = run?.validation.find(point => point.step === step)?.components[component];
        const parameters = run?.parameters ?? report.runs.find(run => run.variant === variant && run.parameters)?.parameters;
        return row([label, parameters ? number(parameters) : 'Awaiting run', point ? point.bits_per_byte.toFixed(4) : 'Awaiting shared update', point ? point.token_perplexity.toFixed(2) : '—']);
      }));
    }
    function draw() {
      const chosen = choose(), component = $('babylm-component').value;
      const previousStep = $('babylm-step').value;
      const common = chosen.length === 3 ? chosen[0].validation.map(point => point.step).filter(step => step > 0 && chosen.every(run => run.validation.some(point => point.step === step))) : [];
      $('babylm-step').replaceChildren(...(common.length ? common.map(step => new Option(number(step), String(step))) : [new Option('Awaiting shared update', '')]));
      $('babylm-step').value = common.includes(Number(previousStep)) ? previousStep : common.length ? String(common.at(-1)) : '';
      $('babylm-step').disabled = !common.length;
      $('babylm-run-status').textContent = chosen.map(run => `${names[run.variant]}: ${run.validation.length ? `${number(run.validation.at(-1).step)} / ${number(report.registered_updates)} measured updates${run.complete ? ' · complete' : ''}` : 'awaiting measurements'}`).join(' · ');
      const exposure = report.registered_updates * 1536 / report.training_text_tokens[$('babylm-scale').value];
      $('babylm-exposure').textContent = `Each finished run presents 18,432,000 tokens, about ${(exposure * 100).toFixed(1)}% of this corpus's token count. Windows are sampled with replacement, so this is an exposure ratio, not the fraction of unique text seen.`;
      const chart = $('babylm-validation-chart'); chart.replaceChildren(svg('title', {}, 'BabyLM validation loss by training updates. Compare curves at the same update.'));
      const width = Math.max(320, Math.min(780, chart.getBoundingClientRect().width));
      chart.setAttribute('viewBox', `0 0 ${width} 320`);
      const values = chosen.flatMap(run => run.validation.map(point => point.components[component].bits_per_byte));
      if (values.length) {
        const low = Math.floor(Math.min(...values) * 2) / 2, high = Math.max(low + .5, Math.ceil(Math.max(...values) * 2) / 2);
        const x = step => 56 + step / report.registered_updates * (width - 88), y = value => 256 - (value - low) / (high - low) * 220;
        for (let i = 0; i <= 4; i++) {
          const value = low + (high - low) * i / 4;
          chart.append(svg('path', { d: `M56 ${y(value)}H${width - 32}`, stroke: '#ddd7cd', fill: 'none' }), svg('text', { x: 46, y: y(value) + 4, 'text-anchor': 'end' }, value.toFixed(2)));
        }
        for (const step of [0, 3000, 6000, 9000, 12000]) chart.append(svg('text', { x: x(step), y: 279, 'text-anchor': 'middle' }, number(step)));
        chart.append(svg('text', { x: width / 2, y: 306, 'text-anchor': 'middle' }, 'Training updates'), svg('text', { x: 56, y: 18 }, width < 550 ? 'Validation bits / byte' : 'Validation bits / UTF-8 byte · lower is better'));
        for (const run of chosen) {
          chart.append(svg('path', { d: run.validation.map((point, i) => `${i ? 'L' : 'M'}${x(point.step)} ${y(point.components[component].bits_per_byte)}`).join(' '), stroke: colors[run.variant], 'stroke-width': 2.5, fill: 'none' }));
          for (const point of run.validation) chart.append(svg('circle', { cx: x(point.step), cy: y(point.components[component].bits_per_byte), r: 2.5, fill: colors[run.variant] }));
        }
      } else chart.append(svg('text', { x: 24, y: 100 }, 'No published measurements yet.'));
      table();
    }
    for (const id of ['babylm-scale', 'babylm-seed', 'babylm-component']) $(id).onchange = draw;
    $('babylm-step').onchange = table; draw();
    chartObserver?.disconnect();
    let lastWidth = $('babylm-validation-chart').getBoundingClientRect().width;
    chartObserver = new ResizeObserver(entries => {
      const width = entries[0].contentRect.width;
      if (width > 0 && Math.abs(width - lastWidth) > 1) { lastWidth = width; draw(); }
    });
    chartObserver.observe($('babylm-validation-chart'));
  } catch (error) { $('babylm-snapshot').textContent = error.message; }
}
