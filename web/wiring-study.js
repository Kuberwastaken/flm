const $ = id => document.getElementById(id);
const percent = value => `${(100 * value).toFixed(2)}%`;
const difference = value => `${value > 0 ? '+' : ''}${(100 * value).toFixed(2)}`;

function row(values) {
  const element = document.createElement('tr');
  values.forEach((value, index) => {
    const cell = document.createElement(index ? 'td' : 'th');
    if (!index) cell.scope = 'row';
    cell.textContent = value; element.append(cell);
  });
  return element;
}

export async function loadWiringStudy() {
  const controls = ['wiring-task', 'wiring-delay', 'wiring-step'];
  try {
    const response = await fetch(`${import.meta.env.BASE_URL}research/wiring-learning.json`);
    if (!response.ok) throw new Error('Wiring comparison is unavailable.');
    const report = await response.json();
    if (report.runs !== 60 || report.seeds.length !== 3) throw new Error('The complete wiring comparison is required.');
    const render = () => {
      const task = $('wiring-task').value, delay = Number($('wiring-delay').value), step = Number($('wiring-step').value);
      const conditions = report.curves.filter(point => point.task === task && point.delay === delay && point.step === step);
      const summaryRows = [], seedRows = [];
      for (const [method, label] of Object.entries(report.methods)) {
        const measured = conditions.find(point => point.method === method && point.topology === 'measured').current_accuracy;
        const randomized = conditions.find(point => point.method === method && point.topology === 'null').current_accuracy;
        const contrast = report.contrasts.find(point => point.task === task && point.method === method && point.step === step && point.delay === delay).measured_minus_null;
        summaryRows.push(row([label, percent(measured.mean), percent(randomized.mean), `${difference(contrast.mean)} pp (${difference(contrast.minimum)} to ${difference(contrast.maximum)})`]));
        report.seeds.forEach((seed, index) => seedRows.push(row([`${label} · seed ${seed}`, percent(measured.values[index]), percent(randomized.values[index]), `${difference(contrast.values[index])} pp`])));
      }
      $('wiring-rows').replaceChildren(...summaryRows); $('wiring-seed-rows').replaceChildren(...seedRows);
      const phase = step === 0 ? 'Initial checkpoint, original rule' : step > 300 && step <= 600 ? 'Reversed rule' : step > 600 ? 'Restored original rule' : 'Original rule';
      $('wiring-selection').textContent = `${report.tasks[task]} · ${phase} · delay ${delay}${delay > 12 ? ', beyond the training delays' : ', within the training range'}. Means and paired difference ranges span three model/null seed pairs.`;
    };
    controls.forEach(id => $(id).addEventListener('change', render)); render();
  } catch (error) {
    $('wiring-selection').textContent = error.message;
    controls.forEach(id => { $(id).disabled = true; });
  }
}
