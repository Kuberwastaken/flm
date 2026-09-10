const $ = id => document.getElementById(id);

export function matchedValidation(report) {
  return report.runs.filter(run => !run.reference).map(control => {
    const measured = report.runs.find(run => run.reference && run.seed === control.seed);
    const shared = control.validation.filter(point => point.step <= control.saved_step &&
      measured?.validation.some(other => other.step === point.step));
    const point = shared.at(-1);
    const reference = measured?.validation.find(other => other.step === point?.step);
    return { control, point, reference, difference: point && reference ? reference.bits_per_byte - point.bits_per_byte : null };
  });
}

export async function loadLanguageTopology() {
  try {
    const response = await fetch(`${import.meta.env.BASE_URL}research/language-topology-progress.json`);
    if (!response.ok) throw new Error('Language topology snapshot unavailable.');
    const report = await response.json();
    $('language-topology-status').textContent = `${report.completed_new_runs} of ${report.new_runs} new runs complete; two completed measured references. Snapshot: ${new Date(report.snapshot_utc).toLocaleString()}. Test comparison: ${report.test_status}.`;
    const rows = matchedValidation(report).map(({ control, point, reference, difference }) => {
      const row = document.createElement('tr');
      const label = control.variant === 'no_slow' ? 'Retrained without slow state' : `Rewired · graph ${control.graph_seed}`;
      const cells = [label, control.seed, point ? point.step.toLocaleString() : 'Awaiting checkpoint',
        reference ? reference.bits_per_byte.toFixed(4) : '—', point ? point.bits_per_byte.toFixed(4) : '—',
        difference === null ? '—' : `${difference > 0 ? '+' : ''}${difference.toFixed(4)}`];
      cells.forEach((value, index) => {
        const cell = document.createElement(index ? 'td' : 'th');
        if (!index) cell.scope = 'row';
        cell.textContent = value; row.append(cell);
      });
      return row;
    });
    $('language-topology-rows').replaceChildren(...rows);
  } catch (error) { $('language-topology-status').textContent = error.message; }
}
