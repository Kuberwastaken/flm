const names = { flm: 'FLM', gru: 'GRU', transformer: 'Transformer' };
const $ = id => document.getElementById(id);
const number = value => value.toLocaleString(undefined, { maximumFractionDigits: 0 });
const signed = value => `${value > 0 ? '+' : ''}${value.toFixed(4)}`;

function row(values) {
  const element = document.createElement('tr');
  values.forEach((value, i) => {
    const cell = document.createElement(i ? 'td' : 'th'); cell.textContent = value;
    if (!i) cell.scope = 'row'; element.append(cell);
  });
  return element;
}
async function verifiedArtifact(record) {
  if (!/^[a-z-]+\.json$/.test(record.file)) throw new Error('Invalid study artifact path.');
  const response = await fetch(`${import.meta.env.BASE_URL}research/${record.file}`);
  if (!response.ok) throw new Error(`Study artifact unavailable (${response.status}).`);
  const bytes = await response.arrayBuffer();
  const digest = [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))].map(x => x.toString(16).padStart(2, '0')).join('');
  if (digest !== record.sha256) throw new Error('Study artifact checksum mismatch. Reload after deployment finishes.');
  return JSON.parse(new TextDecoder().decode(bytes));
}

export async function loadFinalStudy() {
  try {
    const response = await fetch(`${import.meta.env.BASE_URL}research/study-index.json`);
    if (!response.ok) throw new Error(`Study index unavailable (${response.status}).`);
    const index = await response.json();
    if (index.test) {
      const report = await verifiedArtifact(index.test), score = report.runs[0].score;
      $('test-progress').textContent = `${number(score.documents.length)} complete held-out articles · ${number(score.tokens)} text tokens · ${number(score.bytes)} UTF-8 bytes. Checkpoints were selected on validation before these articles were scored.`;
      $('test-rows').replaceChildren(...report.aggregates.map(aggregate => {
        const scores = aggregate.seeds.map(seed => report.runs.find(run => run.variant === aggregate.variant && run.seed === seed).score.bits_per_byte);
        return row([names[aggregate.variant], ...scores.map(x => x.toFixed(4)), aggregate.mean_bpb.toFixed(4), aggregate.seed_standard_deviation.toFixed(4)]);
      }));
      $('test-intervals').replaceChildren(...report.paired_comparisons.map(result =>
        row([`FLM − ${names[result.second]}`, result.training_seed, signed(result.difference_bpb), `${signed(result.lower_95)} to ${signed(result.upper_95)}`])));
      const flm = report.aggregates.find(x => x.variant === 'flm');
      $('test-interpretation').textContent = report.aggregates.filter(x => x.variant !== 'flm').map(other => {
        const delta = flm.mean_bpb - other.mean_bpb;
        return `FLM's mean loss is ${Math.abs(delta).toFixed(4)} bits/byte ${delta > 0 ? 'higher' : 'lower'} than ${names[other.variant]}'s in this study.`;
      }).join(' ') + ' Lower is better. These results compare the declared compact configurations, not every possible recurrent model or transformer.';
      $('ngram-result').textContent = `The separate four-byte-context n-gram scores ${report.ngram.score.bits_per_byte.toFixed(4)} bits/byte on the same complete test articles. Its smoothing was selected on validation; its parameter count and training procedure are not matched to the neural models.`;
      $('test-caution').textContent = report.caution;
      $('test-results').hidden = false;
    }
    if (index.runtime) {
      const report = await verifiedArtifact(index.runtime);
      $('runtime-rows').replaceChildren(...report.results.map(result => row([names[result.variant],
        number(result.median_prefill_tokens_per_second), number(result.median_decode_tokens_per_second), number(result.state.allocated_tensor_bytes)])));
      $('runtime-description').textContent = `Measured CPU inference · batch ${report.batch} · ${report.threads} threads · ${report.dtype}. Median of five trials with identical 128-token prefixes and 128 forced next tokens, after training jobs stopped. Tokenization is excluded.`;
      $('runtime-caution').textContent = report.caveat + ' State counts include owned tensor storage, excluding weights, temporary activations, Python bookkeeping and local adapters.';
      $('runtime-results').hidden = false;
    }
  } catch (error) { $('test-progress').textContent = error.message; }
}
