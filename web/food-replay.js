import { BrainView } from './views.js';
import { RecordedFlyView } from './recorded-fly-view.js';
import { checkedBytes, decodeJSON, decodeTrial, replayFrame, replayManifestSHA } from './food-replay-data.js';
const base = import.meta.env.BASE_URL, directory = `${base}research/food-core-replay/`;
const $ = name => document.getElementById(`food-replay-${name}`);

export function loadFoodReplay() {
  const details = $('details'); if (!details || details.dataset.bound) return;
  details.dataset.bound = 'true';
  let started = false, ready = false, failed = false, manifest, body, brain, trial, request = 0, playing = false, last = 0, frame = 0;
  const setDisabled = value => details.querySelectorAll('select, input, button').forEach(node => { node.disabled = value; });
  const stop = () => { playing = false; $('play').textContent = 'Play at ¼ speed'; };
  const fail = error => { failed = true; stop(); setDisabled(true); $('status').textContent = `Replay unavailable: ${error.message} The downloadable records remain available below.`; };
  function draw() {
    if (!trial) return;
    const state = replayFrame(trial, frame), record = trial.metadata;
    body.update(state, record.cached_thorax_mm[frame]); brain.update(state, $('state').value);
    $('time').textContent = `${record.time_s[frame].toFixed(2)} s · sample ${frame + 1}/201`; $('frame').value = String(frame);
    const p = record.probabilities[frame], x = record.sensory[frame], d = record.descending_signal[frame];
    $('values').textContent = `Action probabilities: right ${p[0].toFixed(3)}, straight ${p[1].toFixed(3)}, left ${p[2].toFixed(3)}. Motor drives: ${d.map(v => v.toFixed(3)).join(', ')}. Sugar contact: ${x[4].toFixed(0)}; source contact: ${x[5].toFixed(0)}.`;
    const m = record.metrics;
    $('outcome').textContent = m.contact_latency_censored ? 'Complete trial: no source contact by 2.00 s.' : `Complete trial: first sampled contact at ${m.first_contact_s.toFixed(2)} s with ${m.first_contact_sources.join(', ').toUpperCase()} (${m.first_contact_sugar ? 'sugar' : 'neutral'}).`;
    $('position').textContent = `Recorded thorax (mm): ${record.cached_thorax_mm[frame].map(v => v.toFixed(3)).join(', ')}. Camera follows the thorax.`;
    const index = Number($('neuron-index').value);
    if (Number.isInteger(index) && index >= 0 && index < 1024) {
      brain.select(index);
      $('neuron').textContent = `Body ID ${brain.anatomy.body_ids[index]} · fast ${state.h[index].toFixed(5)} · slow ${state.a[index].toFixed(5)} at ${record.time_s[frame].toFixed(2)} s.`;
    }
  }
  async function selectTrial() {
    stop(); const current = ++request; setDisabled(true); details.dataset.loading = 'true';
    $('values').textContent = ''; $('outcome').textContent = ''; $('neuron').textContent = '';
    $('status').textContent = 'Loading and verifying the selected recording…';
    const label = `${$('origin').value}-s${$('seed').value}--${$('case').value}`;
    try {
      const entry = manifest.trials.find(x => x.label === label);
      if (!entry) throw new Error('Condition missing from the complete inventory.');
      const metadata = decodeJSON(await checkedBytes(directory + entry.metadata_file, entry.metadata_sha256));
      const bytes = await checkedBytes(directory + metadata.binary_file, entry.binary_sha256);
      if (metadata.label !== label || metadata.binary_sha256 !== entry.binary_sha256) throw new Error('Condition identity differs.');
      const loaded = decodeTrial(metadata, bytes);
      if (current !== request) return;
      trial = loaded; body.field(metadata.field); ready = true; draw(); setDisabled(false); delete details.dataset.loading;
      $('status').textContent = 'Verified recording · all 201 samples · no interpolation or food-task updates.';
      $('record').href = directory + entry.metadata_file;
    } catch (error) { if (current === request) fail(error); }
  }
  async function start() {
    started = true; setDisabled(true); $('status').textContent = 'Loading the recorded body and brain…';
    try {
      manifest = decodeJSON(await checkedBytes(directory + 'manifest.json', replayManifestSHA));
      if (manifest.format !== 'flm-food-recorded-replay-v1' || manifest.trials.length !== 24 || manifest.observations !== 4824)
        throw new Error('Complete replay inventory differs.');
      body = new RecordedFlyView($('body')); brain = new BrainView($('brain'), index => {
        if (!trial) return;
        $('neuron-index').value = String(index); draw();
      });
      await Promise.all([body.load(manifest, base), brain.load({package_path: 'models/flm-wikitext', anatomy_sha256: manifest.anatomy_sha256})]);
      await selectTrial();
    } catch (error) { fail(error); }
  }
  details.addEventListener('toggle', () => {
    if (!details.open) { stop(); return; }
    if (!started) void start(); else if (ready && !failed) { body.resize(); brain.resize(); draw(); }
  });
  for (const name of ['origin','seed','case']) $(name).addEventListener('change', () => { void selectTrial(); });
  $('state').addEventListener('change', draw);
  $('neuron-index').addEventListener('input', draw);
  $('frame').addEventListener('input', () => { stop(); frame = Number($('frame').value); draw(); });
  $('home').addEventListener('click', () => { body.home(); brain.home(); });
  $('play').addEventListener('click', () => {
    if (playing) { stop(); return; }
    if (frame === 200) frame = 0;
    playing = true; last = 0; $('play').textContent = 'Pause'; draw();
    function tick(now) {
      if (!playing) return;
      if (!details.open || document.hidden || location.hash !== '#research') { stop(); return; }
      if (!last) last = now;
      // One exact saved frame per tick; slow down under load rather than inventing/interpolating states.
      if (now - last >= 40) { frame = Math.min(200, frame + 1); last = now; draw(); }
      if (frame === 200) stop(); else requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  });
  details.addEventListener('viewerror', () => fail(new Error('The 3D context was lost. Reload to restore the replay.')));
  if (details.open) void start();
}
