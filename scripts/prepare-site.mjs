import { mkdir, copyFile, readdir, readFile, writeFile } from 'node:fs/promises';
await mkdir('public/licenses', { recursive: true });
await mkdir('public/data', { recursive: true });
await mkdir('public/research', { recursive: true });
await mkdir('public/research/selection-feasibility', { recursive: true });
for (const name of ['summary.json', 'candidate-body-ids.json', 'figures.json'])
  await copyFile(`reports/selection-feasibility/${name}`, `public/research/selection-feasibility/${name}`);
await copyFile('flm/selection_feasibility.py', 'public/research/selection-feasibility/selection_feasibility.py');
await writeFile('public/research/selection-study.md', (await readFile('docs/SELECTION-STUDY.md', 'utf8'))
  .replaceAll('../flm/selection_feasibility.py', 'selection-feasibility/selection_feasibility.py')
  .replaceAll('../reports/selection-feasibility/', 'selection-feasibility/')
  .replaceAll('PUBLISHER-ANNOTATIONS.md', 'publisher-annotations.md')
  .replaceAll('SELECTION-PATHWAYS.md', 'selection-pathways.md')
  .replaceAll('CIRCUIT-SELECTION.md', 'circuit-selection.md')
  .replaceAll('../public/research/', ''));
await mkdir('public/research/selection-pathways', { recursive: true });
for (const [folder, names, source] of [
  ['circuit-selection', ['summary.json', 'candidate-body-ids.json', 'members.csv'], 'circuit_selection.py'],
  ['selection-controls', ['summary.json', 'control-body-ids.json'], 'selection_controls.py'],
]) {
  await mkdir(`public/research/${folder}`, { recursive: true });
  for (const name of names) await copyFile(`reports/${folder}/${name}`, `public/research/${folder}/${name}`);
  await copyFile(`flm/${source}`, `public/research/${folder}/${source}`);
}
await writeFile('public/research/circuit-selection.md', (await readFile('docs/CIRCUIT-SELECTION.md', 'utf8'))
  .replaceAll('SELECTION-REWIRING-RESULTS.md', 'selection-rewiring-results.md')
  .replaceAll('SELECTION-GRAPH-EXPORTS.md', 'selection-graph-exports.md')
  .replaceAll('SELECTION-PATHWAYS.md', 'selection-pathways.md')
  .replaceAll('../reports/circuit-selection/', 'circuit-selection/')
  .replaceAll('../reports/selection-controls/', 'selection-controls/')
  .replaceAll('../flm/circuit_selection.py', 'circuit-selection/circuit_selection.py')
  .replaceAll('../flm/selection_controls.py', 'selection-controls/selection_controls.py'));
await copyFile('reports/selection-graphs/manifest.json', 'public/research/selection-graph-manifest.json');
await copyFile('reports/selection-graphs/export-release.json', 'public/research/selection-graph-release.json');
await copyFile('reports/selection-graphs/rewiring-started.json', 'public/research/selection-rewiring-started.json');
for (const name of ['selection_graphs.py', 'selection_rewiring.py'])
  await copyFile(`flm/${name}`, `public/research/${name}`);
await writeFile('public/research/selection-graph-exports.md', (await readFile('docs/SELECTION-GRAPH-EXPORTS.md', 'utf8'))
  .replaceAll('SELECTION-REWIRING-RESULTS.md', 'selection-rewiring-results.md')
  .replaceAll('SELECTION-TIMING-PILOT.md', 'selection-timing-pilot.md')
  .replaceAll('SELECTION-REWIRING-AUDIT.md', 'selection-rewiring-audit.md')
  .replaceAll('CIRCUIT-SELECTION.md', 'circuit-selection.md')
  .replaceAll('../reports/selection-graphs/manifest.json', 'selection-graph-manifest.json')
  .replaceAll('../reports/selection-graphs/export-release.json', 'selection-graph-release.json')
  .replaceAll('../reports/selection-graphs/rewiring-started.json', 'selection-rewiring-started.json')
  .replaceAll('../flm/selection_rewiring.py', 'selection_rewiring.py')
  .replaceAll('../public/research/', ''));
await mkdir('public/research/selection-rewiring', { recursive: true });
for (const name of ['preflight.json', 'preparation-checks.json', 'manifest.json', 'release.json', 'standalone-audit.json', 'summary.json', 'figures.json'])
  await copyFile(`reports/selection-rewiring/${name}`, `public/research/selection-rewiring/${name}`);
for (const name of ['audit_selection_rewiring.py', 'package_selection_rewiring.py', 'preflight_selection_rewiring.py', 'selection_rewiring_report.py'])
  await copyFile(`scripts/${name}`, `public/research/selection-rewiring/${name}`);
await writeFile('public/research/selection-rewiring-audit.md', (await readFile('docs/SELECTION-REWIRING-AUDIT.md', 'utf8'))
  .replaceAll('SELECTION-REWIRING-RESULTS.md', 'selection-rewiring-results.md')
  .replaceAll('SELECTION-GRAPH-EXPORTS.md', 'selection-graph-exports.md')
  .replaceAll('../reports/selection-rewiring/', 'selection-rewiring/')
  .replaceAll('../scripts/', 'selection-rewiring/'));
await writeFile('public/research/selection-rewiring-results.md', (await readFile('docs/SELECTION-REWIRING-RESULTS.md', 'utf8'))
  .replaceAll('SELECTION-REWIRING-AUDIT.md', 'selection-rewiring-audit.md')
  .replaceAll('CIRCUIT-SELECTION.md', 'circuit-selection.md')
  .replaceAll('SELECTION-TIMING-PILOT.md', 'selection-timing-pilot.md')
  .replaceAll('../reports/selection-rewiring/', 'selection-rewiring/')
  .replaceAll('../scripts/', 'selection-rewiring/')
  .replaceAll('../public/research/', ''));
await mkdir('public/research/selection-pilot', { recursive: true });
await copyFile('reports/selection-pilot/preflight.json', 'public/research/selection-pilot/preflight.json');
await copyFile('reports/selection-pilot/shared-update-preflight.json', 'public/research/selection-pilot/shared-update-preflight.json');
await copyFile('flm/selection_pilot.py', 'public/research/selection-pilot/selection_pilot.py');
await copyFile('scripts/selection_pilot_preflight.py', 'public/research/selection-pilot/selection_pilot_preflight.py');
await writeFile('public/research/selection-timing-pilot.md', (await readFile('docs/SELECTION-TIMING-PILOT.md', 'utf8'))
  .replaceAll('CIRCUIT-SELECTION.md', 'circuit-selection.md')
  .replaceAll('SELECTION-LANGUAGE-TRAINING.md', 'selection-language-training.md')
  .replaceAll('../flm/selection_pilot.py', 'selection-pilot/selection_pilot.py')
  .replaceAll('../scripts/selection_pilot_preflight.py', 'selection-pilot/selection_pilot_preflight.py')
  .replaceAll('../reports/selection-pilot/', 'selection-pilot/'));
await mkdir('public/research/selection-language', { recursive: true });
await copyFile('flm/selection_language.py', 'public/research/selection-language/selection_language.py');
await copyFile('scripts/selection_language_preflight.py', 'public/research/selection-language/selection_language_preflight.py');
await copyFile('reports/selection-language/preparation.json', 'public/research/selection-language/preparation.json');
await copyFile('flm/selection_language_study.py', 'public/research/selection-language/selection_language_study.py');
await copyFile('flm/selection_language_test.py', 'public/research/selection-language/selection_language_test.py');
await copyFile('scripts/selection_language_study_preflight.py', 'public/research/selection-language/selection_language_study_preflight.py');
await copyFile('reports/selection-language/coordinator-preparation.json', 'public/research/selection-language/coordinator-preparation.json');
await copyFile('reports/selection-language/evaluation-verified-preparation.json', 'public/research/selection-language/evaluation-verified-preparation.json');
await writeFile('public/research/selection-language-test.md', (await readFile('docs/SELECTION-LANGUAGE-TEST.md', 'utf8'))
  .replaceAll('../flm/selection_language_test.py', 'selection-language/selection_language_test.py')
  .replaceAll('../reports/selection-language/', 'selection-language/')
  .replaceAll('SELECTION-LANGUAGE-COORDINATOR.md', 'selection-language-coordinator.md'));
await writeFile('public/research/selection-language-coordinator.md', (await readFile('docs/SELECTION-LANGUAGE-COORDINATOR.md', 'utf8'))
  .replaceAll('SELECTION-LANGUAGE-TEST.md', 'selection-language-test.md')
  .replaceAll('../flm/selection_language_study.py', 'selection-language/selection_language_study.py')
  .replaceAll('../reports/selection-language/', 'selection-language/')
  .replaceAll('SELECTION-LANGUAGE-TRAINING.md', 'selection-language-training.md')
  .replaceAll('SELECTION-TIMING-PILOT.md', 'selection-timing-pilot.md'));
await writeFile('public/research/selection-language-training.md', (await readFile('docs/SELECTION-LANGUAGE-TRAINING.md', 'utf8'))
  .replaceAll('SELECTION-LANGUAGE-TEST.md', 'selection-language-test.md')
  .replaceAll('SELECTION-LANGUAGE-COORDINATOR.md', 'selection-language-coordinator.md')
  .replaceAll('../flm/selection_language.py', 'selection-language/selection_language.py')
  .replaceAll('../reports/selection-language/', 'selection-language/')
  .replaceAll('LANGUAGE-LEARNING-RUNNER.md', 'language-learning-runner.md')
  .replaceAll('SELECTION-TIMING-PILOT.md', 'selection-timing-pilot.md'));
await mkdir('public/research/language-learning', { recursive: true });
for (const name of ['language_learning_test.py', 'language_learning_study.py', 'language_learning_validation.py', 'language_learning_pilot.py', 'language_learning_inputs.py', 'language_learning_train.py', 'language_eligibility.py', 'embedding_eligibility.py'])
  await copyFile(`flm/${name}`, `public/research/language-learning/${name}`);
await copyFile('scripts/language_learning_input_preflight.py', 'public/research/language-learning/language_learning_input_preflight.py');
await copyFile('scripts/language_learning_pilot_preflight.py', 'public/research/language-learning/language_learning_pilot_preflight.py');
await copyFile('scripts/language_learning_validation_preflight.py', 'public/research/language-learning/language_learning_validation_preflight.py');
await copyFile('scripts/language_learning_study_preflight.py', 'public/research/language-learning/language_learning_study_preflight.py');
for (const name of ['evaluation-preparation.json', 'study-preparation.json', 'validation-preflight.json', 'pilot-preflight.json', 'input-preflight.json', 'trainer-preflight.json', 'window-preflight.json'])
  await copyFile(`reports/language-eligibility/${name}`, `public/research/language-learning/${name}`);
const learningNotes = ['LANGUAGE-LEARNING-TEST', 'LANGUAGE-LEARNING-STUDY', 'LANGUAGE-LEARNING-VALIDATION', 'LANGUAGE-LEARNING-TIMING', 'LANGUAGE-LEARNING-INPUTS', 'LANGUAGE-LEARNING-RUNNER', 'LANGUAGE-ELIGIBILITY-KERNEL', 'LANGUAGE-ELIGIBILITY-PREPARATION'];
for (const name of learningNotes) {
  let note = await readFile(`docs/${name}.md`, 'utf8');
  for (const linked of learningNotes) note = note.replaceAll(`${linked}.md`, `${linked.toLowerCase()}.md`);
  note = note.replaceAll('LOCAL-LEARNING-PROTOCOL.md', 'local-learning-protocol.md')
    .replaceAll('../flm/', 'language-learning/').replaceAll('../scripts/', 'language-learning/')
    .replaceAll('../reports/language-eligibility/', 'language-learning/');
  await writeFile(`public/research/${name.toLowerCase()}.md`, note);
}
for (const name of ['summary.json', 'directed-groups.csv', 'external-partners.csv', 'kenyon-coverage.csv', 'figures.json'])
  await copyFile(`reports/selection-pathways/${name}`, `public/research/selection-pathways/${name}`);
await copyFile('flm/selection_pathways.py', 'public/research/selection-pathways/selection_pathways.py');
await writeFile('public/research/selection-pathways.md', (await readFile('docs/SELECTION-PATHWAYS.md', 'utf8'))
  .replaceAll('SELECTION-STUDY.md', 'selection-study.md')
  .replaceAll('PUBLISHER-ANNOTATIONS.md', 'publisher-annotations.md')
  .replaceAll('../reports/selection-pathways/', 'selection-pathways/')
  .replaceAll('../flm/selection_pathways.py', 'selection-pathways/selection_pathways.py')
  .replaceAll('../public/research/', ''));
await mkdir('public/research/publisher-annotations', { recursive: true });
for (const name of ['summary.json', 'candidate-annotations.csv', 'missing-transmitter-rows.csv'])
  await copyFile(`reports/publisher-annotations/${name}`, `public/research/publisher-annotations/${name}`);
await copyFile('data/cards/malecns-annotations.json', 'public/research/publisher-annotations/source-card.json');
await copyFile('flm/publisher_annotations.py', 'public/research/publisher-annotations/publisher_annotations.py');
await writeFile('public/research/publisher-annotations.md', (await readFile('docs/PUBLISHER-ANNOTATIONS.md', 'utf8'))
  .replaceAll('SELECTION-STUDY.md', 'selection-study.md')
  .replaceAll('../data/cards/malecns-annotations.json', 'publisher-annotations/source-card.json')
  .replaceAll('../reports/publisher-annotations/', 'publisher-annotations/'));
await copyFile('scripts/choice_replay.py', 'public/research/choice_replay.py');
let foodNote = await readFile('docs/FOOD-RESPONSE-PLAN.md', 'utf8');
for (const [source, target] of [
  ['../scripts/choice_replay.py', 'choice_replay.py'],
  ['../public/research/', ''],
  ['SUBSET-AUDIT.md', 'subset-audit.md'],
  ['FOOD-SENSOR-INTERFACE.md', 'food-sensor-interface.md'],
  ['FOOD-APPROACH-RESULTS.md', 'food-approach-results.md'],
]) foodNote = foodNote.replaceAll(source, target);
await writeFile('public/research/food-response-plan.md', foodNote);
await mkdir('public/research/food-sensors', { recursive: true });
for (const name of ['physical-probe.json', 'geometry-audit.json', 'visible-odor-probe.json', 'visible-odor-audit.json', 'figure.json'])
  await copyFile(`reports/food-sensors/${name}`, `public/research/food-sensors/${name}`);
for (const source of ['flm/food_sensors.py', 'experiments/embodiment/food_sensor_probe.py', 'scripts/audit_food_sensor_probe.py', 'scripts/food_sensor_figure.py'])
  await copyFile(source, `public/research/food-sensors/${source.split('/').at(-1)}`);
await writeFile('public/research/food-sensor-interface.md', (await readFile('docs/FOOD-SENSOR-INTERFACE.md', 'utf8'))
  .replaceAll('FOOD-RESPONSE-PLAN.md', 'food-response-plan.md')
  .replaceAll('../public/research/', '')
  .replaceAll('../reports/food-sensors/', 'food-sensors/')
  .replaceAll('../flm/food_sensors.py', 'food-sensors/food_sensors.py')
  .replaceAll('../experiments/embodiment/food_sensor_probe.py', 'food-sensors/food_sensor_probe.py')
  .replaceAll('../scripts/audit_food_sensor_probe.py', 'food-sensors/audit_food_sensor_probe.py'));
await mkdir('public/research/food-approach', { recursive: true });
for (const name of ['identity.json', 'summary.json', 'audit.json', 'release.json', 'standalone-archive-verification.json'])
  await copyFile(`reports/food-approach/${name}`, `public/research/food-approach/${name}`);
for (const source of ['flm/food_approach.py', 'scripts/audit_food_approach.py', 'scripts/audit_food_sensor_probe.py', 'scripts/food_approach_report.py'])
  await copyFile(source, `public/research/food-approach/${source.split('/').at(-1)}`);
for (const name of ['FOOD-APPROACH-REFERENCE', 'FOOD-APPROACH-RESULTS'])
  await writeFile(`public/research/${name.toLowerCase()}.md`, (await readFile(`docs/${name}.md`, 'utf8'))
    .replaceAll('FOOD-APPROACH-REFERENCE.md', 'food-approach-reference.md')
    .replaceAll('FOOD-SENSOR-INTERFACE.md', 'food-sensor-interface.md')
    .replaceAll('FOOD-RESPONSE-PLAN.md', 'food-response-plan.md')
    .replaceAll('../public/research/', '')
    .replaceAll('../reports/food-approach/', 'food-approach/')
    .replaceAll('../scripts/audit_food_approach.py', 'food-approach/audit_food_approach.py')
    .replaceAll('../flm/food_approach.py', 'food-approach/food_approach.py'));
await copyFile('data/prompts/babylm-original.json', 'public/research/babylm-prompts.json');
await copyFile('docs/BABYLM-EVALUATION.md', 'public/research/babylm-evaluation.md');
await copyFile('docs/LOCAL-LEARNING-PROTOCOL.md', 'public/research/local-learning-protocol.md');
await copyFile('docs/WIRING-LEARNING-PROTOCOL.md', 'public/research/wiring-learning-protocol.md');
await copyFile('docs/LANGUAGE-TOPOLOGY-PROTOCOL.md', 'public/research/language-topology-protocol.md');
await copyFile('docs/WIRING-RESULTS.md', 'public/research/wiring-results.md');
await copyFile('docs/INFERENCE-BUNDLE.md', 'public/research/inference-guide.md');
await copyFile('docs/CLOSED-LOOP-PROTOCOL.md', 'public/research/closed-loop-protocol.md');
await copyFile('docs/CLOSED-LOOP-REPRODUCTION.md', 'public/research/closed-loop-reproduction.md');
await mkdir('public/research/subset-audit', { recursive: true });
for (const name of ['summary.json', 'nodes.csv', 'cell-types.csv'])
  await copyFile(`reports/subset-audit/${name}`, `public/research/subset-audit/${name}`);
await copyFile('flm/subset_audit.py', 'public/research/subset-audit/subset_audit.py');
await copyFile('flm/graph.py', 'public/research/subset-audit/graph.py');
let subsetNote = await readFile('docs/SUBSET-AUDIT.md', 'utf8');
for (const [source, target] of [
  ['../reports/subset-audit/', 'subset-audit/'],
  ['../flm/subset_audit.py', 'subset-audit/subset_audit.py'],
  ['../flm/graph.py', 'subset-audit/graph.py'],
  ['../data/graphs/central-1024/graph-card.json', '../data/graph-card.json'],
  ['LANGUAGE-TOPOLOGY-PROTOCOL.md', 'language-topology-protocol.md'],
]) subsetNote = subsetNote.replaceAll(source, target);
await writeFile('public/research/subset-audit.md', subsetNote);
for (const name of ['controls', 'train', 'study', 'test'])
  await copyFile(`flm/language_core_${name}.py`, `public/research/language_core_${name}.py`);
await copyFile('docs/LANGUAGE-CORE-PROTOCOL.md', 'public/research/language-core-protocol.md');
await copyFile('docs/LANGUAGE-DYNAMICS-PROTOCOL.md', 'public/research/language-dynamics-protocol.md');
for (const name of ['identity', 'replay'])
  await copyFile(`reports/language-dynamics/${name}.json`, `public/research/language-dynamics-${name}.json`);
for (const name of ['language_dynamics', 'language_dynamics_study'])
  await copyFile(`flm/${name}.py`, `public/research/${name}.py`);
let dynamicsNote = await readFile('docs/LANGUAGE-DYNAMICS-FINDINGS.md', 'utf8');
for (const [source, target] of [
  ['../public/research/', ''],
  ['LANGUAGE-DYNAMICS-PROTOCOL.md', 'language-dynamics-protocol.md'],
  ['LANGUAGE-CORE-PROTOCOL.md', 'language-core-protocol.md'],
  ['../reports/language-dynamics/identity.json', 'language-dynamics-identity.json'],
  ['../reports/language-dynamics/replay.json', 'language-dynamics-replay.json'],
  ['../flm/language_dynamics.py', 'language_dynamics.py'],
  ['../flm/language_dynamics_study.py', 'language_dynamics_study.py'],
]) dynamicsNote = dynamicsNote.replaceAll(source, target);
await writeFile('public/research/language-dynamics-findings.md', dynamicsNote);
for (const name of ['identity', 'software-preflight', 'replay-diagnostic', 'selection'])
  await copyFile(`reports/language-core/${name}.json`, `public/research/language-core-${name}.json`);
await writeFile('public/research/language-core-findings.md', (await readFile('docs/LANGUAGE-CORE-RESULTS.md', 'utf8'))
  .replaceAll('../public/research/', '')
  .replaceAll('../reports/language-core/selection.json', 'language-core-selection.json')
  .replaceAll('LANGUAGE-CORE-PROTOCOL.md', 'language-core-protocol.md'));
await writeFile('public/research/language-core-controls.md',
  (await readFile('docs/LANGUAGE-CORE-CONTROLS.md', 'utf8'))
    .replaceAll('LANGUAGE-TOPOLOGY-PROTOCOL.md', 'language-topology-protocol.md')
    .replaceAll('LANGUAGE-CORE-PROTOCOL.md', 'language-core-protocol.md')
    .replaceAll('LANGUAGE-CORE-RESULTS.md', 'language-core-findings.md')
    .replaceAll('../reports/language-core/identity.json', 'language-core-identity.json')
    .replaceAll('../reports/language-core/software-preflight.json', 'language-core-software-preflight.json')
    .replaceAll('../reports/language-core/replay-diagnostic.json', 'language-core-replay-diagnostic.json')
    .replaceAll('../flm/language_core_', 'language_core_'));
await copyFile('data/cards/scan.json', 'public/research/scan-data-card.json');
await copyFile('scripts/scan_data_report.py', 'public/research/scan_data_report.py');
await copyFile('flm/scan_runtime.py', 'public/research/scan_runtime.py');
await copyFile('reports/scan-runtime/preflight.json', 'public/research/scan-runtime-preflight.json');
await copyFile('flm/scan_train.py', 'public/research/scan_train.py');
await copyFile('reports/scan-runtime/training-preflight.json', 'public/research/scan-training-preflight.json');
await copyFile('flm/scan_inputs.py', 'public/research/scan_inputs.py');
await copyFile('scripts/scan_source_audit.py', 'public/research/scan_source_audit.py');
await copyFile('reports/scan-runtime/input-preflight.json', 'public/research/scan-input-preflight.json');
await copyFile('reports/scan-runtime/source-preflight.json', 'public/research/scan-source-preflight.json');
let instructionNote = await readFile('docs/INSTRUCTION-TRANSFER.md', 'utf8');
for (const [source, target] of [
  ['../data/cards/scan.json', 'scan-data-card.json'],
  ['../public/research/', ''],
  ['../scripts/scan_data_report.py', 'scan_data_report.py'],
  ['../flm/scan_runtime.py', 'scan_runtime.py'],
  ['../reports/scan-runtime/preflight.json', 'scan-runtime-preflight.json'],
  ['../flm/scan_train.py', 'scan_train.py'],
  ['../reports/scan-runtime/training-preflight.json', 'scan-training-preflight.json'],
  ['../flm/scan_inputs.py', 'scan_inputs.py'],
  ['../scripts/scan_source_audit.py', 'scan_source_audit.py'],
  ['../reports/scan-runtime/input-preflight.json', 'scan-input-preflight.json'],
  ['../reports/scan-runtime/source-preflight.json', 'scan-source-preflight.json'],
]) instructionNote = instructionNote.replaceAll(source, target);
await writeFile('public/research/instruction-transfer.md', instructionNote);
for (const [source, target] of [['data/cards/ami.json', 'public/data/ami.json'], ['data/cards/wikitext2.json', 'public/data/wikitext2.json'], ['data/graphs/central-1024/graph-card.json', 'public/data/graph-card.json']])
  await copyFile(source, target);
const files = (await readdir('licenses')).filter(name => /\.(txt|md)$/.test(name)).sort();
for (const name of files) await copyFile(`licenses/${name}`, `public/licenses/${name}`);
await copyFile('LICENSE', 'public/licenses/FLM-MIT.txt');
files.unshift('FLM-MIT.txt');
const escape = text => text.replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
await writeFile('public/licenses/index.html', `<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>FLM — Attribution and licenses</title><style>body{max-width:760px;margin:40px auto;padding:0 24px;background:#faf8f5;color:#292820;font:17px/1.7 Georgia,serif}a{color:#a74c20}h1{font-size:30px}li{margin:12px 0}</style><a href="/">← ChatFLM</a><h1>Attribution and licenses</h1><p>FLM's original code is MIT licensed. MaleCNS brain data and AMI meeting transcripts use CC BY 4.0. The NeuroMechFly body and imported kinematics retain their upstream notices. These sources describe different specimens and components.</p><ul>${files.map(name => `<li><a href="${encodeURIComponent(name)}">${escape(name)}</a></li>`).join('')}</ul><p>The complete model and data provenance is available in the <a href="/#research">research notebook</a>.</p></html>\n`);
console.log('Prepared dataset cards and component attribution.');
