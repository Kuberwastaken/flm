import { mkdir, copyFile, readdir, readFile, writeFile } from 'node:fs/promises';
await mkdir('public/licenses', { recursive: true });
await mkdir('public/data', { recursive: true });
await mkdir('public/research', { recursive: true });
await copyFile('scripts/choice_replay.py', 'public/research/choice_replay.py');
let foodNote = await readFile('docs/FOOD-RESPONSE-PLAN.md', 'utf8');
for (const [source, target] of [
  ['../scripts/choice_replay.py', 'choice_replay.py'],
  ['../public/research/', ''],
  ['SUBSET-AUDIT.md', 'subset-audit.md'],
]) foodNote = foodNote.replaceAll(source, target);
await writeFile('public/research/food-response-plan.md', foodNote);
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
for (const name of ['identity', 'software-preflight', 'replay-diagnostic'])
  await copyFile(`reports/language-core/${name}.json`, `public/research/language-core-${name}.json`);
await writeFile('public/research/language-core-controls.md',
  (await readFile('docs/LANGUAGE-CORE-CONTROLS.md', 'utf8'))
    .replaceAll('LANGUAGE-TOPOLOGY-PROTOCOL.md', 'language-topology-protocol.md')
    .replaceAll('LANGUAGE-CORE-PROTOCOL.md', 'language-core-protocol.md')
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
