import { mkdir, copyFile, readdir, readFile, writeFile } from 'node:fs/promises';
await mkdir('public/licenses', { recursive: true });
await mkdir('public/data', { recursive: true });
await mkdir('public/research', { recursive: true });
await mkdir('public/research/selection-feasibility', { recursive: true });
for (const name of ['summary.json', 'candidate-body-ids.json', 'figures.json'])
  await copyFile(`reports/selection-feasibility/${name}`, `public/research/selection-feasibility/${name}`);
await copyFile('flm/selection_feasibility.py', 'public/research/selection-feasibility/selection_feasibility.py');
await writeFile('public/research/selection-study.md', (await readFile('docs/SELECTION-STUDY.md', 'utf8'))
  .replaceAll('SELECTION-GRAPH-EXPORTS.md', 'selection-graph-exports.md')
  .replaceAll('SELECTION-REWIRING-RESULTS.md', 'selection-rewiring-results.md')
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
  ['FOOD-CORE-INTERFACE.md', 'food-core-interface.md'],
  ['FOOD-CORE-PHYSICAL-RESULTS.md', 'food-core-physical-results.md'],
  ['FOOD-READOUT-LEARNING.md', 'food-readout-learning.md'],
  ['FOOD-EPISODE-RUNNER.md', 'food-episode-runner.md'],
  ['FOOD-ADAPTATION-SCHEDULE.md', 'food-adaptation-schedule.md'],
  ['PHYSICAL-STATE-SEMANTICS.md', 'physical-state-semantics.md'],
]) foodNote = foodNote.replaceAll(source, target);
await writeFile('public/research/food-response-plan.md', foodNote);
await mkdir('public/research/food-readout', { recursive: true });
for (const source of ['flm/food_readout_learning.py', 'tests/test_food_readout_learning.py'])
  await copyFile(source, `public/research/food-readout/${source.split('/').at(-1)}`);
await writeFile('public/research/food-readout-learning.md', (await readFile('docs/FOOD-READOUT-LEARNING.md', 'utf8'))
  .replaceAll('FOOD-CORE-PHYSICAL-RESULTS.md', 'food-core-physical-results.md')
  .replaceAll('FOOD-RESPONSE-PLAN.md', 'food-response-plan.md')
  .replaceAll('FOOD-EPISODE-RUNNER.md', 'food-episode-runner.md')
  .replaceAll('FOOD-ADAPTATION-SCHEDULE.md', 'food-adaptation-schedule.md')
  .replaceAll('../flm/food_readout_learning.py', 'food-readout/food_readout_learning.py')
  .replaceAll('../tests/test_food_readout_learning.py', 'food-readout/test_food_readout_learning.py'));
await mkdir('public/research/food-episode', { recursive: true });
for (const source of ['flm/food_episode.py', 'tests/test_food_episode.py',
  'experiments/embodiment/food_learning_environment.py', 'experiments/embodiment/verify_food_episode.py',
  'scripts/package_food_episode_smoke.py', 'scripts/audit_food_episode_smoke.py',
  'flm/food_episode_store.py', 'tests/test_food_episode_store.py',
  'experiments/embodiment/verify_food_episode_store.py', 'scripts/package_food_episode_store.py',
  'reports/food-episode/restart-verification.json', 'reports/food-episode/restart-tests.json',
  'reports/food-episode/restart-repack.json',
  'reports/food-episode/preparation.json', 'reports/food-episode/archive-audit.json'])
  await copyFile(source, `public/research/food-episode/${source.split('/').at(-1)}`);
await writeFile('public/research/food-episode-runner.md', (await readFile('docs/FOOD-EPISODE-RUNNER.md', 'utf8'))
  .replaceAll('FOOD-ADAPTATION-SCHEDULE.md', 'food-adaptation-schedule.md')
  .replaceAll('FOOD-READOUT-LEARNING.md', 'food-readout-learning.md')
  .replaceAll('FOOD-RESPONSE-PLAN.md', 'food-response-plan.md')
  .replaceAll('PHYSICAL-STATE-SEMANTICS.md', 'physical-state-semantics.md')
  .replaceAll('../flm/food_episode.py', 'food-episode/food_episode.py')
  .replaceAll('../flm/food_episode_store.py', 'food-episode/food_episode_store.py')
  .replaceAll('../tests/test_food_episode_store.py', 'food-episode/test_food_episode_store.py')
  .replaceAll('../experiments/embodiment/', 'food-episode/')
  .replaceAll('../scripts/audit_food_episode_smoke.py', 'food-episode/audit_food_episode_smoke.py')
  .replaceAll('../scripts/package_food_episode_store.py', 'food-episode/package_food_episode_store.py')
  .replaceAll('../reports/food-episode/', 'food-episode/')
  .replaceAll('../public/research/', ''));
await mkdir('public/research/food-schedule', { recursive: true });
for (const source of ['flm/food_schedule.py', 'tests/test_food_schedule.py', 'reports/food-schedule/preparation.json'])
  await copyFile(source, `public/research/food-schedule/${source.split('/').at(-1)}`);
await writeFile('public/research/food-adaptation-schedule.md', (await readFile('docs/FOOD-ADAPTATION-SCHEDULE.md', 'utf8'))
  .replaceAll('FOOD-EPISODE-RUNNER.md', 'food-episode-runner.md')
  .replaceAll('FOOD-RESPONSE-PLAN.md', 'food-response-plan.md')
  .replaceAll('../flm/food_schedule.py', 'food-schedule/food_schedule.py')
  .replaceAll('../tests/test_food_schedule.py', 'food-schedule/test_food_schedule.py')
  .replaceAll('../reports/food-schedule/preparation.json', 'food-schedule/preparation.json'));
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
await mkdir('public/research/food-core', { recursive: true });
for (const name of ['preparation.json', 'isolated-parity.json', 'release.json'])
  await copyFile(`reports/food-core/${name}`, `public/research/food-core/${name}`);
for (const source of ['flm/food_core.py', 'flm/food_core_export.py', 'experiments/embodiment/food_core_runtime.py', 'experiments/embodiment/verify_food_core.py', 'scripts/prepare_food_core.py', 'scripts/package_food_core.py'])
  await copyFile(source, `public/research/food-core/${source.split('/').at(-1)}`);
await writeFile('public/research/food-core-interface.md', (await readFile('docs/FOOD-CORE-INTERFACE.md', 'utf8'))
  .replaceAll('FOOD-RESPONSE-PLAN.md', 'food-response-plan.md')
  .replaceAll('../reports/food-core/', 'food-core/')
  .replaceAll('../flm/food_core.py', 'food-core/food_core.py')
  .replaceAll('../scripts/prepare_food_core.py', 'food-core/prepare_food_core.py')
  .replaceAll('../experiments/embodiment/verify_food_core.py', 'food-core/verify_food_core.py'));
await mkdir('public/research/food-core-physical', { recursive: true });
for (const name of ['identity.json', 'summary.json', 'audit.json', 'release.json', 'standalone-archive-verification.json', 'body-assets.json', 'body-semantics.json', 'replay.json'])
  await copyFile(`reports/food-core-physical/${name}`, `public/research/food-core-physical/${name}`);
for (const source of ['experiments/embodiment/food_replay_assets.py', 'experiments/embodiment/inspect_food_body_semantics.py', 'experiments/embodiment/export_food_replay.py'])
  await copyFile(source, `public/research/food-core-physical/${source.split('/').at(-1)}`);
for (const name of ['PHYSICAL-STATE-SEMANTICS', 'FOOD-CORE-PHYSICAL-RESULTS', 'FOOD-CORE-PHYSICAL-REFERENCE']) {
  let note = await readFile(`docs/${name}.md`, 'utf8');
  for (const [source, target] of [
    ['PHYSICAL-STATE-SEMANTICS.md', 'physical-state-semantics.md'],
    ['FOOD-CORE-PHYSICAL-REFERENCE.md', 'food-core-physical-reference.md'],
    ['FOOD-CORE-INTERFACE.md', 'food-core-interface.md'],
    ['FOOD-APPROACH-REFERENCE.md', 'food-approach-reference.md'],
    ['FOOD-RESPONSE-PLAN.md', 'food-response-plan.md'],
    ['../reports/food-core-physical/', 'food-core-physical/'],
    ['../public/research/', ''],
    ['../experiments/embodiment/food_replay_assets.py', 'food-core-physical/food_replay_assets.py'],
    ['../experiments/embodiment/inspect_food_body_semantics.py', 'food-core-physical/inspect_food_body_semantics.py'],
    ['../experiments/embodiment/export_food_replay.py', 'food-core-physical/export_food_replay.py'],
  ]) note = note.replaceAll(source, target);
  await writeFile(`public/research/${name.toLowerCase()}.md`, note);
}
await copyFile('data/prompts/babylm-original.json', 'public/research/babylm-prompts.json');
await mkdir('public/research/babylm-results', { recursive: true });
for (const name of ['summary.json', 'samples.json', 'selection.json', 'completed-evidence-verification.json',
  ...(await readdir('reports/babylm')).filter(name => /^(test-|samples-).*\.json$/.test(name))])
  await copyFile(`reports/babylm/${name}`, `public/research/babylm-results/${name}`);
for (const [sourceFolder, targetFolder] of [['tables-v1', 'tables'], ['figures-v1', 'figures']]) {
  await mkdir(`public/research/babylm-results/${targetFolder}`, { recursive: true });
  for (const name of await readdir(`reports/babylm/${sourceFolder}`))
    await copyFile(`reports/babylm/${sourceFolder}/${name}`, `public/research/babylm-results/${targetFolder}/${name}`);
}
for (const stem of ['babylm-pooled', 'babylm-components'])
  for (const extension of ['png', 'svg'])
    await copyFile(`reports/babylm/figures-v1/${stem}.${extension}`, `public/research/babylm-results/${stem}.${extension}`);
await copyFile('flm/train.py', 'public/research/babylm-results/train.py');
for (const name of ['timing.json', 'cost-decision-v1.json'])
  await copyFile(`reports/selection-pilot/${name}`, `public/research/selection-pilot/${name}`);
for (const name of ['request-v1.json', 'study-identity.json'])
  await copyFile(`reports/selection-language/${name}`, `public/research/selection-language/${name}`);
for (const name of ['BABYLM-FINDINGS', 'BABYLM-PROTOCOL', 'SELECTION-LANGUAGE-PROTOCOL', 'SELECTION-LANGUAGE-ERRATA']) {
  let note = await readFile(`docs/${name}.md`, 'utf8');
  note = note.replaceAll('../reports/babylm/tables-v1/', 'babylm-results/tables/')
    .replaceAll('../reports/babylm/figures-v1/', 'babylm-results/figures/')
    .replaceAll('../reports/babylm/', 'babylm-results/')
    .replaceAll('../reports/selection-pilot/', 'selection-pilot/')
    .replaceAll('../reports/selection-language/', 'selection-language/')
    .replaceAll('../reports/circuit-selection/', 'circuit-selection/')
    .replaceAll('../flm/train.py', 'babylm-results/train.py')
    .replace(/\b([A-Z][A-Z-]+)\.md/g, match => match.toLowerCase());
  await writeFile(`public/research/${name.toLowerCase()}.md`, note);
}
await copyFile('docs/BABYLM-EVALUATION.md', 'public/research/babylm-evaluation.md');
await mkdir('public/research/babylm-result-tables', { recursive: true });
for (const [source, target] of [
  ['scripts/babylm_result_tables.py', 'babylm_result_tables.py'],
  ['scripts/babylm_result_figures.py', 'babylm_result_figures.py'],
  ['tests/test_babylm_result_figures.py', 'test_babylm_result_figures.py'],
  ['reports/babylm/result-figures-preparation.json', 'figures-preparation.json'],
  ['tests/test_babylm_result_tables.py', 'test_babylm_result_tables.py'],
  ['reports/babylm/result-tables-preparation.json', 'preparation.json'],
]) await copyFile(source, `public/research/babylm-result-tables/${target}`);
await writeFile('public/research/babylm-result-tables.md', (await readFile('docs/BABYLM-RESULT-TABLES.md', 'utf8'))
  .replaceAll('../scripts/babylm_result_figures.py', 'babylm-result-tables/babylm_result_figures.py')
  .replaceAll('../tests/test_babylm_result_figures.py', 'babylm-result-tables/test_babylm_result_figures.py')
  .replaceAll('../reports/babylm/result-figures-preparation.json', 'babylm-result-tables/figures-preparation.json')
  .replaceAll('../scripts/babylm_result_tables.py', 'babylm-result-tables/babylm_result_tables.py')
  .replaceAll('../tests/test_babylm_result_tables.py', 'babylm-result-tables/test_babylm_result_tables.py')
  .replaceAll('../reports/babylm/result-tables-preparation.json', 'babylm-result-tables/preparation.json')
  .replaceAll('BABYLM-HANDOFF.md', 'babylm-handoff.md')
  .replaceAll('BABYLM-EVALUATION.md', 'babylm-evaluation.md'));
await mkdir('public/research/babylm-handoff', { recursive: true });
for (const [source, target] of [
  ['scripts/continue_babylm_research.py', 'continue_babylm_research.py'],
  ['scripts/audit_babylm_completion.py', 'audit_babylm_completion.py'],
  ['tests/test_babylm_completion_command.py', 'test_babylm_completion_command.py'],
  ['reports/babylm/completion-command-preparation.json', 'completion-preparation.json'],
  ['reports/babylm/completion-command-transformer-100m-s42.json', 'completion-transformer-100m-s42.json'],
  ['reports/babylm/completion-flm-100m-s43.json', 'completion-flm-100m-s43.json'],
  ['reports/babylm/completion-gru-100m-s43.json', 'completion-gru-100m-s43.json'],
  ['reports/babylm/completion-transformer-100m-s43.json', 'completion-transformer-100m-s43.json'],
  ['reports/babylm/selection.json', 'selection.json'],
  ['reports/babylm/selection-verification.json', 'selection-verification.json'],
  ['reports/babylm/evaluation-recovery.json', 'evaluation-recovery.json'],
  ['tests/test_babylm_result_writing.py', 'test_babylm_result_writing.py'],
  ['tests/test_babylm_handoff.py', 'test_babylm_handoff.py'],
  ['requirements-operations.txt', 'requirements-operations.txt'],
  ['reports/babylm/handoff-preparation.json', 'preparation.json'],
]) await copyFile(source, `public/research/babylm-handoff/${target}`);
await writeFile('public/research/babylm-handoff.md', (await readFile('docs/BABYLM-HANDOFF.md', 'utf8'))
  .replaceAll('../scripts/audit_babylm_completion.py', 'babylm-handoff/audit_babylm_completion.py')
  .replaceAll('../tests/test_babylm_completion_command.py', 'babylm-handoff/test_babylm_completion_command.py')
  .replaceAll('../reports/babylm/completion-command-preparation.json', 'babylm-handoff/completion-preparation.json')
  .replaceAll('../reports/babylm/completion-command-transformer-100m-s42.json', 'babylm-handoff/completion-transformer-100m-s42.json')
  .replaceAll('../reports/babylm/completion-flm-100m-s43.json', 'babylm-handoff/completion-flm-100m-s43.json')
  .replaceAll('../reports/babylm/completion-gru-100m-s43.json', 'babylm-handoff/completion-gru-100m-s43.json')
  .replaceAll('../reports/babylm/completion-transformer-100m-s43.json', 'babylm-handoff/completion-transformer-100m-s43.json')
  .replaceAll('../reports/babylm/selection.json', 'babylm-handoff/selection.json')
  .replaceAll('../reports/babylm/selection-verification.json', 'babylm-handoff/selection-verification.json')
  .replaceAll('../reports/babylm/evaluation-recovery.json', 'babylm-handoff/evaluation-recovery.json')
  .replaceAll('../scripts/continue_babylm_research.py', 'babylm-handoff/continue_babylm_research.py')
  .replaceAll('../tests/test_babylm_handoff.py', 'babylm-handoff/test_babylm_handoff.py')
  .replaceAll('../reports/babylm/handoff-preparation.json', 'babylm-handoff/preparation.json')
  .replaceAll('BABYLM-EVALUATION.md', 'babylm-evaluation.md')
  .replaceAll('SELECTION-TIMING-PILOT.md', 'selection-timing-pilot.md')
  .replaceAll('SELECTION-LANGUAGE-COORDINATOR.md', 'selection-language-coordinator.md'));

await copyFile('docs/LOCAL-LEARNING-PROTOCOL.md', 'public/research/local-learning-protocol.md');
await copyFile('docs/WIRING-LEARNING-PROTOCOL.md', 'public/research/wiring-learning-protocol.md');
await copyFile('docs/LANGUAGE-TOPOLOGY-PROTOCOL.md', 'public/research/language-topology-protocol.md');
await copyFile('docs/WIRING-RESULTS.md', 'public/research/wiring-results.md');
await writeFile('public/research/inference-guide.md', (await readFile('docs/INFERENCE-BUNDLE.md','utf8')).replaceAll('CHATFLM.md','chatflm.md'));
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
for (const name of ['scan_study.py', 'scan_evaluate.py', 'scan_test_inputs.py', 'scan_conditions.py', 'scan_pilot.py'])
  await copyFile(`flm/${name}`, `public/research/${name}`);
for (const name of ['test_scan_study.py', 'test_scan_evaluate.py', 'test_scan_test_inputs.py'])
  await copyFile(`tests/${name}`, `public/research/${name}`);
for (const name of ['condition-preflight', 'pilot-preflight', 'study-preparation', 'evaluation-preparation', 'evaluation-inventory-verification'])
  await copyFile(`reports/scan-runtime/${name}.json`, `public/research/scan-${name}.json`);
for (const name of ['SCAN-STUDY-COORDINATOR', 'SCAN-EVALUATION', 'SCAN-CONDITION-PREPARATION', 'SCAN-TIMING-PILOT']) {
  let note = await readFile(`docs/${name}.md`, 'utf8');
  for (const [source, target] of [
    ['INSTRUCTION-TRANSFER.md', 'instruction-transfer.md'],
    ['SCAN-STUDY-COORDINATOR.md', 'scan-study-coordinator.md'],
    ['SCAN-EVALUATION.md', 'scan-evaluation.md'],
    ['SCAN-CONDITION-PREPARATION.md', 'scan-condition-preparation.md'],
    ['SCAN-TIMING-PILOT.md', 'scan-timing-pilot.md'],
    ['../flm/', ''], ['../tests/', ''],
    ['../reports/scan-runtime/', 'scan-'],
  ]) note = note.replaceAll(source, target);
  await writeFile(`public/research/${name.toLowerCase()}.md`, note);
}
let instructionNote = await readFile('docs/INSTRUCTION-TRANSFER.md', 'utf8');
for (const [source, target] of [
  ['SCAN-CONDITION-PREPARATION.md', 'scan-condition-preparation.md'],
  ['SCAN-STUDY-COORDINATOR.md', 'scan-study-coordinator.md'],
  ['SCAN-EVALUATION.md', 'scan-evaluation.md'],
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
await writeFile('public/licenses/BODY-PROVENANCE.md', (await readFile('licenses/BODY-PROVENANCE.md', 'utf8'))
  .replaceAll('../docs/PHYSICAL-STATE-SEMANTICS.md', '../research/physical-state-semantics.md'));
await copyFile('LICENSE', 'public/licenses/FLM-MIT.txt');
files.unshift('FLM-MIT.txt');
const escape = text => text.replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
await writeFile('public/licenses/index.html', `<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>FLM — Attribution and licenses</title><style>body{max-width:760px;margin:40px auto;padding:0 24px;background:#faf8f5;color:#292820;font:17px/1.7 Georgia,serif}a{color:#a74c20}h1{font-size:30px}li{margin:12px 0}</style><a href="/">← ChatFLM</a><h1>Attribution and licenses</h1><p>FLM's original code is MIT licensed. MaleCNS brain data and AMI meeting transcripts use CC BY 4.0. The NeuroMechFly body and imported kinematics retain their upstream notices. These sources describe different specimens and components.</p><ul>${files.map(name => `<li><a href="${encodeURIComponent(name)}">${escape(name)}</a></li>`).join('')}</ul><p>The complete model and data provenance is available in the <a href="/#research">research notebook</a>.</p></html>\n`);
for (const name of ['babylm-result-tables', 'babylm-handoff']) {
  const path = `public/research/${name}.md`;
  await writeFile(path, (await readFile(path, 'utf8'))
    .replaceAll('BABYLM-FINDINGS.md', 'babylm-findings.md')
    .replaceAll('../reports/babylm/tables-v1/', 'babylm-results/tables/')
    .replaceAll('../reports/babylm/completed-evidence-verification.json', 'babylm-results/completed-evidence-verification.json'));
}
console.log('Prepared dataset cards and component attribution.');

await writeFile('public/research/chatflm.md', (await readFile('docs/CHATFLM.md', 'utf8'))
  .replaceAll('LANGUAGE-CORE-RESULTS.md','language-core-findings.md')
  .replaceAll('LANGUAGE-DYNAMICS-FINDINGS.md','language-dynamics-findings.md')
  .replaceAll('FOOD-CORE-PHYSICAL-RESULTS.md','food-core-physical-results.md')
  .replaceAll('BABYLM-FINDINGS.md','babylm-findings.md')
  .replaceAll('SELECTION-LANGUAGE-PROTOCOL.md','selection-language-protocol.md')
  .replaceAll('../public/models/catalog.json','/models/catalog.json')
  .replaceAll('../tests/catalog.test.js','/research/browser-catalog-test.js'));
await copyFile('tests/catalog.test.js','public/research/browser-catalog-test.js');

// Publish the prospective allocation rule separately from the frozen study protocol.
const priorityLinks = {
  '../reports/selection-language/interpretation-v1.json': 'selection-language-interpretation.json',
  'SELECTION-LANGUAGE-PROTOCOL.md': 'selection-language-protocol.md',
  'SELECTION-LANGUAGE-ERRATA.md': 'selection-language-errata.md',
  'LANGUAGE-CORE-RESULTS.md': 'language-core-findings.md',
  'LANGUAGE-DYNAMICS-FINDINGS.md': 'language-dynamics-findings.md',
  'ANATOMICAL-PRIOR-DECISION.md': 'anatomical-prior-decision.md',
  '../public/research/': ''
};
for (const [source, target] of [['ANATOMICAL-PRIOR-DECISION.md','anatomical-prior-decision.md'], ['PREDICTIVE-COMPUTATION.md','predictive-computation.md']]) {
  let text = await readFile(`docs/${source}`, 'utf8');
  for (const [from, to] of Object.entries(priorityLinks)) text = text.replaceAll(from, to);
  await writeFile(`public/research/${target}`, text);
}
await copyFile('reports/selection-language/interpretation-v1.json', 'public/research/selection-language-interpretation.json');
