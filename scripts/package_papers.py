"""Package reviewed paper sources and published figure data with file hashes."""
from pathlib import Path
import hashlib
import json
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    files = sorted(path for path in (ROOT / 'papers').rglob('*') if path.is_file())
    files += sorted((ROOT / 'public/research/figures').glob('*.csv'))
    files += [ROOT / 'scripts' / name for name in (
        'research_figures.py', 'continuing_figures.py', 'behavior_figures.py',
        'physical_choice_report.py', 'choice_paper_data.py', 'build_papers.py',
        'package_papers.py', 'wiring_figures.py', 'wiring_paper_data.py',
        'closed_loop_report.py', 'feedback_paper_data.py', 'audit_feedback_release.py', 'scan_data_report.py',
        'audit_language_topology_report.py', 'language_topology_report.py', 'package_language_topology.py',
        'package_topology_inference.py', 'verify_topology_inference_release.py')]
    files += [ROOT / name for name in ('README.md', 'LICENSE', 'pyproject.toml', 'package.json',
        'docs/figures/readme_figures.py', 'public/research/test-results.json',
        'public/brand/provenance.json', 'docs/RESEARCH-PROGRAM.md', 'docs/INFERENCE-BUNDLE.md',
        'docs/LOCAL-LEARNING-PROTOCOL.md', 'reports/local-learning/summary.json',
        'public/research/learned-choice.json', 'docs/WIRING-RESULTS.md',
        'docs/WIRING-LEARNING-PROTOCOL.md', 'docs/LANGUAGE-TOPOLOGY-PROTOCOL.md',
        'reports/language-topology/identity.json', 'reports/language-topology/selection.json',
        'reports/language-topology/summary.json', 'reports/language-topology/paper-inputs.json',
        'reports/language-topology/paper-review.json',
        'public/research/language-topology-results.json', 'public/research/language-topology-release.json',
        'public/research/language-topology-inference-release.json', 'public/research/language-topology-samples.json',
        'reports/language-topology/standalone-records-verification.json',
        'reports/wiring-learning/summary.json', 'reports/wiring-learning/paper-inputs.json',
        'docs/CLOSED-LOOP-PROTOCOL.md', 'docs/CLOSED-LOOP-REPRODUCTION.md',
        'public/research/closed-loop.json', 'public/research/closed-loop.csv',
        'reports/embodiment/closed-loop/paper-inputs.json',
        'docs/INSTRUCTION-TRANSFER.md', 'data/cards/scan.json',
        'reports/scan/data-preparation.json', 'public/research/scan-data.json',
        'public/research/figures/scan-data.png', 'public/research/figures/scan-data.svg',
        'flm/__init__.py', 'flm/scan.py', 'flm/scan_task.py', 'flm/provenance.py',
        'flm/tokenizer.py', 'tests/test_scan.py',
        'data/tokenizers/wikitext2-4096/tokenizer.json',
        'docs/SUBSET-AUDIT.md', 'docs/LANGUAGE-CORE-CONTROLS.md',
        'reports/subset-audit/summary.json', 'reports/subset-audit/nodes.csv',
        'reports/subset-audit/cell-types.csv', 'flm/subset_audit.py',
        'flm/graph.py', 'tests/test_subset_audit.py',
        'flm/language_core_controls.py', 'tests/test_language_core_controls.py',
        'flm/model.py', 'flm/baselines.py', 'flm/train.py', 'flm/language_train.py',
        'flm/corpus_cache.py', 'tests/test_model.py',
        'flm/inference.py', 'flm/topology_inference.py', 'flm/language_report.py',
        'flm/language_topology.py', 'flm/language_topology_test.py', 'flm/language_test.py',
        'flm/language_bridge.py', 'flm/wiring_controls.py', 'flm/ngram.py', 'flm/study_index.py',
        'tests/test_topology_inference.py', 'tests/test_topology_inference_release.py',
        'tests/test_language_topology_audit.py', 'tests/test_language_topology_report.py',
        'tests/test_language_topology_evaluation.py', 'tests/language-topology.test.js',
        'web/language-topology.js', 'web/language-structure.js',
        'docs/WIKITEXT-PROTOCOL.md', 'docs/LANGUAGE-TOPOLOGY-OPERATIONS.md', 'data/cards/wikitext2.json',
        'data/tokenizers/wikitext2-4096/tokenizer-card.json',
        'data/graphs/central-1024/graph-card.json')]
    files += [ROOT / name for name in (
        'docs/LANGUAGE-CORE-PROTOCOL.md', 'docs/LANGUAGE-CORE-RESULTS.md',
        'reports/language-core/identity.json', 'reports/language-core/selection.json',
        'reports/language-core/summary.json', 'reports/language-core/figures.json',
        'reports/language-core/paper-review.json',
        'public/research/language-core-results.json', 'public/research/language-core-release.json',
        'public/research/language-core-inference-release.json', 'public/research/language-core-samples.json',
        'flm/language_core_train.py', 'flm/language_core_study.py', 'flm/language_core_test.py',
        'flm/core_inference.py', 'scripts/language_core_report.py',
        'scripts/audit_language_core_report.py', 'scripts/package_language_core.py',
        'scripts/package_core_inference.py', 'scripts/verify_core_inference_release.py',
        'scripts/language_core_figures.py', 'web/language-core.js',
        'tests/test_language_core_training.py', 'tests/test_language_core_report.py',
        'tests/test_language_core_records.py', 'tests/test_language_core_figures.py',
        'tests/test_language_core_audit.py', 'tests/test_core_inference_release.py',
        'tests/test_core_inference.py', 'tests/language-core.test.js',
        'tests/language-core-loading.test.js', 'tests/helpers/core-results-fixture.js',
        'docs/FOOD-RESPONSE-PLAN.md', 'scripts/choice_replay.py',
        'docs/SCAN-CONDITION-PREPARATION.md', 'docs/SCAN-TIMING-PILOT.md',
        'flm/scan_inputs.py', 'flm/scan_runtime.py', 'flm/scan_train.py',
        'flm/scan_conditions.py', 'flm/scan_pilot.py', 'scripts/scan_source_audit.py',
        'reports/scan-runtime/input-preflight.json', 'reports/scan-runtime/source-preflight.json',
        'reports/scan-runtime/condition-preflight.json', 'reports/scan-runtime/pilot-preflight.json',
        'docs/LANGUAGE-ELIGIBILITY-PREPARATION.md', 'flm/embedding_eligibility.py',
        'flm/local_learning.py', 'tests/test_embedding_eligibility.py',
        'docs/LANGUAGE-ELIGIBILITY-KERNEL.md', 'flm/language_eligibility.py',
        'tests/test_language_eligibility.py', 'reports/language-eligibility/window-preflight.json',
        'docs/LANGUAGE-LEARNING-RUNNER.md', 'flm/language_learning_train.py',
        'tests/test_language_learning_train.py', 'reports/language-eligibility/trainer-preflight.json',
        'docs/SELECTION-STUDY.md', 'flm/selection_feasibility.py',
        'tests/test_selection_feasibility.py', 'scripts/selection_feasibility_figures.py',
        'reports/selection-feasibility/summary.json', 'reports/selection-feasibility/candidate-body-ids.json',
        'reports/selection-feasibility/figures.json',
        'docs/PUBLISHER-ANNOTATIONS.md', 'data/cards/malecns-annotations.json',
        'flm/publisher_annotations.py', 'tests/test_publisher_annotations.py',
        'reports/publisher-annotations/summary.json',
        'reports/publisher-annotations/candidate-annotations.csv',
        'reports/publisher-annotations/missing-transmitter-rows.csv',
        'docs/SELECTION-PATHWAYS.md', 'flm/selection_pathways.py',
        'tests/test_selection_pathways.py', 'scripts/selection_pathway_figures.py',
        'reports/selection-pathways/summary.json', 'reports/selection-pathways/directed-groups.csv',
        'reports/selection-pathways/external-partners.csv', 'reports/selection-pathways/kenyon-coverage.csv',
        'reports/selection-pathways/figures.json',
        'docs/CIRCUIT-SELECTION.md', 'flm/circuit_selection.py', 'flm/selection_controls.py',
        'tests/test_circuit_selection.py', 'tests/test_selection_controls.py',
        'reports/circuit-selection/summary.json', 'reports/circuit-selection/candidate-body-ids.json',
        'reports/circuit-selection/members.csv', 'reports/selection-controls/summary.json',
        'reports/selection-controls/control-body-ids.json',
        'docs/SELECTION-GRAPH-EXPORTS.md', 'flm/selection_graphs.py', 'flm/selection_rewiring.py',
        'tests/test_selection_graphs.py', 'tests/test_selection_rewiring.py',
        'reports/selection-graphs/manifest.json', 'reports/selection-graphs/export-release.json',
        'reports/selection-graphs/rewiring-started.json',
        'docs/SELECTION-REWIRING-AUDIT.md', 'scripts/audit_selection_rewiring.py',
        'scripts/package_selection_rewiring.py', 'scripts/preflight_selection_rewiring.py',
        'tests/test_selection_rewiring_archive.py',
        'reports/selection-rewiring/preflight.json', 'reports/selection-rewiring/preparation-checks.json',
        'docs/SELECTION-TIMING-PILOT.md', 'flm/selection_pilot.py', 'scripts/selection_pilot_preflight.py',
        'tests/test_selection_pilot.py', 'reports/selection-pilot/preflight.json',
        'docs/LANGUAGE-LEARNING-INPUTS.md', 'flm/language_learning_inputs.py',
        'docs/LANGUAGE-LEARNING-TIMING.md', 'flm/language_learning_pilot.py',
        'docs/LANGUAGE-LEARNING-VALIDATION.md', 'flm/language_learning_validation.py',
        'scripts/language_learning_validation_preflight.py', 'tests/test_language_learning_validation.py',
        'reports/language-eligibility/validation-preflight.json',
        'reports/babylm/completion-flm-10m-s43.json',
        'scripts/language_learning_pilot_preflight.py', 'tests/test_language_learning_pilot.py',
        'reports/language-eligibility/pilot-preflight.json',
        'scripts/language_learning_input_preflight.py', 'tests/test_language_learning_inputs.py',
        'reports/language-eligibility/input-preflight.json', 'flm/babylm.py',
        'data/sources/babylm-2026.json', 'data/cards/babylm-2026-acquisition.json',
        'data/tokenizers/babylm-2026-4096/tokenizer-card.json',
        'data/tokenizers/babylm-2026-4096/tokenizer.json',
        'docs/SELECTION-REWIRING-RESULTS.md', 'scripts/selection_rewiring_report.py',
        'tests/test_selection_rewiring_report.py', 'reports/selection-rewiring/manifest.json',
        'reports/selection-rewiring/release.json', 'reports/selection-rewiring/standalone-audit.json',
        'reports/selection-rewiring/summary.json', 'reports/selection-rewiring/figures.json')]
    files += sorted((ROOT / 'reports/language-core').glob('test-*.json'))
    # Preserve the project's overview and its actual embedded figures together.
    # The source archive remains a paper snapshot, not a runnable repository.
    readme = (ROOT / 'README.md').read_text(encoding='utf8')
    for target in re.findall(r'!\[[^\]]*\]\(([^)\s]+)\)', readme):
        if '://' in target:
            raise ValueError('README figures must have local, archived source assets')
        figure = (ROOT / target).resolve()
        if not figure.is_relative_to(ROOT.resolve()) or not figure.is_file():
            raise ValueError('Missing or nonlocal README figure: ' + target)
        files.append(figure)
        if figure.with_suffix('.svg').is_file():
            files.append(figure.with_suffix('.svg'))
    files += sorted(path for path in (ROOT / 'licenses').iterdir() if path.suffix in ('.txt', '.md'))
    contents = {path.relative_to(ROOT).as_posix(): path.read_bytes() for path in files}
    manifest = {name: {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
                for name, data in sorted(contents.items())}
    contents['source-manifest.json'] = (json.dumps(manifest, indent=2) + '\n').encode()
    destination = ROOT / 'public/research/paper-source.zip'
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(contents.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 9, 10, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, data)
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise RuntimeError('Paper source archive failed CRC validation')
        for name, record in manifest.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != record['sha256']:
                raise RuntimeError(f'Paper source archive changed {name}')
    print(f'Packaged and verified {len(manifest)} source files: {destination}')


if __name__ == '__main__':
    main()
