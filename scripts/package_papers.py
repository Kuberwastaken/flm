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
