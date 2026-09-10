"""Package final topology score records only after the complete checkpoint gate."""
from datetime import datetime, timezone
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_language_topology_report import audit
from scripts.language_topology_report import verified_report, contrast_rows, draw


def encoded(value):
    return (json.dumps(value, indent=2)+'\n').encode('utf8')


def package(root):
    # This verifies all ten selected checkpoints and every source/input identity
    # before reading test tokens or creating any downloadable result artifact.
    report = verified_report(root)
    arithmetic = audit(report)
    published_path = root/'public/research/language-topology-results.json'
    if json.loads(published_path.read_text(encoding='utf8')) != report:
        raise ValueError('Published scores differ from the verified complete report')
    folder = root/'reports/language-topology'
    identity = json.loads((folder/'identity.json').read_text(encoding='utf8'))
    selection = json.loads((folder/'selection.json').read_text(encoding='utf8'))
    sources = set(identity['sources']) | set(selection['scoring_sources'])
    names = sources | {'docs/LANGUAGE-TOPOLOGY-PROTOCOL.md', 'docs/WIKITEXT-PROTOCOL.md',
        'data/cards/wikitext2.json', 'LICENSE', 'scripts/audit_language_topology_report.py',
        'scripts/language_topology_report.py', 'scripts/package_language_topology.py',
        'reports/language-topology/identity.json', 'reports/language-topology/selection.json',
        'reports/language-topology/summary.json', 'public/research/language-topology-results.json'}
    names |= {f'reports/language-topology/test-{run["label"]}.json' for run in report['runs']}
    contents = {name: (root/name).read_bytes() for name in sorted(names)}

    rows = contrast_rows(report)
    figure = root/'public/research/figures/language-topology-test'
    draw(rows, figure)
    csv_file = io.StringIO(newline='')
    writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader(); writer.writerows(rows)
    figure.with_suffix('.csv').write_bytes(csv_file.getvalue().encode('utf8'))
    for suffix in ('.csv', '.png', '.svg'):
        path = figure.with_suffix(suffix)
        contents[path.relative_to(root).as_posix()] = path.read_bytes()
    contents['audit-report.json'] = encoded(arithmetic)
    contents['requirements-audit.txt'] = b'numpy==2.2.6\n'
    contents['README.md'] = b'''# FLM language topology: complete score records

These records contain all ten selected models' article scores, the six paired
wiring contrasts, two retrained slow-state contrasts, protocols, frozen identity
records, figure data, and relevant numerical source. All eight new controls
finished before their ten checkpoint selections were frozen and test scored.

To recompute the published arithmetic from this extracted folder:

```sh
python -m pip install -r requirements-audit.txt
python scripts/audit_language_topology_report.py public/research/language-topology-results.json
```

Only Python and NumPy are needed for that command. The checker recomputes exact
byte-weighted losses, per-graph and per-training-seed means, and all eight paired
article-bootstrap intervals (10,000 draws, seed 31415). Negative measured-minus-
control differences favor the measured fast/slow model. Six topology contrasts
share two measured references. Article intervals do not include graph, training
initialization or dataset uncertainty; this is an exploratory extension after
the original measured test results were seen.

This is an arithmetic audit of supplied scores. It is not independent retraining,
test inference, or external authentication of the corpus/checkpoints. The archive
does not contain training data, optimizer state, or model weights. Preserved
training/scoring source is included for inspection, not as a complete training
environment. The corpus card and protocols document what was scored. File hashes
in manifest.json detect changes relative to this supplied record.

The exact four graph inputs are separately downloadable from
https://flm.kuber.studio/research/language-topology-graphs.zip.
The original six-model FLM/GRU/transformer inference release is at
https://flm.kuber.studio/research/wikitext2-inference.zip; that release excludes
these new topology and slow-state checkpoints. Neither the body demonstrations
nor the earlier sensory studies answer the language-topology question.
'''
    manifest = dict(format='flm-language-topology-records-v1',
        study_identity_sha256=report['study_identity_sha256'], selection_sha256=report['selection_sha256'],
        scope=arithmetic['scope'], files={name: dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
                                       for name, data in sorted(contents.items())})
    contents['manifest.json'] = encoded(manifest)
    destination = root/'public/research/language-topology-records.zip'
    temporary = destination.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(contents.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 9, 10, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED; info.external_attr = 0o644 << 16
            archive.writestr(info, data)
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None or set(archive.namelist()) != set(contents):
            raise ValueError('Final score archive failed validation')
        for name, data in contents.items():
            if archive.read(name) != data:
                raise ValueError('Final score archive changed: ' + name)
    temporary.replace(destination)
    release = dict(verified_utc=datetime.now(timezone.utc).isoformat(),
        archive=destination.relative_to(root).as_posix(), bytes=destination.stat().st_size,
        archive_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(), payloads=len(contents),
        full_checkpoint_and_score_gate_passed=True, arithmetic_audit=arithmetic,
        manifest_sha256=hashlib.sha256(contents['manifest.json']).hexdigest())
    (folder/'records-release.json').write_bytes(encoded(release))
    (root/'public/research/language-topology-release.json').write_bytes(encoded(release))
    return release


if __name__ == '__main__':
    print(json.dumps(package(ROOT), indent=2))
