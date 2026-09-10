"""Release all computation scores after a full gate and fresh arithmetic audit."""
from datetime import datetime, timezone
import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys
import tempfile
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_language_core_report import audit
from scripts.language_core_report import verified_report


FOLDER = 'reports/language-core'
SUMMARY = FOLDER + '/summary.json'
ARCHIVE = 'public/research/language-core-records.zip'
PUBLIC_RESULT = 'public/research/language-core-results.json'
RELEASE = FOLDER + '/records-release.json'
PUBLIC_RELEASE = 'public/research/language-core-release.json'
ASSETS = ('docs/LANGUAGE-CORE-PROTOCOL.md', 'docs/WIKITEXT-PROTOCOL.md',
          'data/cards/wikitext2.json', 'data/graphs/central-1024/graph-card.json',
          'data/tokenizers/wikitext2-4096/tokenizer-card.json',
          'data/tokenizers/wikitext2-4096/tokenization-card.json',
          'reports/language-core/software-preflight.json', 'reports/language-core/replay-diagnostic.json',
          'reports/wikitext2/test-flm-s42.json', 'reports/wikitext2/test-flm-s43.json',
          'LICENSE', 'licenses/CC-BY-4.0.txt', 'licenses/DATA-ATTRIBUTION.md',
          'scripts/audit_language_core_report.py', 'scripts/language_core_report.py',
          'scripts/package_language_core.py')
README = b'''# FLM language computation: complete score records

FLM by Kuber Mehta. These records contain all eight selected conditions:
full, fixed dynamics, no lateral recurrence and no temporal state, each with
training seeds 42 and 43. The two full-model references reuse the original
WikiText scores; the other six conditions were retrained. The complete-study
gate must pass before this archive is built. No partial score release is made.

To audit the supplied arithmetic from this extracted directory:

```sh
python -m pip install -r requirements-audit.txt
python scripts/audit_language_core_report.py reports/language-core/summary.json
```

Only Python and NumPy are needed for that command. The independent checker
recomputes article and aggregate losses, byte-weighted BPB, shared-tokenizer
perplexity, all six primary and two secondary paired-article intervals, and
their declared descriptive means. The release is also checked from a fresh
archive extraction outside the repository with FLM and PyTorch imports disabled.
Numeric comparisons allow relative 1e-9 / absolute 1e-12 rounding tolerance.

All eight conditions, 480 article-score rows, and eight contrast rows are also
provided as CSVs under tables/. No article text or tokenized corpus is included.
The graph card and protocols describe the measured 1,024-neuron subset and its
limits. This is not an intact fly brain. Freezing recurrent dynamics leaves
the lexical interface trainable through time. No lateral recurrence retains
fast and slow temporal memory; no temporal state resets both on every token.
Allocated, trainable and frozen parameter counts must not be conflated with
effective capacity. See the protocol for disconnected trainable parameters.

Primary effects are full minus each control; negative favors full. The
secondary effect is no-lateral minus no-temporal-state. All eight intervals
use 10,000 paired-article draws with seed 31415. They condition on fitted models;
shared references/articles and two initializations do not establish training
uncertainty. No confidence interval for a descriptive mean is declared. This
exploratory follow-up does not test equivalence or topology-by-trainability.
It cannot reverse the separate topology study's no-anatomical-advantage result.

This is an arithmetic audit of supplied scores, not independent retraining,
held-out inference or authentication of data/checkpoints. Manifest hashes
describe this supplied record. Numerical sources are included for inspection,
not as a complete training environment. No model weights, optimizer states,
training RNG, raw corpus or user conversations are included. Source cards and
protocols preserve data and graph attribution; code uses MIT.

The selected models are released separately in language-core-inference.zip.
The earlier wikitext2-inference.zip includes GRU and transformer baselines;
language-topology-records.zip contains the separate anatomical wiring controls.
These archives are served under https://flm.kuber.studio/research/.
'''


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf8')


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def csv_bytes(rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader(); writer.writerows(rows)
    return stream.getvalue().encode('utf8')


def tables(report):
    runs = {row['label']: row for row in report['runs']}
    models = []; articles = []; contrasts = []
    for label in sorted(runs):
        run = runs[label]; score = run['score']
        models.append(dict(condition=label, control=run['control'], training_seed=run['seed'], reference=run['reference'],
            checkpoint_step=run['checkpoint_step'], allocated_parameters=run['allocated_parameters'],
            trainable_parameters=run['trainable_parameters'], frozen_parameters=run['frozen_parameters'],
            test_bits_per_byte=score['bits_per_byte'], test_token_perplexity=score['token_perplexity'],
            test_bytes=score['bytes'], test_tokens=score['tokens'], checkpoint_sha256=run['checkpoint_sha256']))
        for article in sorted(score['documents'], key=lambda row: row['document']):
            articles.append(dict(condition=label, article=article['document'], nll=article['nll'],
                                 bytes=article['bytes'], tokens=article['tokens'], bits_per_byte=article['bits_per_byte']))
    for family, key in [('primary', 'primary_contrasts'), ('secondary', 'independent_unit_memory_contrasts')]:
        for row in sorted(report[key], key=lambda row: (row['second'], row['training_seed'])):
            first = runs[f'{row["first"]}-s{row["training_seed"]}']['score']
            second = runs[f'{row["second"]}-s{row["training_seed"]}']['score']
            contrasts.append(dict(family=family, first=row['first'], second=row['second'], training_seed=row['training_seed'],
                first_test_bpb=first['bits_per_byte'], second_test_bpb=second['bits_per_byte'],
                difference_bpb=row['difference_bpb'], lower_95=row['lower_95'], upper_95=row['upper_95'],
                replicates=row['replicates'], bootstrap_seed=row['seed'], resampling_unit=row['unit']))
    return {'tables/model-scores.csv': csv_bytes(models), 'tables/article-scores.csv': csv_bytes(articles),
            'tables/paired-contrasts.csv': csv_bytes(contrasts)}


def extract_checked(archive_path, folder, expected):
    """Verify the serialized archive before writing its strictly scoped paths."""
    folder = Path(folder).resolve()
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != set(expected) or archive.testzip() is not None:
            raise ValueError('Computation records archive inventory or CRC changed')
        for item in archive.infolist():
            name = item.orig_filename; relative = PurePosixPath(name)
            if (name != item.filename or '\\' in name or ':' in name or relative.is_absolute()
                    or '..' in relative.parts or relative.as_posix() != name or item.is_dir()
                    or stat.S_ISLNK(item.external_attr >> 16)):
                raise ValueError('Invalid computation records archive path')
            payload = archive.read(item)
            if payload != expected[name]:
                raise ValueError('Computation records archive payload changed: ' + name)
            path = (folder / name).resolve()
            if not path.is_relative_to(folder):
                raise ValueError('Computation records extraction leaves its directory')
            path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(payload)


def fresh_audit(archive_path, contents, root):
    environment = dict(os.environ); environment.pop('PYTHONPATH', None)
    with tempfile.TemporaryDirectory(prefix='flm-core-records-audit-') as temporary:
        folder = Path(temporary).resolve()
        if folder.is_relative_to(root):
            raise ValueError('Fresh score audit must run outside the repository')
        extract_checked(archive_path, folder, contents)
        runner = ("import runpy,sys; sys.modules['flm']=None; sys.modules['torch']=None; "
                  "sys.argv=['scripts/audit_language_core_report.py','reports/language-core/summary.json']; "
                  "runpy.run_path(sys.argv[0],run_name='__main__')")
        process = subprocess.run([sys.executable, '-E', '-X', 'utf8', '-c', runner], cwd=folder,
            env=environment, capture_output=True, text=True, encoding='utf8', timeout=60, check=True)
        return json.loads(process.stdout)


def package(root):
    root = Path(root).resolve()
    # All six fits, eight selections, identities and scores precede any output.
    report = verified_report(root)
    arithmetic = audit(report)
    if (root / RELEASE).exists() and json.loads((root / RELEASE).read_bytes())['selection_sha256'] != report['selection_sha256']:
        raise ValueError('A different selected study already owns the score-record release')
    identity_name = FOLDER + '/identity.json'; selection_name = FOLDER + '/selection.json'
    identity = json.loads((root / identity_name).read_bytes())
    names = set(identity['sources']) | set(ASSETS) | {identity_name, selection_name, SUMMARY}
    names |= {FOLDER + '/test-' + run['label'] + '.json' for run in report['runs']}
    contents = {name: (root / name).read_bytes() for name in sorted(names)}
    if (digest(contents[identity_name]) != report['study_identity_sha256'] or
            digest(contents[selection_name]) != report['selection_sha256'] or
            json.loads(contents[SUMMARY]) != report):
        raise ValueError('Computation report binding changed during packaging')
    selection = json.loads(contents[selection_name])
    selected = {row['label']: row for row in selection['runs']}
    if len(selection['runs']) != 8 or set(selected) != {run['label'] for run in report['runs']}:
        raise ValueError('Serialized computation selection inventory differs')
    for run in report['runs']:
        if (any(run.get(key) != value for key, value in selected[run['label']].items()) or
                json.loads(contents[FOLDER + '/test-' + run['label'] + '.json']) != run):
            raise ValueError('Serialized computation selection or individual score changed')
        if run['reference']:
            name = f'reports/wikitext2/test-flm-s{run["seed"]}.json'
            if digest(contents[name]) != run['reused_score_sha256'] or json.loads(contents[name])['score'] != run['score']:
                raise ValueError('Serialized reused reference score changed')
    source_bytes = dict(contents)
    contents.update(tables(report))
    contents['audit-report.json'] = encoded(arithmetic)
    contents['requirements-audit.txt'] = b'numpy==2.2.6\n'
    contents['README.md'] = README
    manifest = dict(format='flm-language-core-records-v1', study_identity_sha256=report['study_identity_sha256'],
        selection_sha256=report['selection_sha256'], scope=arithmetic['scope'],
        files={name: dict(bytes=len(data), sha256=digest(data)) for name, data in sorted(contents.items())})
    contents['manifest.json'] = encoded(manifest)
    with tempfile.TemporaryDirectory(prefix='flm-core-records-stage-') as temporary:
        staged_archive = Path(temporary) / 'records.zip'
        with zipfile.ZipFile(staged_archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for name, data in sorted(contents.items()):
                info = zipfile.ZipInfo(name, date_time=(2026, 9, 11, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED; info.external_attr = 0o644 << 16
                archive.writestr(info, data)
        standalone = fresh_audit(staged_archive, contents, root)
        if standalone != arithmetic:
            raise ValueError('Fresh computation score arithmetic differs')
        archive_bytes = staged_archive.read_bytes()
    if any((root / name).read_bytes() != data for name, data in source_bytes.items()):
        raise ValueError('Computation sources or score records changed during packaging')
    release = dict(verified_utc=datetime.now(timezone.utc).isoformat(), archive=ARCHIVE,
        archive_sha256=digest(archive_bytes), bytes=len(archive_bytes), payloads=len(contents),
        study_identity_sha256=report['study_identity_sha256'], selection_sha256=report['selection_sha256'],
        full_checkpoint_and_score_gate_passed=True, arithmetic_audit=arithmetic,
        manifest_sha256=digest(contents['manifest.json']), summary_sha256=digest(contents[SUMMARY]),
        fresh_archive_arithmetic_verified=True, extraction_outside_repository=True,
        repository_pythonpath_removed=True, flm_and_torch_imports_disabled=True,
        python=sys.version, numpy=str(np.__version__), exporter_sha256=digest(Path(__file__).read_bytes()))
    outputs = {ARCHIVE: archive_bytes, PUBLIC_RESULT: contents[SUMMARY], RELEASE: encoded(release), PUBLIC_RELEASE: encoded(release)}
    for name, data in tables(report).items():
        outputs['public/research/language-core-' + Path(name).name] = data
    for name, data in outputs.items():
        path = root / name; path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + '.tmp'); temporary.write_bytes(data); temporary.replace(path)
    return release


if __name__ == '__main__':
    print(json.dumps(package(ROOT), indent=2))
