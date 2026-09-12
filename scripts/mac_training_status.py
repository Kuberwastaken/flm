"""Read Mac queue metadata and process telemetry without opening model/test data."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time


def command(*args):
    result = subprocess.run(args, capture_output=True, text=True, timeout=10)
    return dict(returncode=result.returncode, output=result.stdout.strip(), error=result.stderr.strip())


def status(root, supervisor):
    now = time.time()
    study = root / 'runs/selection-language-mac-v1'
    reports = root / 'reports/selection-language/mac-v1'
    declarations = list(study.rglob('declaration.json'))
    complete = sorted(study.rglob('complete.json'), key=lambda p: p.stat().st_mtime)
    saved = sorted(study.rglob('saved-*.json'), key=lambda p: p.stat().st_mtime)
    receipt = json.loads((root / supervisor).read_bytes())
    pid = int(receipt['pid'])
    process = command('ps', '-p', str(pid), '-o', 'command=')
    expected = f'{root}/scripts/selection_mac_runtime.py run'
    verified = process['returncode'] == 0 and process['output'].endswith(expected)
    telemetry = command('ps', '-p', str(pid), '-o', 'pid=,etime=,pcpu=,rss=') if verified else None
    cores = command('sysctl', '-n', 'hw.logicalcpu')
    percent = float(telemetry['output'].split()[2]) if telemetry and telemetry['returncode'] == 0 else None
    count = int(cores['output']) if cores['returncode'] == 0 else None

    def marker(path):
        return dict(path=str(path.relative_to(study)), modified_unix=path.stat().st_mtime,
                    age_seconds=round(max(0, now-path.stat().st_mtime), 1))

    active = []
    for declaration in declarations:
        directory = declaration.parent
        if not (directory/'complete.json').exists():
            checkpoints = sorted(directory.glob('saved-*.json'))
            active.append(dict(condition=str(directory.relative_to(study)),
                               committed_step=int(checkpoints[-1].stem.split('-')[1]) if checkpoints else 0))
    has_selection = (reports/'study-selection.json').is_file()
    has_summary = (reports/'test-summary.json').is_file()
    stage = ('evaluation_record_available_requires_audit' if has_summary else 'held_out_evaluation'
             if has_selection else 'validation_selection' if len(complete) == 128 else 'training')
    return dict(checked_at=datetime.now(timezone.utc).isoformat(), completed_fits=len(complete),
                declared_fits=len(declarations), expected_fits=128, stage=stage, active_conditions=active,
                latest_completions=[marker(p) for p in complete[-8:]],
                latest_committed_checkpoint=marker(saved[-1]) if saved else None,
                verified_trainer=verified, process=process, telemetry=telemetry,
                logical_cpu_count=count, process_cpu_percent=percent,
                approximate_machine_cpu_percent=round(percent/count, 2) if percent is not None and count else None,
                power=command('pmset', '-g', 'batt'), thermals=command('pmset', '-g', 'therm'),
                scope='Best-effort read of file metadata and ps/pmset; not a checkpoint integrity or scientific-result audit.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--supervisor', required=True, help='Current receipt from the private connection record')
    args = parser.parse_args()
    print(json.dumps(status(args.root.resolve(), args.supervisor), indent=2))
