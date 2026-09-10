"""Record the original-to-bounded scheduler handoff and its two replayed windows."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flm.language_topology_queue import existing_trainers
from flm.provenance import sha256, write_json


def main():
    path = ROOT / 'reports/language-topology/scheduler-transition.json'
    report = json.loads(path.read_text(encoding='utf8'))
    if 'replay_verification' in report:
        raise ValueError('The one-time handoff verification has already been recorded')
    rows = [json.loads(line) for line in
            (ROOT / 'runs/language-topology-v1/null101-s42/history.jsonl').read_text().splitlines()]
    fields = ['train_loss', 'gradient_norm', 'learning_rate',
              'presented_tokens', 'presented_bytes', 'scored_bytes']
    comparisons = []
    for step in (5100, 5200):
        pair = [row for row in rows if row['step'] == step]
        if len(pair) != 2 or any(pair[0][field] != pair[1][field] for field in fields):
            raise ValueError(f'Replay window does not match the original logged values: {step}')
        comparisons.append(dict(step=step, exact_fields=fields, original=pair[0], resumed=pair[1]))
    trainers = existing_trainers()
    if len(trainers) != 2 or any(not any(f'runs/language-topology-v1/{label}' in row['CommandLine']
                                       for row in trainers) for label in ('null101-s42', 'null103-s42')):
        raise ValueError('The expected two actual training processes are not live')
    command = (
        "$taskMemory = Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory; "
        "$taskCPU = Get-CimInstance Win32_PerfFormattedData_PerfOS_Processor -Filter \"Name = '_Total'\"; "
        "[PSCustomObject]@{available_mib=$taskMemory.AvailableMBytes; "
        "pages_input_per_second=$taskMemory.PagesInputPersec; page_reads_per_second=$taskMemory.PageReadsPersec; "
        "cpu_percent=$taskCPU.PercentProcessorTime; checked_utc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json -Compress"
    )
    resource = json.loads(subprocess.check_output(
        ['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', command],
        text=True, creationflags=subprocess.CREATE_NO_WINDOW))
    resource['qualification'] = 'One observed resource sample on the shared laptop, not an isolated performance benchmark.'
    report['replay_verification'] = dict(
        checked_utc=datetime.now(timezone.utc).isoformat(), verifier_sha256=sha256(Path(__file__)),
        matching_logged_updates_after_resume=200, comparisons=comparisons,
        limitation='Equality of logged losses, gradient norms and exposure fields; no old 5100/5200 full-tensor checkpoint exists for an independent tensor comparison.')
    report['scheduler'] = json.loads((ROOT / 'runs/language-topology-v1/scheduling.jsonl').read_text().splitlines()[-1])
    report['observed_live_trainers'] = trainers
    report['concurrent_resource_observation'] = resource
    write_json(path, report)
    print(json.dumps(dict(replayed_updates=200, exact_fields=fields,
                          active_trainers=len(trainers), resource=resource), indent=2))


if __name__ == '__main__':
    main()
