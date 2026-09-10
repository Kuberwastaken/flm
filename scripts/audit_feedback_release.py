"""Audit the complete downloaded pose-feedback records without running physics."""
import os
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.embodiment.choice_runtime import verify_parity
from experiments.embodiment.verify_feedback import verify_all


def main():
    manifest_path = ROOT / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf8'))
        for name, digest in manifest.items():
            path = (ROOT / name).resolve()
            if not path.is_relative_to(ROOT) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError('Downloaded record or source changed: ' + name)
    parity = verify_parity(ROOT / 'data/controllers/choice-v1')
    rows = verify_all(ROOT)
    print(json.dumps(dict(conditions=len(rows), repeat_arrays_exact=True,
        sensory_and_commands_exact=all(row['controller_replay']['sensory_and_commands_exact'] for row in rows),
        control_frames=sum(row['controller_replay']['frames'] for row in rows),
        delayed_decisions=sum(row['controller_replay']['decisions'] for row in rows),
        maximum_state_logit_error=max(row['controller_replay']['maximum_state_logit_error'] for row in rows),
        runtime_parity=parity), indent=2))


if __name__ == '__main__':
    main()
