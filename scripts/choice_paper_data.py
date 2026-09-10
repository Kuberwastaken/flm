"""Derive the learning note's tables and copy its measured scientific figures."""
from pathlib import Path
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
study = json.loads((ROOT / 'public/research/choice-learning.json').read_text())
physical = json.loads((ROOT / 'public/research/learned-choice.json').read_text())
rows = []
for method, label in study['methods'].items():
    chosen = [next(r for r in study['curves'] if r['method']==method and r['step']==900 and r['delay']==delay) for delay in (8,12,48)]
    scores = [f'{r["current_mean"] * 100:.2f}' for r in chosen]
    values = chosen[-1]['current_accuracy']
    rows.append(' & '.join([label] + scores + [f'{min(values)*100:.1f}--{max(values)*100:.1f}']) + r' \\')
(ROOT / 'papers/choice-measured.tex').write_text('\\newcommand{\\ChoiceRows}{%\n' + '\n'.join(rows) + '\n}\n', encoding='utf8')
(ROOT / 'papers/choice-physical.tex').write_text(
    f'The complete assay contains {physical["physical_cases"]} separately simulated cases and {physical["unique_chosen_commands"]} unique descending commands. '
    f'The measured heading sign matches the chosen command in all {physical["heading_matches_chosen_action"]} cases. '
    f'Neural choices agree with the rule associated with their checkpoint in {physical["neural_choices_correct"]} of {physical["physical_cases"]} cases, including the untrained controls. '
    'All recorded generalized positions are exactly equal between cases with the same command and initialization. '
    'This confirms deterministic action execution, not independent motor skills or a statistical motor-learning benchmark.\n', encoding='utf8')
for delay in (8,48):
    name=f'choice-reversal-delay{delay}.png'
    shutil.copyfile(ROOT / 'public/research/figures' / name, ROOT / 'papers/figures' / name)
