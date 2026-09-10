"""Build the LaTeX reports; retain logs for mandatory rendered-page review."""
import argparse
from pathlib import Path
import shutil
import subprocess

root = Path(__file__).resolve().parents[1]
latex = shutil.which('pdflatex') or str(Path.home() / 'AppData/Local/Programs/MiKTeX/miktex/bin/x64/pdflatex.exe')
bibtex = shutil.which('bibtex') or str(Path(latex).with_name('bibtex.exe'))
output = root / 'output/pdf'; output.mkdir(parents=True, exist_ok=True)
work = root / 'work/papers'; work.mkdir(parents=True, exist_ok=True)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--only', nargs='+', choices=('flm', 'data-and-reproduction', 'local-learning', 'wiring-controls'), default=['flm', 'data-and-reproduction', 'local-learning', 'wiring-controls'])
args = parser.parse_args()
for name in args.only:
    commands = [
        [latex, '-interaction=nonstopmode', '-halt-on-error', f'-output-directory={work}', f'{name}.tex'],
        [bibtex, str(work / name)],
        [latex, '-interaction=nonstopmode', '-halt-on-error', f'-output-directory={work}', f'{name}.tex'],
        [latex, '-interaction=nonstopmode', '-halt-on-error', f'-output-directory={work}', f'{name}.tex'],
    ]
    for command in commands:
        result = subprocess.run(command, cwd=root / 'papers', capture_output=True, text=True, errors='replace')
        with (work / 'build-console.txt').open('a', encoding='utf8') as handle: handle.write(result.stdout + result.stderr)
        if result.returncode:
            print((result.stdout + result.stderr)[-7000:]); raise SystemExit(result.returncode)
    shutil.copyfile(work / f'{name}.pdf', output / f'{name}.pdf')
    print(f'Built {output / (name + ".pdf")}; render and inspect before publication.')
