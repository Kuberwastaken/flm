"""Run or resume the complete registered BabyLM training and evaluation pipeline.

Only one pipeline/training writer may run at a time. Completed stages are verified
and reused; any failed subprocess stops the pipeline before its dependent stage.
"""
import subprocess
import sys


def main():
    for module in ('flm.babylm_suite', 'flm.babylm_test', 'flm.babylm_samples'):
        print(f'Starting {module}', flush=True)
        subprocess.run([sys.executable, '-X', 'utf8', '-m', module], check=True)


if __name__ == '__main__': main()
