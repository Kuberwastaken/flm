# Research papers and generated figures

`flm.tex` contains the audited language topology and retrained slow-state
comparisons, subset losses, completed two-seed WikiText baseline comparison,
isolated CPU measurements, acute mechanism controls and the completed twelve-fit BabyLM comparison.
`data-and-reproduction.tex` records source transformations, completed BabyLM
evaluation, overlap analysis, recovery and reproduction boundaries. `local-learning.tex` reports fifteen
learning-rule runs and forty physical choice replays, including negative results.
`wiring-controls.tex` adds the complete sixty-run topology/context experiment
and exact-versus-local gradient diagnostics. `closed-loop.tex` adds 27 physical
pose-feedback conditions, a scripted reference and an exact repeat. All five
remain working reports for a continuing research program; circuit-selection language training,
language-to-control transfer and learned gait policies remain later stages.

```sh
python -m pip install -e ".[language,research]"
python scripts/research_figures.py
python scripts/build_papers.py
```

To update only the local-learning note after reproducing its numerical reports:

```sh
python scripts/behavior_figures.py
python scripts/choice_paper_data.py
python scripts/build_papers.py --only local-learning
```

The build needs a LaTeX distribution with `pdflatex`, `bibtex`, Latin Modern, geometry, amsmath, booktabs, graphicx, xcolor, hyperref and microtype. On Windows the builder also checks the standard per-user MiKTeX location. On other systems put the tools on PATH.

Figures derive from `public/research/validation.json`, the published WikiText model
package and `reports/wikitext2/summary.json`. The script emits PNG/SVG figures,
validation CSV and included LaTeX result tables. Mechanism and physical figures
come from `scripts/continuing_figures.py`, with measured CSV data and provenance
in `public/research/figures/`; their reviewed PNGs are included here. Rerun only
when intentionally updating the snapshot. PDFs go to ignored `output/pdf/` and
build intermediates to ignored `work/papers/`.

Render every PDF page and inspect it before copying reviewed PDFs into
`public/research/`. Check compiler logs for missing citations and overflow. The
completed-BabyLM revision has twelve methods pages and six data pages. Preserve
the working-report label while the larger program continues.

The local-learning note has four reviewed pages. Its tables are generated from
the checked choice summaries, and its figures show all three seeds' mean and
observed range. After updating reviewed sources, run
`python scripts/package_papers.py` to create the public source ZIP with a SHA-256
manifest. The archive includes all five LaTeX sources, their included figures,
generating scripts and published figure CSVs; raw corpora and checkpoints are
excluded. Re-running numerical studies requires the repository and environments
described in the main README.

The source archive also includes the repository README, its embedded
figures, available SVG versions, component notices, and the completed WikiText
test report used by `docs/figures/readme_figures.py`. That chart can be regenerated
with NumPy and Matplotlib without loading a checkpoint. The README is a project
overview: its broader repository links and application/training commands require
the full checkout. The archive itself supports rebuilding the papers from their
included LaTeX, tables and figures with `python scripts/build_papers.py`; no model
weights or raw corpora are required for that build.

The archive additionally includes the SCAN preparation note, source audit,
tokenizer, codec checks and figure generator. From a fresh extraction, run
`python -m flm.scan` to acquire its nine hash-pinned source files, then
`python -m unittest discover -s tests -p test_scan.py` and
`python scripts/scan_data_report.py`. This reproduces data measurements and
figures, not a fitted instruction model. It needs NumPy, tokenizers 0.22.2 and
Matplotlib 3.10.9. The command data are acquired from the publisher rather than
embedded in the archive.

The wiring-controls note has five reviewed pages, including its references.
Its numerical include and three figures are bound to the verified sixty-run
summary. To rebuild it after reproducing those runs:

```sh
python -m flm.wiring_report
python scripts/wiring_figures.py
python scripts/wiring_paper_data.py
python scripts/build_papers.py --only wiring-controls
```

The language topology comparison is complete and appears in the main FLM paper;
the sensory note does not substitute toy-task outcomes for language evidence.
After reproducing the complete study, generate its figure and numerical include
with `python scripts/language_topology_report.py`, then rebuild with
`python scripts/build_papers.py --only flm`. The generator verifies all ten
selected checkpoints and scores before writing `language-topology-measured.tex`
and its figure. The generated include, summary and `paper-inputs.json` retain
their provenance in the source archive. Rebuilding the PDF from these included
artifacts needs no checkpoints. The separate score ZIP supplies a NumPy-only
arithmetic audit, and the separate ten-model ZIP supports local generation.

The closed-loop note has four reviewed pages. Its measurements and figures are
generated only from the complete physical cohort after independent causal replay:

```sh
python scripts/closed_loop_report.py
python scripts/feedback_paper_data.py
python scripts/build_papers.py --only closed-loop
```

Review all pages before copying the PDF to the public site. The separate records
ZIP includes the fixed controllers and every body/neural observation; its own
audit runs from a fresh extraction without repository access. The paper source
ZIP contains the compact summary and generated tables, without duplicating that
45.9 MB record archive.

The BabyLM table generator (`python scripts/babylm_paper_data.py`) audits the frozen table payload and copies hash-verified measured figures before a paper build. Its exact inputs are in `reports/babylm/paper-inputs-v1.json`.
