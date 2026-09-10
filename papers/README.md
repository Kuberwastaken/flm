# Research papers and generated figures

`flm.tex` contains the methods, completed two-seed WikiText test comparison,
isolated CPU measurements and acute mechanism/physical controls.
`data-and-reproduction.tex` records source transformations, completed BabyLM
preparation and reproduction boundaries. `local-learning.tex` reports fifteen
learning-rule runs and forty physical choice replays, including negative results.
`wiring-controls.tex` adds the complete sixty-run topology/context experiment
and exact-versus-local gradient diagnostics. All four remain working reports
for a continuing research program; larger-data
training, closed-loop neural control and learned gait policies are separate stages.

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
completed-WikiText revision has eight methods pages and five data pages. Preserve
the working-report label while the larger program continues.

The local-learning note has four reviewed pages. Its tables are generated from
the checked choice summaries, and its figures show all three seeds' mean and
observed range. After updating reviewed sources, run
`python scripts/package_papers.py` to create the public source ZIP with a SHA-256
manifest. The archive includes all four LaTeX sources, their included figures,
generating scripts and published figure CSVs; raw corpora and checkpoints are
excluded. Re-running numerical studies requires the repository and environments
described in the main README.

The wiring-controls note has five reviewed pages, including its references.
Its numerical include and three figures are bound to the verified sixty-run
summary. To rebuild it after reproducing those runs:

```sh
python -m flm.wiring_report
python scripts/wiring_figures.py
python scripts/wiring_paper_data.py
python scripts/build_papers.py --only wiring-controls
```

The language topology comparison remains pending. This note does not present
toy-task outcomes as answers to the language-prior question. The separate
language test view is implemented behind the complete-study gate and will be
reviewed with actual test data after the eight new controls finish.
