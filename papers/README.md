# Research papers and generated figures

`flm.tex` is the methods and preliminary-results report; `data-and-reproduction.tex` records source data, transformations and reproduction boundaries. Both are working reports. Final test results and mechanism controls are not yet completed.

```sh
python -m pip install -e ".[language,research]"
python scripts/research_figures.py
python scripts/build_papers.py
```

The build needs a LaTeX distribution with `pdflatex`, `bibtex`, Latin Modern, geometry, amsmath, booktabs, graphicx, xcolor, hyperref and microtype. On Windows the builder also checks the standard per-user MiKTeX location. On other systems put the tools on PATH.

Figures derive from `public/research/validation.json` and the exact published WikiText model package. The script emits PNG and editable SVG figures, the validation CSV and an included LaTeX results table. Rerun only when intentionally updating the report snapshot. The two PDFs are written to ignored `output/pdf/`; build intermediates remain in ignored `work/papers/`.

Render every PDF page and inspect it before copying reviewed PDFs into `public/research/`. Check compiler logs for missing citations and overflow. The first report has six pages and the companion has four in the initial reviewed build. Preserve the working-report label until all declared results have been completed and the corresponding text is updated.
