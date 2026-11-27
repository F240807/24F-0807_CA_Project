# Project Rules: 24F-0807 Computer Architecture Benchmark Project

## Hard Constraints
- NEVER modify, overwrite, or fabricate any raw benchmark CSVs in `data/` (except creating `data/bs1_tail_tpu.csv` from the verified values provided in the prompt and generating `data/amdahl_analysis.csv`).
- EVERY number written into the report MUST be verified directly against the CSVs in `data/`. Do not round inconsistently or invent missing metrics.
- Keep all math symbols Word/Docs-safe (`×`, `→`, `≥`, `%`) rather than raw LaTeX delimiters (`$...$`) inside report prose.
- All plots must use the project's dark theme (`#131313` background, white labels/ticks, 300 DPI) and be saved to `plots/`.

## Verification Standard
- After writing any script, run it in the terminal and confirm exit code 0 and non-empty output files.
- Before claiming any report section is finished, cross-check every table entry against the corresponding CSV in `data/`.