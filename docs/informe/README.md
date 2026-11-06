# Informe en español (v0.3)

`informe_rejas_v0.3.pdf` es la versión en español y en formato de informe técnico de
[`REPORT.md`](../../REPORT.md), con las mismas cifras (ejecución de referencia en Apple M1).

Para recompilarlo desde la fuente, con XeLaTeX (usa fuentes OpenType), desde esta carpeta:

```bash
xelatex informe_rejas_v0.3.tex && xelatex informe_rejas_v0.3.tex
```

Las figuras se leen desde `docs/figures/`. En Overleaf, sube la carpeta `docs/` completa y elige
*Menu → Compiler → XeLaTeX*.
