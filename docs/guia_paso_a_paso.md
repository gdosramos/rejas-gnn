# Guía paso a paso: qué calles se cierran en Pedro Aguirre Cerda y qué lo predice

> Esta guía acompaña al [informe en inglés](../REPORT.md), a su [versión en PDF en español](informe/)
> y al [README](../README.md). Cuenta, en
> orden y con sus comandos, qué hace cada pieza del estudio y cómo leer lo que imprime. Las marcas
> `[reproducido el AAAA-MM-DD en <máquina>]` indican dónde y cuándo se obtuvo cada cifra.
> Versión 0.3: incorpora líneas base, comparaciones pareadas y el análisis por cohorte, y
> reemplaza las conclusiones de la versión 0.2.

## 0. La pregunta

En Pedro Aguirre Cerda (PAC) muchos pasajes y calles han sido cerrados por los propios vecinos
con rejas y portones. Un levantamiento clasificó los 2 790 tramos de calle de la comuna (según
OpenStreetMap) en `abierta`, `cierre`, `informal`, `medio` o `condominio`. La pregunta es:

**¿Se puede predecir qué tramos se cierran a partir de su posición en la red vial y de las
características de su entorno, y cuánto aporta cada tipo de información?**

Un trabajo exploratorio previo, no publicado, describió el patrón con un campo de densidad
fractal y mapas autoorganizados. Este repositorio convierte la pregunta en una afirmación
**falsable**: un modelo que no vio un barrio debe predecir sus cierres. El diseño es predictivo:
no permite afirmar *por qué* los vecinos cierran.

## 1. Preparar el entorno

```bash
cd ~/Proyectos/rejas-gnn
conda create -n rejas-gnn python=3.11 -y && conda activate rejas-gnn
python3 -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install -r requirements-lock.txt     # versiones exactas de la corrida de referencia
python3 -m pip install -e ".[dev]" --no-deps
python3 -m pytest
```

**Ojo con `python`.** Si tu shell define un alias de `python` (por ejemplo en `~/.zshrc`), puede
apuntar a otro entorno aunque `conda activate` diga lo contrario. Usa `python3`, que es lo que
usa el `Makefile`, y verifica con `which python3`.

**Qué comprueban los 42 tests:** que el grafo dual se construye bien (`test_graph`), que ninguna
variable derivada de la etiqueta entra a las features y que los datos faltantes se marcan con un
indicador por fuente (`test_features`), que ningún tramo supervisado queda a menos de 200 m de un
tramo de test (`test_splits`), que los modelos de grafos usan el grafo y la misma semilla produce
los mismos pesos (`test_models`), que la carga geográfica y la exportación son válidas
(`test_pipeline`) y que el puntaje de habilidad de Brier y el umbral calibrado se calculan bien
(`test_calibration`).

`[reproducido el 2026-11-06 en MacBook Apple M1, Python 3.11: 42 passed]`

## 2. El grafo urbano: qué ve el modelo

```bash
python3 scripts/fig_01_graph.py
```

La unidad de análisis es el **tramo de calle** (arista OSM entre dos esquinas). Para que un
modelo de grafos razone sobre tramos, se construye el **grafo dual** (*line graph*): cada tramo
es un nodo y dos tramos son vecinos si comparten una esquina.

| Magnitud | Valor |
|---|---|
| Tramos (nodos del grafo dual) | 2 790 |
| Cierres sociales (`cierre` + `informal` + `medio`) | 217 (7,8 %) |
| Intersecciones OSM | 1 826 |
| Vecindades en el grafo dual | 6 255 |
| Grado medio | 4,48 (ningún tramo aislado) |
| Features de red y geometría | 28 |
| I de Moran de la etiqueta sobre el grafo dual | 0,170 |

`[reproducido el 2026-10-19 en Apple M1: idéntico]`

**Figuras** (`results/figures/`):
- `fig01a_red_y_cierres.png`: los cierres no están dispersos; forman **"peines"**, filas de
  pasajes cortos y paralelos que desembocan en una colectora. Es la morfología de los conjuntos
  de vivienda social de los años 60–80: el pasaje sin salida es la unidad que se cierra y, como
  vienen en serie, los cierres también.
- `fig01b_grado_dual.png`: los tramos cerrados tienen pocos vecinos; son tramos terminales.
- `fig01c_tasa_por_tipo.png`: `living_street`, `service` y `pedestrian` cierran al 18–23 %,
  `residential` al 1,5 %, las avenidas al 0 %. Una calle sin salida cierra **5,7 veces más**
  (35,5 % vs 6,2 %).

**La I de Moran (0,17)** mide si un tramo cerrado tiende a tener vecinos cerrados: positiva y
significativa, pero moderada. Suficiente para que un reparto aleatorio train/test sea trampa
(§3), insuficiente para que "copiar al vecino" baste como modelo.

## 3. Validación espacial: por qué no se puede partir al azar

```bash
python3 scripts/fig_02_splits.py
```

Si un pasaje de test tiene a sus hermanos del mismo peine en train, cualquier modelo (y un GNN
más que ninguno) obtendría una nota alta copiando al vecino sin haber aprendido nada que sirva
en un barrio no relevado. Por eso:

1. La comuna se corta en **20 bloques contiguos** (k-means sobre coordenadas; `fig02a`).
2. Los bloques se reparten en **5 folds**; en cada fold, sus bloques son el test (rojo en
   `fig02b`), un 25 % de los restantes es validación (azul: para parar el entrenamiento y elegir
   el umbral) y el resto es train (gris).
3. **Buffer**: todo tramo a < 200 m de un tramo de test pierde su etiqueta (amarillo). Sigue en
   el grafo, pero no enseña nada.
4. Se repite con varias semillas (cada una reparte los bloques distinto).

| fold | train | val | test | buffer | cierres test | cierres train |
|---|---|---|---|---|---|---|
| 0 | 1 405 | 361 | 588 | 436 | 60 | 83 |
| 1 | 1 581 | 347 | 605 | 257 | 36 | 120 |
| 2 | 1 588 | 371 | 492 | 339 | 30 | 114 |
| 3 | 1 566 | 244 | 557 | 423 | 36 | 136 |
| 4 | 1 502 | 304 | 548 | 436 | 55 | 107 |

`fig02c`: con un split aleatorio, el **99 %** de los tramos de test tiene un tramo etiquetado a
< 100 m (mediana ≈ 30 m); con bloques + buffer, **ninguno** a < 200 m.

**El precio, que el informe declara:** cada fold entrena con solo 83–136 cierres de 217, el
buffer "gasta" 250–440 tramos y los folds son desiguales (30–60 cierres en test). Las métricas
tendrán varianza alta; es un estimador **pesimista** (extrapolación a un barrio nuevo), que es
el que corresponde a la pregunta.

`[reproducido el 2026-10-19 en Apple M1: tabla idéntica]`

## 4. Las líneas base: cuánto dice la morfología sola

```bash
make baselines
```

Antes de cualquier red neuronal hay que saber qué logra lo más simple, con **las mismas
particiones, semillas y métricas** que la batería: una tabla de 12 celdas
P(cierre | clase de vía, sin salida) y una regresión logística sobre las 28 variables del bloque
A, más dos versiones de la logística sin clase de vía y sin distancia al centroide.

| Modelo | Variables | PR-AUC | ROC-AUC | BSS calibrado |
|---|---|---|---|---|
| Prevalencia constante | – | 0,071 | 0,440 | ≈ 0 |
| Tabla clase de vía × sin salida | 2 | 0,202 ± 0,025 | 0,805 ± 0,018 | +0,011 |
| **Regresión logística** | 28 | **0,286 ± 0,024** | **0,871 ± 0,012** | **+0,056** |
| Logística sin clase de vía | 18 | 0,197 ± 0,018 | 0,772 ± 0,013 | −0,003 |
| Logística sin distancia al centroide | 27 | 0,218 ± 0,018 | 0,822 ± 0,016 | −0,002 |

`[reproducido el 2026-11-06 en Apple M1]`

**Cómo leerlo.** ROC-AUC 0,87: en un barrio nunca visto, el modelo ordena un tramo cerrado por
sobre uno abierto el 87 % de las veces. PR-AUC 0,29 es la métrica honesta con 7,8 % de positivos
(el azar vale 0,078): 3,7 veces el azar. El **BSS** (puntaje de habilidad de Brier) compara las
probabilidades con la estrategia trivial de predecir 7,8 % para todos los tramos: positivo
significa que las probabilidades informan algo; negativo, que no. Quitar la clase de vía o la
distancia al centroide cuesta mucho, así que la señal se concentra en pocas variables, y parte
de ella es *ubicación*, no morfología.

## 5. La batería: ablación × arquitecturas × 10 semillas

```bash
make run-all          # 20-40 min en un portátil
cat results/battery/battery_results.md
```

Entrena un MLP (que ignora el grafo) y tres redes de grafos (GCN, GraphSAGE, GATv2) con el
bloque A (28 variables) y con A+B (101 variables: A más 71 covariables del Censo 2024 y de la
MBHT 2017, y dos indicadores de datos faltantes, uno por fuente), con las semillas 42–51. Todas
las arquitecturas comparten los mismos hiperparámetros, sin ajuste específico.

| Modelo (bloque A) | PR-AUC | ROC-AUC | BSS calibrado |
|---|---|---|---|
| MLP | 0,211 ± 0,023 | 0,787 ± 0,061 | −0,067 |
| GraphSAGE | 0,206 ± 0,031 | 0,755 ± 0,048 | −0,080 |
| GATv2 | 0,127 ± 0,022 | 0,636 ± 0,028 | −0,165 |
| GCN | 0,122 ± 0,019 | 0,611 ± 0,050 | −0,116 |

`[reproducido el 2026-11-06 en Apple M1; tabla completa con A+B en docs/reference_results/v03_apple_m1/]`

Ninguna red supera a la regresión logística de §4, y todas tienen BSS negativo: aunque la
calibración isotónica deja el error de calibración (ECE) cerca de 0,05, sus probabilidades son
peores que predecir la prevalencia. **Un ECE bajo no basta** cuando los positivos son escasos.

## 6. Comparaciones pareadas: qué diferencias son reales

```bash
make paired
cat results/paired/paired_comparisons.md
```

Como la semilla fija la partición, en cada semilla todos los modelos se evalúan sobre los mismos
barrios. Comparar medias sueltas mezcla la variación del modelo con la de la partición; por eso
cada afirmación se contrasta con las 10 diferencias por semilla: diferencia media, intervalo de
confianza del 95 % por bootstrap y prueba de Wilcoxon.

| Comparación (A − B) | ΔPR-AUC [IC 95 %] | ΔROC-AUC [IC 95 %] | Veredicto |
|---|---|---|---|
| Logística − MLP | +0,076 [+0,061; +0,092] | +0,084 [+0,053; +0,125] | la logística gana en 10/10 semillas |
| MLP − GraphSAGE | +0,005 [−0,018; +0,027] | +0,032 [−0,001; +0,064] | sin diferencia demostrable: el grafo no aporta |
| MLP − tabla de 2 variables | +0,009 [−0,002; +0,019] | −0,018 [−0,049; +0,006] | el MLP no ordena mejor que la tabla |
| Bloque A − A+B, MLP | +0,006 [−0,016; +0,030] | +0,050 [+0,008; +0,089] | lo socioeconómico baja el ROC-AUC |
| Bloque A − A+B, GraphSAGE | +0,025 [+0,004; +0,047] | +0,064 [+0,025; +0,094] | lo socioeconómico empeora ambas |
| GraphSAGE − GCN | +0,084 [+0,059; +0,107] | +0,144 [+0,112; +0,174] | GCN es claramente peor |

`[reproducido el 2026-11-06 en Apple M1]`

**Cómo leerlo.** Una diferencia está respaldada cuando su intervalo excluye el cero. Con 10
semillas, el menor *p* posible de Wilcoxon es 0,002, y "no demostrada" no significa "iguales".

## 7. Las dos cohortes de registro

```bash
make cohorts
cat results/cohorts/cohort_analysis.md
```

La columna `año` es el año en que se registró el cierre. Los cierres caen en dos cohortes que
siguen a las dos leyes que regulan los cierres en Chile: 2012–2015 (109 cierres, tras la Ley
20.499 de 2011, que exige una misma vía de acceso y salida) y 2022–2025 (108, tras la Ley 21.411
de 2022, que admite calles con acceso y salida distintos). Los tramos abiertos llevan el año del
levantamiento (2025). Si el modelo hubiera aprendido solo la *elegibilidad legal* de la primera
ley, no debería transferirse a la segunda.

| Entrenado con cierres de | Evaluado en 2012–2015 | Evaluado en 2022–2025 |
|---|---|---|
| ambas cohortes | ROC-AUC 0,894 ± 0,017 | 0,847 ± 0,013 |
| solo 2012–2015 | 0,871 ± 0,011 | **0,807 ± 0,019** |
| solo 2022–2025 | 0,848 ± 0,030 | **0,800 ± 0,016** |

`[reproducido el 2026-11-06 en Apple M1]`

**Cómo leerlo.** El modelo que solo vio los cierres de la primera ley ordena los de la segunda
igual de bien que uno entrenado con ellos (diferencia −0,007, IC [−0,020; +0,006]). Ambas cohortes
comparten el mismo patrón. La primera es algo más típica, con más pasajes sin salida (31 % frente
a 19 %), como cabe esperar de la primera ley.

## 8. El mapa

```bash
make figures
```

`results/baselines/logreg_all_p_closure.gpkg` guarda, para cada tramo, la probabilidad calibrada
que la regresión logística le asignó **cuando ese tramo era de test**, promediada sobre 10
semillas, y la desviación entre semillas. Es el único modelo con BSS positivo, por eso es el mapa
del informe.

- `fig03a_mapa_logreg_all.png`: a la izquierda, el mapa; a la derecha, la incertidumbre entre
  semillas.
- `fig03b_calibracion_logreg_all.png`: curva de confiabilidad y separación entre abiertos y
  cerrados.
- `fig03c_candidatos_logreg_all.png` y `candidatos_logreg_all.csv`: los tramos **abiertos** con
  mayor probabilidad. Los primeros son pasajes y senderos como Pasaje 6 Norte, Pasaje 1, Huafo,
  Anaxágoras, Tongoy y Paseo Grohnert (p ≈ 0,47–0,60). Algunos senderos sin nombre conviene
  verificarlos en terreno, porque podrían no ser pasajes cerrables.

**Cómo leerlo.** El mapa sirve como **priorización**: el 10 % de tramos con mayor probabilidad
(279) contiene 83 de los 217 cierres (38 %, 3,8 veces lo que daría un orden al azar). Hay 207
tramos abiertos con p ≥ 0,25, frente a 96 de los cierres reales: son hipótesis que un
levantamiento futuro puede confirmar o refutar.

`[reproducido el 2026-11-06 en Apple M1: ROC-AUC 0,871, PR-AUC 0,258 del promedio de semillas; 207 candidatos]`

## 9. Reproducibilidad: qué es exacto y qué no

- **Dentro de una máquina**: bit a bit (tests `test_seed_reproducibility`,
  `test_run_experiment_is_deterministic`).
- **Entre máquinas** (Linux x86 frente a macOS Apple M1): particiones, grafo, variables y tests
  son idénticos; el entrenamiento de las redes difiere en los últimos decimales por aritmética de
  coma flotante. Las AUC coinciden a la tercera decimal; las métricas con umbral (F1, MCC) pueden
  moverse en la segunda. Por eso las AUC son las métricas principales.
- Las referencias versionadas están en `docs/reference_results/v03_apple_m1/`; las de versiones
  anteriores, con la explicación de por qué difieren, en `docs/archive/v0.1-v0.2/`. `results/` se
  genera y no se versiona.

## Limitaciones declaradas

- La señal depende en parte de la ubicación (distancia al centroide), no solo de la morfología.
- Las covariables del bloque B (Censo 2024, MBHT 2017) se midieron, para la mayoría de los
  cierres, después del registro del cierre.
- Las redes neuronales comparten una receta de entrenamiento sin ajuste por arquitectura.
- Los tramos son rectas entre extremos (no hay polilíneas).
- `condominio` se trata como no cierre para mantener la definición del trabajo previo
  (`data.drop_condominio` lo cambia).
