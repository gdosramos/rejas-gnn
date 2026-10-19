# Guía paso a paso: de la red vial de Pedro Aguirre Cerda a un mapa de probabilidad de cierre

> Esta guía acompaña al [README](../README.md) técnico. Cuenta, en orden y con sus comandos,
> qué hace cada pieza del estudio y qué significa lo que imprime. Los números marcados con
> `[reproducido el AAAA-MM-DD en <máquina>]` deben actualizarse cada vez que alguien repita el
> recorrido; los que no coinciden exactamente se explican en §Reproducibilidad.

## 0. La pregunta

En Pedro Aguirre Cerda (PAC) muchos pasajes y calles han sido cerrados por los propios vecinos
con rejas y portones. Un levantamiento en terreno clasificó los 2 790 tramos de calle de la
comuna (según OpenStreetMap) en `abierta`, `cierre`, `informal`, `medio` o `condominio`.
La pregunta del estudio es:

**¿Se puede predecir qué tramos se cierran a partir de su posición en la red vial y de las
características de su entorno, y cuánto aporta cada tipo de información?**

El estudio original (dos enfoques: campo de densidad fractal y mapas auto-organizados sobre
índices de bienestar territorial) describía patrones; este repositorio convierte la pregunta en
una afirmación **falsable**: un modelo que no vio un barrio debe predecir sus cierres.

## 1. Preparar el entorno

```bash
cd ~/Proyectos/rejas-gnn
conda create -n rejas-gnn python=3.11 -y && conda activate rejas-gnn
PY=~/miniconda3/envs/rejas-gnn/bin/python          # usar siempre este intérprete
$PY -m pip install torch --index-url https://download.pytorch.org/whl/cpu
$PY -m pip install -e ".[dev]"
$PY -m pytest -v
```

**Qué comprueban los 39 tests:** que el grafo dual se construye bien (`test_graph`), que ninguna
variable derivada de la etiqueta entra a las features (`test_features`), que ningún tramo
supervisado queda a < 200 m de un tramo de test (`test_splits`), que los GNN usan el grafo y los
baselines no, y que la misma semilla produce los mismos pesos (`test_models`), y que la carga
geográfica, los tensores y la exportación son válidos (`test_pipeline`).

`[reproducido el 2026-10-19 en MacBook Apple M1, Python 3.11: 39 passed in 93 s]`

## 2. El grafo urbano: qué ve el modelo

```bash
$PY scripts/fig_01_graph.py
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
$PY scripts/fig_02_splits.py
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

**El precio, que el paper debe declarar:** cada fold entrena con solo 83–136 cierres de 217, el
buffer "gasta" 250–440 tramos y los folds son desiguales (30–60 cierres en test). Las métricas
tendrán varianza alta; es un estimador **pesimista** (extrapolación a un barrio nuevo), que es
el que corresponde a la pregunta.

`[reproducido el 2026-10-19 en Apple M1: tabla idéntica]`

## 4. Un primer modelo: cuánto dice la morfología sola

```bash
$PY -m rejas_gnn.cli run --config configs/logreg.yaml --set data.feature_set=geometry name=paso4_logreg_geometria
$PY -m rejas_gnn.cli run --config configs/sage.yaml   --set data.feature_set=geometry name=paso4_sage_geometria
```

Solo las 28 variables de red y geometría; 5 semillas × 5 folds; evaluación únicamente en los
bloques de test.

| Modelo | PR-AUC | ROC-AUC | F1 | I Moran residuos |
|---|---|---|---|---|
| Regresión logística | 0,274 ± 0,126 | 0,836 ± 0,082 | 0,307 ± 0,150 | 0,253 |
| GraphSAGE | 0,257 ± 0,108 | 0,831 ± 0,048 | 0,316 ± 0,140 | 0,232 |

**Cómo leerlo.** ROC-AUC 0,83: en un barrio nunca visto, el modelo ordena correctamente un
tramo cerrado frente a uno abierto el 83 % de las veces. PR-AUC 0,26 es la métrica honesta con
7,8 % de positivos (el azar vale 0,078): 3,4× el azar. El ± refleja la varianza entre folds
anticipada en §3. La I de Moran de los residuos (0,23) dice que lo que el modelo no explica
sigue agrupado: hay estructura local que estas 28 variables no capturan. Y **la regresión
logística iguala a GraphSAGE**: con validación honesta, mirar a los vecinos no suma de forma
medible.

Conviene abrir `results/paso4_sage_geometria/metrics_per_fold.csv`: hay folds que paran en la
época 2 con F1 = 0 porque la validación tenía pocos cierres; con 10 semillas estos accidentes
se promedian, pero el CSV deja que cualquiera los audite.

`[reproducido el 2026-10-19 en Apple M1: SAGE PR-AUC 0,257 ± 0,107, ROC-AUC 0,831 ± 0,048, F1 0,313 ± 0,134, Moran 0,223]`

## 5. La batería: ablación × arquitecturas × 10 semillas

```bash
nohup $PY scripts/run_battery.py > results/battery_run.out 2>&1 &
grep "== " results/battery_run.out | cut -c66-        # avance: 8 líneas al terminar
```

Responde tres preguntas a la vez: (1) ¿la socioeconomía zonal (Censo 2024 + MBHT 2017, 140
variables) mejora la predicción? — Experimento A (solo geometría, 28 features) vs B
(geometría + socio, 168 features); (2) ¿qué arquitectura funciona? — MLP (sin grafo), GCN,
GraphSAGE, GATv2; (3) ¿cuánto varían los resultados? — semillas 42–51. Son 400 entrenamientos.
Además **calibra** las probabilidades (isotónica ajustada solo en validación) y exporta un mapa
por experimento.

Resultado de referencia (Linux x86, 2026-10-18; métricas sobre las predicciones *out-of-fold*
de cada semilla, media ± DE sobre 10 semillas):

| Features | Modelo | PR-AUC | ROC-AUC | Brier raw → calibrado | ECE raw → calibrado |
|---|---|---|---|---|---|
| geometría | MLP | **0,214 ± 0,020** | **0,794 ± 0,045** | 0,171 → 0,077 | 0,241 → 0,053 |
| geometría | SAGE | 0,207 ± 0,033 | 0,757 ± 0,048 | 0,145 → 0,077 | 0,184 → 0,051 |
| geo + socio | SAGE | 0,191 ± 0,038 | 0,710 ± 0,065 | 0,119 → 0,083 | 0,120 → 0,059 |
| geo + socio | MLP | 0,186 ± 0,023 | 0,712 ± 0,055 | 0,140 → 0,080 | 0,177 → 0,054 |
| geo + socio | GAT | 0,130 ± 0,031 | 0,616 ± 0,045 | 0,180 → 0,089 | 0,217 → 0,073 |
| geometría | GAT | 0,123 ± 0,018 | 0,630 ± 0,032 | 0,179 → 0,084 | 0,240 → 0,061 |
| geometría | GCN | 0,122 ± 0,020 | 0,610 ± 0,050 | 0,193 → 0,080 | 0,277 → 0,055 |
| geo + socio | GCN | 0,116 ± 0,026 | 0,592 ± 0,050 | 0,165 → 0,092 | 0,208 → 0,072 |

**Tres lecturas:**
1. **Añadir socioeconomía empeora** (PR-AUC 0,214 → 0,186; ROC-AUC 0,79 → 0,71). Con 217
   positivos y 168 variables, el bloque socio (espacialmente suave y de 2017) aporta más ruido
   que señal fuera de muestra. Coincide con que las 94 "candidatas" del SOM original no se
   recuperan con un modelo predictivo (AUC 0,53).
2. **La morfología local es la señal**; el MLP sin grafo iguala o supera a GraphSAGE, y GCN /
   GATv2 fallan (sobre-suavizado y sobreajuste con pocos positivos). El message passing no
   aporta valor incremental medible en PAC.
3. **Sin calibrar, las probabilidades no son probabilidades** (ECE 0,18–0,28 por el
   `pos_weight` balanceado); calibradas, ECE ≈ 0,05. Los mapas usan la versión calibrada.

Reproducción en Apple M1 (2026-10-19), misma batería:

| Features | Modelo | PR-AUC (M1) | ROC-AUC (M1) | PR-AUC (Linux) | ROC-AUC (Linux) |
|---|---|---|---|---|---|
| geometría | MLP | 0,211 ± 0,023 | 0,787 ± 0,061 | 0,214 ± 0,020 | 0,794 ± 0,045 |
| geometría | SAGE | 0,206 ± 0,031 | 0,755 ± 0,048 | 0,207 ± 0,033 | 0,757 ± 0,048 |
| geo + socio | SAGE | 0,194 ± 0,042 | 0,716 ± 0,066 | 0,191 ± 0,038 | 0,710 ± 0,065 |
| geo + socio | MLP | 0,192 ± 0,023 | 0,727 ± 0,056 | 0,186 ± 0,023 | 0,712 ± 0,055 |
| geo + socio | GAT | 0,132 ± 0,030 | 0,622 ± 0,041 | 0,130 ± 0,031 | 0,616 ± 0,045 |
| geometría | GAT | 0,127 ± 0,022 | 0,636 ± 0,028 | 0,123 ± 0,018 | 0,630 ± 0,032 |
| geometría | GCN | 0,122 ± 0,019 | 0,611 ± 0,050 | 0,122 ± 0,020 | 0,610 ± 0,050 |
| geo + socio | GCN | 0,117 ± 0,028 | 0,589 ± 0,048 | 0,116 ± 0,026 | 0,592 ± 0,050 |

Todas las diferencias entre plataformas están por debajo de la DE entre semillas y muy por
debajo de las diferencias entre modelos: el ranking (MLP ≈ SAGE ≫ GAT ≈ GCN) y el efecto de
la ablación (la socioeconomía empeora) se replican. Incidente registrado: una primera corrida en
M1 reutilizó tres archivos de referencia que viajaban en `results/`; por eso las referencias
viven ahora en `docs/reference_results/` y `results/` nace vacío.

## 6. El mapa de probabilidad y la calibración

```bash
$PY scripts/fig_03_map.py --experiment geometry__mlp     # o geometry__sage, geometry_socio__mlp…
```

La batería guardó, para cada tramo, la probabilidad calibrada de cierre que el modelo le asignó
**cuando ese tramo era test**, promediada sobre 10 semillas, y la desviación entre semillas.
Es el producto final: un mapa P(cierre | morfología) para el 100 % de la comuna, con su
incertidumbre (`results/battery/<experimento>_p_closure.gpkg` y `.geojson`).

- `fig03a`: izquierda, el mapa; los "peines" se iluminan aunque el modelo nunca vio sus
  etiquetas al predecirlos. Derecha, la incertidumbre, concentrada en los peines del
  surponiente, donde cada fold dejaba pocos cierres para entrenar.
- `fig03b`: curva de confiabilidad (tras calibrar, cuando el modelo dice 0,3 cierra ≈ 30 %) y
  separación de distribuciones entre abiertos y cerrados.
- `fig03c` y `candidatos_<experimento>.csv`: tramos **abiertos** con mayor P(cierre). En ambas
  plataformas el top-5 son cinco pasajes `living_street` del surponiente (Los Jazmines, Las
  Maravillas, Los Geranios, Las Buganvilias, Los Tulipanes) empatados en P = 0,599 porque
  comparten exactamente el mismo vector de features (misma geometría repetida en serie), con DE
  entre semillas 0,34; luego tramos de Av. Clotario Blest (P ≈ 0,52–0,54).

**Lectura para el paper.** PR-AUC ≈ 0,20 dice que el mapa no adivina tramo a tramo; ROC-AUC
0,83 y la calibración lo hacen útil como **priorización**: el decil superior concentra una
fracción desproporcionada de los cierres. Los ~227 tramos abiertos con P ≥ 0,25 son una
hipótesis que un levantamiento futuro puede confirmar o refutar: eso hace al estudio predictivo
en sentido estricto, a diferencia de las "candidatas" descriptivas del SOM original.

`[reproducido el 2026-10-19 en Apple M1: mismo top-10, P idénticas en los cinco empatados,
diferencias en la tercera decimal en Clotario Blest]`

## 7. Reproducibilidad: qué es exacto y qué no

- **Dentro de una máquina**: bit a bit (tests `test_seed_reproducibility`,
  `test_run_experiment_is_deterministic`).
- **Entre máquinas** (Linux x86 vs macOS Apple M1, verificado el 2026-10-19): las particiones,
  el grafo, las features y los tests son idénticos; el entrenamiento difiere en el último
  decimal por aritmética de coma flotante.
  Las métricas de ordenamiento (PR-AUC, ROC-AUC) coinciden a la tercera decimal; las que
  dependen de un umbral (F1, MCC) pueden moverse en la segunda. Por eso el estudio reporta las
  AUC como métricas principales.

## Limitaciones declaradas
- Geometrías de los tramos construidas como rectas entre extremos (no hay polilíneas).
- Covariables de 2017/2024 frente a un levantamiento mayoritariamente de 2025.
- La columna `año` del levantamiento y su protocolo de campo no están documentados.
- `condominio` se trata como no-cierre para mantener la definición del estudio original
  (`data.drop_condominio` lo cambia).
