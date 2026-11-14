# Evidencia de funcionamiento

## MLflow Model Registry: `pronostico_ventas`

| Versión | Alias | Tipo | Promovido | Costo validación | Run |
|---|---|---|---|---|---|
| 1 |  | hgb | True | 91770.7 | `234d448c566d` |
| 2 |  | hgb | False | 91026.9 | `4d857af28268` |
| 3 | campeon | hgb | True | 98234.7 | `891b02490fd7` |
| 4 |  | hgb | False | 111567.4 | `238691f55a07` |
| 5 |  | hgb | False | 111567.4 | `9ff7d9671c28` |

## MLflow: experimento `novamercado_pronostico` (15 runs)

| Run | Costo prueba | WAPE | MAE | Costo vigente | Decisión |
|---|---|---|---|---|---|
| inicial | 91,771 | 0.175 | 37.5 |  | No había modelo vigente; se promueve el candidato (costo en validación 91 771) (candidato hgb, versión 1) |
| inicial-baseline | 131,357 | 0.215 | 45.9 |  |  |
| inicial-hgb | 91,771 | 0.175 | 37.5 |  |  |
| reentreno_9c63fa3e146a | 91,027 | 0.178 | 37.7 | 91,646 | Costo en validación 91 027 vs 91 646 del vigente; no mejora al menos 1%, se conserva el vigente (candidato hgb, versión 2) |
| reentreno_9c63fa3e146a-baseline | 111,509 | 0.214 | 45.1 |  |  |
| reentreno_9c63fa3e146a-hgb | 91,027 | 0.178 | 37.7 |  |  |
| reentreno_108b5de6a5a4 | 98,235 | 0.191 | 42.1 | 99,861 | Costo en validación 98 235 vs 99 861 del vigente (candidato hgb, versión 3) |
| reentreno_108b5de6a5a4-baseline | 118,952 | 0.227 | 50.0 |  |  |
| reentreno_108b5de6a5a4-hgb | 98,235 | 0.191 | 42.1 |  |  |
| reentreno_d7e1ff79d315 | 111,567 | 0.200 | 46.6 | 110,020 | Costo en validación 111 567 vs 110 020 del vigente; no mejora al menos 1%, se conserva el vigente (candidato hgb, versión 4) |
| reentreno_d7e1ff79d315-baseline | 131,248 | 0.240 | 55.8 |  |  |
| reentreno_d7e1ff79d315-hgb | 111,567 | 0.200 | 46.6 |  |  |
| reentreno_0b47db918eef | 111,567 | 0.200 | 46.6 | 110,020 | Costo en validación 111 567 vs 110 020 del vigente; no mejora al menos 1%, se conserva el vigente (candidato hgb, versión 5) |
| reentreno_0b47db918eef-baseline | 131,248 | 0.240 | 55.8 |  |  |
| reentreno_0b47db918eef-hgb | 111,567 | 0.200 | 46.6 |  |  |

## Airflow: ejecuciones de `reentrenamiento_pronostico` (4)

| dag_run_id | Estado | Inicio | Fin |
|---|---|---|---|
| reentreno_9c63fa3e146a | success | 2026-11-14T01:22:33.969710Z | 2026-11-14T01:22:56.242411Z |
| reentreno_108b5de6a5a4 | success | 2026-11-14T01:23:11.283420Z | 2026-11-14T01:23:27.463996Z |
| reentreno_d7e1ff79d315 | success | 2026-11-14T01:23:44.267695Z | 2026-11-14T01:24:00.661670Z |
| reentreno_0b47db918eef | success | 2026-11-14T01:24:37.525937Z | 2026-11-14T01:24:59.811835Z |
