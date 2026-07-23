# Reporte de Monitoreo de Sol AI

**Fecha de generación:** 2026-07-22T21:04:41

## 1. Resumen general

| Indicador | Resultado |
|---|---:|
| Total de preguntas | 1 |
| Respuestas con contexto suficiente | 1 |
| Tasa de contexto suficiente | 100.0% |
| Respuestas con contexto insuficiente | 0 |
| Tasa de contexto insuficiente | 0.0% |
| Respuestas sin fuentes | 0 |
| Promedio de fuentes por respuesta | 5.0 |

## 2. Rendimiento

| Indicador | Resultado |
|---|---:|
| Tiempo promedio de respuesta | 4.691 s |
| Mediana del tiempo de respuesta | 4.691 s |
| Tiempo máximo de respuesta | 4.691 s |

## 3. Feedback de usuarios

| Indicador | Resultado |
|---|---:|
| Total de evaluaciones | 1 |
| Feedback positivo | 1 |
| Feedback negativo | 0 |
| Tasa de feedback positivo | 100.0% |
| Tasa de feedback negativo | 0.0% |
| Cobertura de feedback | 100.0% |

## 4. Estado de calidad

| Dimensión | Estado | Interpretación |
|---|---|---|
| Cobertura de contexto | Adecuado | 100.0% de las respuestas tienen contexto suficiente. |
| Satisfacción | Adecuado | 100.0% del feedback es positivo. |
| Tiempo de respuesta | Adecuado | Promedio de 4.691 segundos. |

## 5. Acciones recomendadas

- Revisar las interacciones marcadas con feedback negativo.
- Analizar preguntas con `insufficient_context`.
- Incorporar nuevos documentos solo después de su aprobación en el inventario.
- Ejecutar el pipeline documental después de cada actualización.
- Revisar este reporte periódicamente para detectar cambios en calidad y rendimiento.

## 6. Archivos de origen

- `data/interaction_logs.csv`
- `data/feedback_logs.csv`

Este reporte se genera automáticamente a partir de las interacciones reales de Sol AI.
