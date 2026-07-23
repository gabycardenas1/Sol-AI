# Plan de Mantenimiento y Mejora Continua de Sol AI

## 1. Objetivo

Establecer las actividades necesarias para mantener actualizada, segura, trazable y útil la solución Sol AI de NexoData Consulting.

Este plan cubre:

- actualización documental;
- revisión de calidad de respuestas;
- análisis de feedback;
- monitoreo de tiempos y cobertura;
- mejora del modelo RAG;
- control de cambios;
- gestión de incidentes;
- actualización de dependencias y modelos.

---

## 2. Alcance

El plan aplica a los siguientes componentes:

- documentos almacenados en `documents/`;
- inventario documental;
- procesamiento y fragmentación;
- embeddings;
- base vectorial ChromaDB;
- recuperación y reranking;
- generación con Gemini;
- interfaz Streamlit;
- logs de interacción;
- feedback de usuarios;
- pipeline de actualización documental;
- reporte de monitoreo.

---

## 3. Responsables

| Rol | Responsabilidad principal |
|---|---|
| Responsable documental | Revisar vigencia, versión, estado y contenido de los documentos |
| Responsable funcional | Validar que las respuestas sean útiles para los usuarios internos |
| Responsable técnico | Mantener código, dependencias, embeddings, ChromaDB y despliegue |
| Responsable de calidad | Revisar métricas, feedback negativo y casos problemáticos |
| Responsable de seguridad | Verificar accesos, protección de credenciales y tratamiento de información |

En un proyecto pequeño, una misma persona puede asumir varios roles, pero las responsabilidades deben permanecer claramente documentadas.

---

## 4. Mantenimiento documental

### 4.1 Frecuencia

- Revisión mensual del inventario documental.
- Revisión inmediata cuando un área informe una nueva versión.
- Ejecución del pipeline después de agregar, modificar o eliminar un documento.
- Revisión trimestral de documentos que no hayan sido actualizados recientemente.

### 4.2 Procedimiento

1. El área responsable entrega el documento actualizado.
2. Se revisa que el archivo esté completo y aprobado.
3. Se actualiza el inventario documental.
4. El documento anterior se reemplaza o se marca como no vigente.
5. Se ejecuta:

```bash
python -m src.document_update_pipeline
```

6. Se revisa `data/index_update_logs.csv`.
7. Se prueba al menos una pregunta relacionada con el documento modificado.
8. Se registra el cambio en Git.

### 4.3 Regla de publicación

Solo se indexarán documentos con estado aprobado y vigente en el inventario documental.

No se deben incorporar:

- borradores;
- documentos duplicados;
- versiones no aprobadas;
- archivos sin responsable;
- contenido con información sensible no autorizada.

---

## 5. Monitoreo de calidad

### 5.1 Frecuencia

- Revisión semanal durante la etapa inicial.
- Revisión mensual cuando el sistema esté estable.
- Revisión extraordinaria después de cambios importantes.

### 5.2 Ejecución

```bash
python -m src.monitoring_report
```

El reporte generado debe revisarse en:

```text
reports/monitoring_report.md
```

### 5.3 Indicadores mínimos

| Indicador | Meta inicial |
|---|---:|
| Respuestas con contexto suficiente | 85% o más |
| Respuestas sin fuentes | Menos de 5% |
| Feedback positivo | 80% o más |
| Tiempo promedio de respuesta | 8 segundos o menos |
| Cobertura de feedback | 30% o más |
| Fuentes promedio por respuesta | Entre 2 y 5 |

Estas metas son iniciales y deben ajustarse cuando exista una muestra mayor de interacciones.

---

## 6. Tratamiento del feedback negativo

Cada respuesta marcada como negativa debe revisarse mediante su `interaction_id`.

### 6.1 Proceso de revisión

1. Buscar la interacción en `data/interaction_logs.csv`.
2. Revisar la pregunta, respuesta y fuentes utilizadas.
3. Clasificar el problema.
4. Definir la acción correctiva.
5. Probar nuevamente la pregunta.
6. Registrar el cambio realizado.

### 6.2 Clasificación de errores

| Tipo de problema | Acción recomendada |
|---|---|
| Documento inexistente | Solicitar o incorporar una fuente aprobada |
| Documento desactualizado | Actualizar inventario y reconstruir el índice |
| Recuperación incorrecta | Ajustar filtros, cantidad de resultados o reranking |
| Respuesta poco clara | Mejorar el prompt del agente |
| Respuesta sin sustento | Revisar umbrales de confianza y fallback |
| Tiempo elevado | Revisar cantidad de documentos recuperados y modelo |
| Fuente irrelevante | Ajustar reranker o metadatos |
| Consulta fuera de alcance | Mantener o reforzar el fallback |

---

## 7. Ciclo de mejora continua

Sol AI seguirá el siguiente ciclo:

1. **Observar:** recopilar interacciones, tiempos, fuentes y feedback.
2. **Analizar:** identificar errores frecuentes y preguntas sin cobertura.
3. **Priorizar:** seleccionar mejoras según impacto y frecuencia.
4. **Corregir:** ajustar documentos, prompts, filtros o modelos.
5. **Probar:** ejecutar casos de prueba antes de publicar.
6. **Desplegar:** actualizar el sistema y registrar el cambio.
7. **Medir:** comparar métricas antes y después de la mejora.

---

## 8. Actualización de modelos y embeddings

No se deben cambiar modelos únicamente porque exista una versión nueva.

Una actualización se justificará cuando:

- disminuya la calidad de recuperación;
- aumenten las respuestas insuficientes;
- aparezca un modelo claramente mejor para español;
- el modelo actual deje de tener soporte;
- exista una mejora importante en costo, velocidad o precisión;
- se detecten riesgos de seguridad o compatibilidad.

Antes de reemplazar un modelo:

1. guardar la configuración actual;
2. crear una rama de prueba;
3. ejecutar un conjunto fijo de preguntas;
4. comparar recuperación, tiempo y calidad;
5. documentar los resultados;
6. aprobar el cambio;
7. reconstruir la base vectorial si cambian los embeddings.

---

## 9. Actualización de dependencias

### Frecuencia

- Revisión mensual de dependencias.
- Actualización inmediata ante vulnerabilidades críticas.
- Evitar actualizaciones masivas sin pruebas.

### Procedimiento

1. crear una rama;
2. actualizar una dependencia a la vez;
3. ejecutar pruebas;
4. iniciar Streamlit;
5. probar recuperación, generación, fuentes y feedback;
6. actualizar `requirements.txt`;
7. hacer commit con el cambio.

Ejemplo:

```bash
git checkout -b chore/update-dependencies
pip install -U nombre-paquete
pip freeze > requirements.txt
```

---

## 10. Pruebas mínimas después de cada cambio

Se deben ejecutar al menos estos casos:

1. pregunta con respuesta clara en los documentos;
2. pregunta que requiera varias fuentes;
3. pregunta fuera de alcance;
4. pregunta ambigua;
5. consulta sobre un archivo Excel o CSV;
6. prueba de visualización de fuentes;
7. prueba de feedback positivo y negativo;
8. validación del tiempo de respuesta;
9. ejecución del pipeline documental;
10. generación del reporte de monitoreo.

---

## 11. Gestión de incidentes

Se considera incidente cuando:

- Sol AI no inicia;
- ChromaDB no puede abrirse;
- Gemini no responde;
- se muestran fuentes incorrectas;
- se expone información sensible;
- se pierde el historial o los logs;
- el pipeline falla;
- la interfaz deja de funcionar.

### Procedimiento

1. registrar fecha y descripción;
2. conservar el mensaje de error;
3. identificar el componente afectado;
4. detener el servicio si existe riesgo;
5. corregir en una rama;
6. probar;
7. desplegar;
8. documentar la causa y solución.

---

## 12. Control de versiones

Todo cambio debe registrarse en Git.

Convenciones sugeridas:

- `feat:` nueva funcionalidad;
- `fix:` corrección;
- `docs:` documentación;
- `refactor:` mejora interna;
- `test:` pruebas;
- `chore:` mantenimiento;
- `perf:` rendimiento.

Ejemplos:

```bash
git commit -m "docs: add maintenance and improvement plan"
git commit -m "fix: improve fallback for unsupported questions"
git commit -m "chore: update document inventory"
```

---

## 13. Respaldo y recuperación

Se recomienda respaldar periódicamente:

- `documents/`;
- inventario documental;
- `data/interaction_logs.csv`;
- `data/feedback_logs.csv`;
- `data/index_update_logs.csv`;
- `data/document_manifest.json`;
- código fuente;
- configuración de despliegue.

La carpeta `.env` no debe subirse al repositorio.

La base vectorial puede reconstruirse desde los documentos, por lo que el respaldo prioritario debe centrarse en documentos, inventario, logs y código.

---

## 14. Periodicidad resumida

| Actividad | Frecuencia |
|---|---|
| Revisar feedback negativo | Semanal |
| Generar reporte de monitoreo | Semanal al inicio, luego mensual |
| Revisar inventario documental | Mensual |
| Actualizar índice | Cuando cambien documentos |
| Revisar dependencias | Mensual |
| Evaluar modelos | Trimestral o por necesidad |
| Probar fallback y fuentes | Después de cada cambio |
| Revisar accesos y seguridad | Trimestral |
| Respaldar logs y documentos | Mensual |

---

## 15. Criterios de cierre de una mejora

Una mejora se considera completada cuando:

- el problema fue identificado;
- la causa fue documentada;
- el cambio fue implementado;
- las pruebas fueron satisfactorias;
- no se afectaron otras funciones;
- las métricas no empeoraron;
- el cambio fue registrado en Git;
- la documentación fue actualizada.

---

## 16. Conclusión

El mantenimiento de Sol AI debe tratarse como un proceso continuo y no como una actividad aislada.

La calidad del agente depende de tres elementos principales:

- documentos confiables y actualizados;
- monitoreo constante de respuestas;
- mejora controlada de modelos, prompts y componentes técnicos.

Este plan permite conservar la trazabilidad, reducir errores y asegurar que Sol AI continúe siendo útil para los colaboradores de NexoData Consulting.
