# Trabajo de Fin del Grado en Ingenieria en Sistemas de las Telecomunicaciones de la Universidad Rey Juan Carlos.
## Resumen
El volumen de datos que se almacena sobre las personas crece exponencialmente en el tiempo y no tiene previsto parar. El hecho de que las bases de datos estén en muchas ocasiones saturadas de datos, trae consigo que aumente la brecha entre el almacenamiento de esos datos y su utilización real. Para extraer información útil de los datos almacenados es posible usar técnicas de Data Mining. Dichas técnicas consisten en la búsqueda y descubrimiento de patrones, pudiendo construir modelos a partir de los cuales realizar inferencia sobre nuevos casos. El área de \textit{Data Mining} denominada aprendizaje automático permite abordar, entre otras, tareas de clasificación si se dispone de un conjunto de casos previamente etiquetados y suficiente representativo de la tarea a resolver. 

Las enfermedades crónicas suponen una gran demanda de recursos en el Sistema Nacional de Salud. En España, más del 43% de la población adulta es hipertensa, y casi el 20% diabética. Las cronicidades nunca aparecen solas, por lo que comorbilidades aparecen con los años en los pacientes crónicos. Los pacientes con hipertensión duplican el gasto sanitario de los pacientes con tensión normal, y un paciente con diabetes cuesta al año un 67% más que un paciente sin diabetes. A esas cifras hay que sumarle los tratamientos y gastos asociados a las comorbilidades de hipertensión y diabetes. Normalmente, para clasificar a los pacientes crónicos se considera un estado de salud por categoría (enfoque multi-clase). El problema de tener categorías excluyentes entre sí, es cuando aparecen estados de salud que encajan en más de una categoría (enfoque multi-etiqueta). Ese enfoque multi-etiqueta es la manera natural de relacionar estados de salud con categorías, debido a que cada paciente es distinto y no siempre el estado de salud ''encaja'' en una sola categoría. La realización de este trabajo encuentra su motivación en el diseño de sistemas automáticos predictivos del estado de salud, abordando el diseño de dichos sistemas bajo dos enfoques: multi-clase y multi-etiqueta.


%evaluar una manera natural de trar pacinetes cronicos considerar un estado de salud , catogerias con una patologia cronica y que en el tfg evaluo si hay diferencias en las prestaciones obtenidas con el diseño de clasificaores automaticos cuando un refoque multi clase y multi categoria.

Este trabajo tiene como objetivo diseñar modelos predictivos del estado de salud de pacientes, considerando:  pacientes sanos y crónicos (de manera conjunta); y únicamente pacientes crónicos. Las cronicidades en las que nos enfocamos son hipertensión, diabetes y sus comorbilidades. Disponemos de 65.201 casos asignados al Hospital Universitario de Fuenlabrada. Para este trabajo se han considerado 2735 pacientes, representando con el mismo número de casos cada estado de salud considerado. El vector de características de un paciente está formado por características demográficas, diagnósticos y dispensación farmacéutica (2265 campos). Las características clínicas pueden estar codificadas bien como la presencia/ausencia de un diagnóstico o fármaco (características binarias), o bien como el número de veces que ese diagnóstico/fármaco aparece en un periodo de tiempo (ocurrencia).  Dado el número elevado de características frente al número de casos disponibles (2265 vs 2735), se realiza un proceso de selección de características previo al diseño de modelos de clasificación, tanto para multi-clase como para multi-etiqueta. Los esquemas de clasificación evaluados en este trabajo son: Máquinas de Vectores Soporte (lineal y no lineal), Regresión Logística Nominal, Árboles de decisión, \textit{Random Forest} y Perceptrón Multicapa. Las prestaciones de los modelos se evalúan en base a la tasa de acierto (enfoque multi-clase y multi-label), o. Se obtienen varias conclusiones. Por un lado, es preferible considerar características clínicas binarias en  modelos no lineales, y características clínicas basadas en ocurrencia en modelos lineales. La mejor tasa de acierto se obtiene cuando se considera las características clínicas binarias. Por otro lado, se observa un ligero incremento en las prestaciones de los modelos no lineales para los esquemas multi-etiqueta con respecto a los esquemas multi-clase.

Tras la finalización de este trabajo, se concluye que aunque las tasas de acierto para los enfoques multi-clase y multi-etiqueta son muy prometedoras, el número de pacientes considerado para cada categoría limita el aprendizaje de los modelos propuestos. En esa línea, sería muy interesante analizar un número de pacientes más elevado de cada categoría.

## Código Python:

Para poder ejecutar este TFG es necesario las bases de datos (BBDD) de pacientes proporcionadas por el Hospital Universitario de Fuenlabrada. Se adjunta un ejemplo sintético del tipo de BBDD usadas en este TFG.

### Procedimiento seguido para ajustar los modelos a las BBDD:

1. Carga de las 50 BBDD de train y test y listado de características perteneciente a esa configuración de BBDD.

2.  Filtrado de datos duplicados en las BBDD con las características seleccionadas.

3.  Normalización del conjunto de train.

4. Ejecución de la función GridSearch con CrossValidation para la búsqueda de parámetros libres en base al algoritmo de clasificación.

5. Una vez terminada la búsqueda se evalúan los parámetros que mejor tasa de acierto posean y se guardan.

6. Los pasos 2-4 se repiten 50 veces y los parámetros libres que se seleccionan son los que mejor tasa de acierto aporten.

7. Con los mejores parámetros procedemos a la predicción con el conjunto de test.

8. Ajustamos el algoritmo para el conjunto de train normalizado y con los parámetros óptimos.

9. Normalizamos el conjunto de test con la media y la desviación típica del conjunto de train.

10. Predecimos las etiquetas para test, pasando el conjunto de test normalizado por el algoritmo clasificador.

11. Dependiendo si el problema es multi-clase o multi-etiqueta usamos medidas estadísticas diferentes para evaluar los resultados que nos aportan las etiquetas predichas y las etiquetas reales.

12. Los pasos 8-11 los repetimos 50 veces y realizamos la media de las medidas estadísticas obtenidas.
