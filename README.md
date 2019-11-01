# Trabajo de Fin del Grado en Ingenieria en Sistemas de las Telecomunicaciones de la Universidad Rey Juan Carlos.
## Resumen
El volumen de datos que se almacena sobre las personas crece exponencialmente en el tiempo y no tiene previsto parar. El hecho de que las bases de datos estén en muchas ocasiones saturadas de datos, trae consigo que aumente la brecha entre el almacenamiento de esos datos y su utilización real. La técnicas de Data Mining tratan de extraer información útil de los datos. A partir de la búsqueda y descubrimiento de patrones, de manera que se puedan realizar inferencia sobre nuevos casos. El área de Data Mining denominada aprendizaje automático, permite abordar, entre otras, tareas de clasificación si se dispone de un conjunto de casos previamente etiquetados y suficiente representativo de la tarea a resolver.

Las enfermedades crónicas suponen una gran demanda de recursos en el Sistema Nacional de Salud. En España, más del 43% de la población adulta es hipertensa, y casi el 20% diabética. Las cronicidades nunca aparecen solas, por lo que comorbilidades aparecen con los años en los pacientes crónicos. Los pacientes con hipertensión duplican el gasto sanitario de los pacientes con tensión normal; y un paciente con diabetes cuesta al año un 67% más que un paciente sin diabetes. A esas cifras hay que sumarle los tratamientos y gastos de las comorbilidades asociadas a la hipertensión y diabetes. Para evaluar de manera natural a los pacientes crónicos consideramos su estado de salud, asumimos una patología crónica por categoría (multi-clase). Eso es válido para clasificar a los pacientes crónicos en base a una sola cronicidad, pero cuando aparecen las comorbilidades no podemos asociar sólo una patología crónica a un paciente, nos vemos obligados a asignarle otra categoría más (multi-etiqueta). La realización de este trabajo se ha visto motivada por hacer más eficaz el Sistema Nacional de Salud, evaluando las prestaciones obtenidas con el diseño de clasificadores automáticos bajo un enfoque multi-clase y multi-etiqueta.

Este trabajo tiene como objetivo evaluar, en termino de prestaciones, métodos que nos permitan de predecir el estado de salud de pacientes que presenten algún tipo de cronicidad. Para nuestros experimentos hemos tenido en cuenta dos escenarios:  pacientes sanos y crónicos; y  pacientes crónicos. El vector de características para clasificar el estado de salud de un paciente está formado por características demográficas, diagnósticos y dispensa farmacéutica. Los datos del estado de salud pueden estar basados en la presencia de una característica; o en el número de veces que está presente en un periodo de tiempo (ocurrencia).  Una vez seleccionamos las mejores características procedemos a entrenar modelos de clasificación lineales y no lineales, tanto para multi-clase como para multi-etiqueta. Los modelos de clasificación evaluados en este trabajo son: Máquinas de Soporte Vectorial (lineal y no lineal), Regresión Logística Nominal, Árboles de decisión, Random Forest y Multi- Layer Percepton. Evaluamos sus predicciones en base a la tasa de acierto y otras medidas estadísticas. Se obtienen varias conclusiones. Por un lado es preferible considerar la presencia de las características para modelos no lineales; y cuando consideramos la ocurrencia de las características es más adecuada para modelos lineales. Además la presencia de las características ofrece una tasa de acierto mayor con respecto a la ocurrencia. Por otro lado, cuando comparamos las prestaciones de los modelos no lineales, hay una tendencia en los resultados que los esquemas multi-etiqueta se observa en ellos tienden a ofrecer mejores prestaciones que los modelos multi-clase.

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
