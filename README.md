# Interfaz MLP para Predicción de Energía CNT-H₂

## Archivos del proyecto

| Archivo | Descripción |
|----------|-------------|
| main.py | Interfaz gráfica principal |
| mlp.py | Carga y ejecución del modelo MLP |
| convertir.py | Conversión y preprocesamiento de datos |
| graficas.py | Generación de gráficas |
| modelos/ | Modelos entrenados y escaladores |

## Pantalla principal
La pantalla principal de la aplicación permite realizar la conversión de archivos, 
ejecutar predicciones de energía de interacción y visualizar los resultados 
obtenidos por el modelo de red neuronal. 

### Pestaña: Predicción desde archivo

| Elemento | Función|
|----------|-------------|
| Predicción desde archivo | Permite seleccionar y cargar un archivo de entrada en formato CSV o TXT. |
| Convertir a CSV | Convierte un archivo de texto (.txt) al formato CSV (.csv) para facilitar su procesamiento. |
| Calcular energía | Ejecuta el modelo de red neuronal y realiza la predicción de la energía de interacción. |
| Limpiar | Elimina los datos cargados y restablece los campos de la interfaz a su estado inicial. |
| Espacio de resultados | Muestra la energía de interacción predicha y los mensajes generados durante la ejecución. |

Una vez convertido el documento usted podrá calcular la energía de interacción 
Formato del archivo CSV 
• Coordenadas atómicas  
• Ángulo  
• Distancia  
• Energía real

La pestaña Predicción desde archivo constituye el principal módulo de la aplicación. 
Mediante esta sección, el usuario puede cargar archivos de entrada, convertirlos al 
formato adecuado cuando sea necesario y ejecutar la predicción de la energía de 
interacción utilizando el modelo entrenado. Los resultados se muestran 
automáticamente en el área destinada para tal fin.

### Pestaña: Métricas de evaluación 

| Elemento | Función|
|----------|-------------|
| Coeficiente de determinación (R²) | Muestra el porcentaje de variabilidad de la energía de interacción explicado por el modelo. |
| Error Absoluto Medio (MAE) | Presenta el error promedio absoluto entre los valores reales y los valores predichos por la red neuronal. |
| Error Cuadrático Medio (MSE) | Indica el promedio de los errores elevados al cuadrado, utilizado para evaluar el desempeño global del modelo. |
| Raíz del Error Cuadrático Medio (RMSE) | Muestra el error de predicción expresado en las mismas unidades de la energía de interacción (eV). |

### Pestaña: Gráficas de resultados
| Elemento | Función|
|----------|-------------|
| Energía real vs. energía predicha | Muestra la comparación entre los valores reales obtenidos mediante simulación y los valores estimados por la red neuronal. |
| Distribución del error | Presenta la distribución de los errores de predicción, permitiendo identificar la dispersión de las diferencias entre los valores reales y predichos. |
| Evolución de la pérdida (MSE) | Muestra el comportamiento del Error Cuadrático Medio durante el entrenamiento y la validación del modelo. |
| Curva MAE | Presenta la evolución del Error Absoluto Medio durante el proceso de entrenamiento y validación. |
| Visualizador de gráficas | Área destinada a mostrar las gráficas generadas por el modelo para facilitar el análisis de resultados. |

La pestaña Gráficas de resultados permite visualizar de forma gráfica el desempeño 
de la red neuronal desarrollada. Mediante estas representaciones es posible 
analizar el comportamiento del modelo durante el entrenamiento, evaluar la 
precisión de las predicciones realizadas y verificar su capacidad de generalización. 
La gráfica de energía real vs. energía predicha permite identificar el grado de 
concordancia entre los valores obtenidos mediante simulación y los estimados por 
el modelo. La distribución del error facilita el análisis de la dispersión de las 
predicciones, mientras que las curvas de MSE y MAE muestran la evolución del 
aprendizaje de la red neuronal a lo largo de las épocas de entrenamiento. 
Estas herramientas proporcionan al usuario una visión integral del desempeño del 
modelo y contribuyen a una mejor interpretación de los resultados obtenidos.
