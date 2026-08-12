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
