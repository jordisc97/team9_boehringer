# 🚀 Asistente de Diagnóstico para Fibrosis Pulmonar Idiopática

Haz clic en la imagen a continuación para ver una **demostración** de la aplicación en YouTube:

[![Ver la demostración en YouTube](https://img.youtube.com/vi/kS4T7r7cydg/0.jpg)](https://www.youtube.com/watch?v=kS4T7r7cydg)

## 📜 **Descripción del Proyecto**
Los médicos clínicos enfrentan una alta carga de trabajo al tratar de identificar y manejar pacientes con posibles enfermedades pulmonares como la **Fibrosis Pulmonar Idiopática (FPI)**. Este proyecto tiene como objetivo proporcionar una herramienta basada en **data-driven decision-making** que los ayude a tomar decisiones más ágiles y fundamentadas. La herramienta puede analizar datos clínicos e imágenes para determinar si un paciente debería:

- Ser seguido más de cerca.
- Derivado a un especialista.
- Continuar con el seguimiento habitual.

Al utilizar esta solución, se busca optimizar el tiempo de diagnostico, priorizar los casos más urgentes y mejorar el manejo clínico de los pacientes.

## 🎯 **Objetivo del Proyecto**
- Asistir a los médicos en la priorización de pacientes mediante análisis de datos clínicos y de imágenes.
- Identificar patrones sutiles que indiquen mayor riesgo de FPI.
- Generar recomendaciones basadas en el riesgo para apoyar la toma de decisiones clínicas.
- Mejorar la eficiencia y precisión en la identificación de pacientes que necesitan seguimiento intensivo o derivación.

## 🛠️ **Enfoque y Solución**
La solución implementada es una aplicación web construida con **Gradio** que integra modelos de aprendizaje profundo y machine learning para analizar imágenes y datos clínicos del paciente.

### 🧩 **Componentes Clave**
- **Modelos de Visión por Computadora Pre-entrenados**: Se utilizaron modelos como ResNet34, SqueezeNet y DenseNet121 para extraer características de las imágenes de tomografías computarizadas.
- **Modelo LightGBM**: Un modelo de LightGBM se entrenó utilizando las probabilidades de los modelos de visión y características clínicas para predecir la probabilidad de FPI.
- **Llama 3 Model**: Modelo LLM que se usa para agrupar todos los outputs en un mensaje en lenguaja natural, para que paciente y medico se entiendan.
- **Generación de Animaciones**: Se creó la clase `AnimateScans` para generar animaciones GIF de las tomografías, facilitando la visualización dinámica de las imágenes.

## 📂 **Estructura del Proyecto**
- `gradio_app.py`: Script principal que ejecuta la aplicación Gradio.
- `utils.py`: Módulo que contiene la clase `AnimateScans` para procesar las imágenes DICOM y generar animaciones.
- `models/`: Carpeta que contiene los modelos pre-entrenados y el modelo LightGBM.
- `data/`: Carpeta con datos de muestra de pacientes.
- `dicom/`: Directorio con las imágenes DICOM de los pacientes, organizadas por ID de paciente.
- `animations/`: Carpeta donde se guardan las animaciones GIF generadas.

## ⚙️ **Instrucciones de Instalación y Uso**

### 📋 **Requisitos Previos**
- Python 3.8 o superior.
- Conda o Miniconda instalado en el sistema.

### 📥 **Instalación**
Clonar el repositorio:
```bash
git clone https://github.com/jordisc97/team9_boehringer.git
cd team9_boehringer
```
Crear y activar el entorno conda:
```bash
conda env create -f environment.yml
conda activate fibrosis_pulmonar_env
```
Verificar los modelos:
- Los modelos pre-entrenados ya están incluidos en la carpeta `models/` del repositorio. Asegúrate de que los siguientes archivos estén disponibles:
  - `resnet34_model.pkl`
  - `squeezenet1_0_model.pkl`
  - `densenet121_model.pkl`
  - `lgbm_model.txt`

Descargar las carpetas con archivos DICOM y datos:
1. Descarga las carpetas necesarias desde Google Drive utilizando este enlace:  
   [Archivos DICOM de muestra](https://drive.google.com/drive/folders/1ZXwMteDDFa1I9ihyeYkD70Urql4fUho5?usp=sharing)
2. Coloca las carpetas descargadas en la estructura indicada:
   - Las imágenes DICOM deben estar en la carpeta `dicom/`, con un subdirectorio para cada paciente basado en su ID (e.g., `dicom/ID000123456789/`).
   - El archivo de datos `sample_patient_data.csv` debe estar en la carpeta `data/`.

### ▶️ **Ejecución de la Aplicación**
Ejecuta el siguiente comando en la raíz del proyecto para iniciar la aplicación:
```bash
python gradio_app.py
```
La aplicación se ejecutará localmente y estará disponible en http://localhost:7860.

## 🖥️ **Uso de la Aplicación**
1. **Seleccionar ID de Paciente**: En el menú desplegable, elige el ID del paciente cuyas imágenes y datos deseas analizar.
2. **Ingresar Datos Clínicos**: Proporciona la información requerida como edad, FVC, sexo, estado de tabaquismo y semana base.
3. **Obtener Resultados**: Haz clic en el botón para generar la predicción. La aplicación mostrará:
   - La probabilidad de que el paciente tenga FPI.
   - Un mensaje de riesgo indicando si se recomienda un monitoreo más frecuente.
   - Una animación GIF de las tomografías computarizadas del paciente.

## 🧠 **Decisiones Técnicas Clave**
- **Integración de Modelos de Visión y Datos Clínicos**: Combinar modelos de visión por computadora con datos clínicos en un modelo LightGBM permite aprovechar múltiples fuentes de información para mejorar la precisión diagnóstica.
- **Uso de Gradio para la Interfaz de Usuario**: Gradio facilita la creación de interfaces web interactivas para modelos de machine learning sin necesidad de desarrollar una aplicación web desde cero.
- **Procesamiento de Imágenes DICOM**: Implementar la clase `AnimateScans` para manejar imágenes DICOM y generar animaciones mejora la interpretación visual de las tomografías por parte de los radiólogos.
- **Estructura Modular del Código**: Separar el código en módulos (`app.py` y `utils.py`) mejora la mantenibilidad y claridad del proyecto.
