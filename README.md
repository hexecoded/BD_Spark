# Big Data II: Práctica final de Spark
![License](https://img.shields.io/badge/license-MIT-orange.svg?style=for-the-badge)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Apache Spark](https://img.shields.io/badge/Apache%20Spark-FDEE21?style=for-the-badge&logo=apachespark&logoColor=black)   

## Descripción
El objetivo de esta práctica es resolver un problema de aprendizaje automático con big data usando Spark. Para ello, usaremos un conjunto de datos basado en datos procedentes de listas de Spotify, donde podemos encontrar información variada acerca de canciones y sus metadatos: autor, año de salida, género, y variables numéricas asociadas a su bailabilidad, felicidad, y otros aspectos emocionales de la misma.

Se trata de un trabajo realizado de manera colaborativa en un grupo de 4 personas, donde estudiaremos diferentes alternativas distribuidas de paralelizar un algoritmo de clústering basado en densidad: DBScan.

## Estructura

El proyecto constará de una serie de apartados donde evaluaremos diferentes enfoques del algoritmo y analizaremos su escalabilidad:
- Contexto 
- Objetivos 
- Descripción detallada de la soluciónes propuestas
  - Breve discusión de las decisiones clave que hacen que tu algoritmo sea escalable
- Experimentos
  - Estudio breve de escalabilidad (scale-up, size-up, speedup)
  - Resultados acorde a métricas relevantes
- Discusión de resultados
- Resumen de la contribución de cada miembro del equipo


El resto de material se encuentra en formato zip en la raíz del directorio de trabajo.

## Entorno de ejecución

Para ejecutar la práctica, se ha creado un entorno específico de Conda para que las ejecuciones puedan realizarse sin problemas de compatbilidad. Esta se encuentra en el archivo requirements.yml, y puede ser instalado con un breve comando:
```console
 conda env create -f environment.yml
```
También es importante seleccionar al inicio del notebook, cuando se configura Spark, la cantidad adecuada de RAM y cores de acorde a la máquina donde se va a ejecutar.
Tras esto, el código puede ser ejecutado sin problemas, siempre y cuando se disponga al menos de JDK 17 de Java.
