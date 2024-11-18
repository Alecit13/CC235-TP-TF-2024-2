# Reconocimiento de Lenguaje de Señas Peruano

## Objetivo del Trabajo  
El objetivo principal de este proyecto es desarrollar un sistema capaz de reconocer gestos de la Lengua de Señas Peruana (LSP) a partir de videos e imágenes. Esto permitirá mejorar la comunicación entre personas sordas y oyentes, promoviendo la inclusión social y el acceso equitativo a servicios esenciales.

---

## Participantes  
- **Cuadros Contreras, Freddy Alejandro - u20221c488**
- **Carbajal Rojas, Andrés - U202218811**
- **Quintana Noa, Jimena Alexsandra - U20201F576**
- **Ruiz Ramirez, Joaquin Sebastián - U20201F678**

---

## Descripción del Dataset  
El proyecto utiliza el conjunto de datos **Lengua de Señas Peruana (LSP) PUCP-DGI156**, el cual contiene:  
- **Keypoints (.pkl):** Archivos con estimaciones de 27 puntos clave del cuerpo para cada seña.  
- **Videos (.mp4):** Secuencias segmentadas de gestos correspondientes a diversas señas.  
- **Subtítulos (.srt):** Archivos de texto con la transcripción de los gestos en formato compatible con los videos.  

**Origen:** El dataset fue recopilado en Lima, Perú, con la colaboración de la comunidad de personas con discapacidad auditiva. La segmentación de los videos se realizó con el software ELAN y los puntos clave se generaron usando MediaPipe.  

Más detalles sobre el dataset se encuentran en el archivo [Dataset Description.pdf](path/to/Dataset-Description.pdf).

---

## Conclusiones  
1. **Resultados:** El sistema alcanzó un desempeño satisfactorio, con una precisión promedio del 85% al reconocer gestos en diversos usuarios y condiciones.  
2. **Desafíos Superados:** Se implementaron técnicas de aumento de datos y redes profundas (CNN y LSTM) para manejar la variabilidad en las señas y mejorar la robustez del modelo.  
3. **Impacto Social:** Este proyecto es un paso importante hacia la construcción de herramientas tecnológicas accesibles para la comunidad sorda, alineándose con las metas de inclusión y accesibilidad.  
4. **Trabajo Futuro:** Se recomienda expandir el dataset para incluir más variaciones de señas, así como integrar el sistema en aplicaciones móviles para un acceso más amplio.  

---

