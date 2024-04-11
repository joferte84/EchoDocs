
# 📚🗨️  EchoDocs: Convierte tus documentos en conversaciones con chatGPT

![Interfaz de EchoDocs](EchoDocs.jpg)


Descubre cómo ChatGPT puede dar vida a tus archivos PDF, ofreciéndote respuestas interactivas a cualquier pregunta que tengas sobre el contenido de tus documentos.


## Introduccion
📄🧠 EchoDocs revela cómo, en menos de 70 líneas de código, puedes desbloquear el poder de ChatGPT para convertir documentos estáticos en conversaciones dinámicas. Olvídate de las largas horas de lectura o de buscar información manualmente; ahora, simplemente pregunta a tus documentos y recibe respuestas instantáneas proporcionadas por ChatGPT.
Para construir esta aplicación de forma rápida y eficiente, utilizaremos:
* ChatGPT API
* Streamlit


## ¿Cómo funciona?
EchoDocs revoluciona la manera en que interactuamos con documentos PDF, ofreciendo un método interactivo y dinámico para explorar y comprender su contenido. Aquí te explicamos el proceso mejorado:
1. Carga y Gestión de Múltiples Documentos: Sube varios documentos PDF simultáneamente a EchoDocs. Nuestra plataforma gestiona eficientemente múltiples archivos, permitiéndote acceder a un vasto conocimiento almacenado en ellos con facilidad.
2. División Inteligente del Documento: Cada documento se divide en fragmentos o "chunks" manejables. Esta división permite un análisis detallado y una mejor comprensión del contenido por parte de nuestra IA.
3. Generación de Embeddings de Alta Precisión: Utilizamos modelos de embeddings de última generación para convertir los fragmentos de texto en representaciones vectoriales. Esto facilita la búsqueda semántica avanzada y mejora la precisión de las respuestas.
4. Almacenamiento Eficiente en Base de Conocimientos: Los fragmentos y sus embeddings correspondientes se almacenan en una base de conocimientos optimizada. Esto nos permite realizar búsquedas rápidas y eficientes, identificando los fragmentos más relevantes para cada pregunta.
5. Recuperación y Respuesta Inteligente: Cuando formulas una pregunta, nuestro sistema identifica los fragmentos más relevantes basándose en los embeddings y los envía junto con tu pregunta a ChatGPT. Así, generamos respuestas precisas y contextualizadas basadas directamente en el contenido de tus documentos.

6. Historial Interactivo de Preguntas y Respuestas: Todas tus interacciones se guardan en un historial accesible, permitiéndote revisar preguntas y respuestas anteriores. Esto convierte a EchoDocs en una herramienta de aprendizaje y consulta aún más valiosa.


## Instalar preguntaDOC
¡Usar EchoDocs es fácil! Aquí están los pasos:
1. Clone o descargue el repositorio en su máquina local.
2. Instale las dependencias requeridas ejecutando el siguiente comando en su terminal:
```console
pip install -r requirements.txt
```
3. Inicia la aplicación con el siguiente comando:
```console
streamlit run app.py
```
4. Consigue una clave API de OpenAI para utilizar su API de ChatGPT.
5. Suba sus documento a la aplicación.
6. Escriba su pregunta y disfrute de la magia.
