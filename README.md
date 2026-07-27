# Mesa de Ayuda IA — Recursos Humanos, Patito S.A.

Prototipo de mesa de ayuda con agentes especializados (LangChain + Google Gemini) para el
departamento de Recursos Humanos de Patito S.A. Un orquestador recibe la pregunta del
usuario, decide qué agente(s) deben responder, y consolida una respuesta final trazable,
basada únicamente en la base documental entregada.

---

## Tabla de contenido

1. [Arquitectura](#1-arquitectura)
2. [Flujo de la solución](#2-flujo-de-la-solución)
3. [Instrucciones de ejecución](#3-instrucciones-de-ejecución)
4. [Decisiones técnicas](#4-decisiones-técnicas)
5. [Ejemplos de preguntas y respuestas](#5-ejemplos-de-preguntas-y-respuestas)
6. [Riesgos y mejoras futuras](#6-riesgos-y-mejoras-futuras)
7. [Estructura del repositorio](#7-estructura-del-repositorio)

---

## 1. Arquitectura

La solución está compuesta por **5 agentes especializados** + **1 agente orquestador**,
todos construidos con LangChain y usando Google Gemini como LLM.

```
                        ┌─────────────────────────┐
                        │   Usuario (Streamlit)   │
                        └────────────┬────────────┘
                                     │ pregunta (+ imagen opcional)
                                     ▼
                        ┌─────────────────────────┐
                        │   AGENTE ORQUESTADOR    │
                        │  (ChatGoogleGenerativeAI │
                        │   + create_agent)        │
                        └────────────┬────────────┘
                                     │ decide qué tool(s) invocar
          ┌──────────────┬───────────┼───────────┬──────────────┐
          ▼              ▼           ▼           ▼              ▼
  ┌───────────────┐┌───────────┐┌───────────┐┌───────────┐┌──────────────┐
  │  Agente de    ││ Agente de ││ Agente de ││  Agente   ││   Agente     │
  │  Beneficios   ││ Políticas ││Reclutamie-││Multimodal ││  de Acción   │
  │               ││ Internas  ││nto/Onboard││ (Imagen)  ││  (Registro)  │
  ├───────────────┤├───────────┤├───────────┤├───────────┤├──────────────┤
  │ Retriever      ││ Retriever ││ Retriever ││ Gemini    ││ Tool de      │
  │ Chroma          ││ Chroma    ││ Chroma    ││ Vision    ││ escritura en │
  │ (doc.1)         ││ (doc.2)   ││ (doc.3)   ││           ││ .txt         │
  └───────────────┘└───────────┘└───────────┘└───────────┘└──────────────┘
```

**Agentes de lectura (RAG)** — cada uno con su propia base de conocimiento embebida
(un índice vectorial independiente):

| Agente | Base de conocimiento | Responde sobre |
|---|---|---|
| Beneficios y Compensaciones | Manual de Beneficios y Compensaciones | seguro médico, dependientes, bonos, compensación |
| Políticas Internas | Reglamento Interno de Trabajo y Código de Conducta | vacaciones, permisos, jornada, código de conducta |
| Reclutamiento y Onboarding | Guía de Reclutamiento, Referidos y Onboarding | selección, referidos, onboarding |

**Agentes adicionales** (se implementaron ambos, según lo permitido por el enunciado):

| Agente | Capacidad |
|---|---|
| Multimodal de Imagen | Analiza imágenes de formularios de RR.HH. (visión de Gemini), valida si están completos y extrae sus datos |
| De Acción (Registro) | Valida datos obligatorios y registra solicitudes de vacaciones / inscripción de dependiente en `registro_solicitudes_rrhh.txt` |

**Orquestador**: agente LangChain (`create_agent` + `langgraph.checkpoint.memory.InMemorySaver`
para memoria multi-turno) cuyas *tools* son, en realidad, cada uno de los 5 agentes anteriores.
El propio LLM (Gemini) decide, según el `system_prompt`, cuál o cuáles invocar.

---

## 2. Flujo de la solución

1. El usuario escribe una pregunta en lenguaje natural (y opcionalmente adjunta una imagen).
2. El orquestador recibe la pregunta y **clasifica la intención**: beneficios, políticas
   internas, reclutamiento, imagen, acción, o una combinación de varias.
3. El orquestador **invoca la(s) tool(s)** correspondientes (uno o varios agentes a la vez
   si la pregunta es mixta).
4. Cada agente de lectura consulta **su propio retriever** (vector store dedicado) y genera
   una respuesta basada únicamente en los fragmentos recuperados.
5. Si hay una imagen, el Agente Multimodal la procesa con la capacidad de visión de Gemini.
6. Si se pide un registro, el Agente de Acción valida los datos obligatorios, pide los que
   falten, pide confirmación explícita y solo entonces escribe en el archivo `.txt`.
7. El orquestador **consolida** todas las respuestas parciales en una sola respuesta final,
   indicando qué agente(s) se usaron (trazabilidad).
8. Si ninguna tool encuentra información relevante, el sistema responde explícitamente:
   *"No encontré información suficiente en la base documental proporcionada."*

---

## 3. Instrucciones de ejecución

Esta guía asume que no tienes nada instalado todavía y explica cada paso en detalle.
Los ejemplos de comandos son para **Windows (CMD/PowerShell)**; en Mac/Linux los comandos
`pip`, `python` y `streamlit` son iguales, solo cambia `cd` a la ruta correspondiente.

### Requisitos previos
- **Python 3.10 o superior** instalado ([descarga aquí](https://www.python.org/downloads/)).
  Verifica que lo tienes abriendo una terminal y escribiendo `python --version`.
- **Git** instalado ([descarga aquí](https://git-scm.com/downloads)), solo si vas a clonar
  el repositorio en vez de descargarlo como ZIP.
- Una **API key de Google Gemini**, gratuita, obtenida en
  [Google AI Studio](https://aistudio.google.com/apikey) (botón "Create API key").

### 3.1 Obtener el proyecto en tu computadora

**Opción A — Clonando con Git** (recomendado):
```bash
git clone https://github.com/geomarin12/Proyecto-Semillero-IA--TELEMATICOS-S.A.git
cd Proyecto-Semillero-IA--TELEMATICOS-S.A
```

**Opción B — Descargando el ZIP**: en la página del repositorio en GitHub, botón verde
**"Code" → "Download ZIP"**, descomprime la carpeta, y abre una terminal dentro de ella.

### 3.2 Instalar las dependencias del proyecto

Dentro de la carpeta del proyecto (verifica con `dir` que veas ahí `app.py` y `backend.py`),
ejecuta:
```bash
pip install -r requirements.txt
```
Esto instala automáticamente todo lo necesario (LangChain, Streamlit, Chroma, etc.). Puede
tardar uno o dos minutos la primera vez.

### 3.3 Configurar tu API key (paso obligatorio)

El proyecto necesita tu propia API key de Gemini guardada en un archivo llamado `.env`
(este archivo **no viene incluido** en el repositorio por seguridad — cada persona debe
crear el suyo con su propia key).

1. En la carpeta del proyecto, crea el archivo escribiendo en la terminal:
   ```bash
   notepad .env
   ```
   Windows te preguntará si quieres crear el archivo — acepta que sí.
2. Dentro del Bloc de notas que se abre, escribe esta línea (reemplazando por tu key real,
   sin espacios ni comillas):
   ```
   GOOGLE_API_KEY=tu_key_real_aqui
   ```
3. Guarda (`Ctrl+S`) y cierra el Bloc de notas.
4. Verifica que el archivo quedó bien creado (debe llamarse exactamente `.env`, sin `.txt`
   al final):
   ```bash
   dir /a
   ```

⚠️ **Nunca compartas ni subas tu archivo `.env` a GitHub ni a nadie** — contiene tu
credencial personal. El archivo `.env.example` (sí incluido en el repositorio) es solo
una plantilla de referencia, sin ninguna key real.

### 3.4 Ejecutar el notebook (opcional — desarrollo y pruebas paso a paso)

Abre `proyecto.ipynb` en Jupyter y ejecuta todas las celdas en orden (de arriba hacia
abajo). Ahí se construyen y prueban los 5 agentes y el orquestador paso a paso, incluida
la consulta mixta. Cuando una celda te pida la API key de forma interactiva, pégala ahí
(no queda guardada en el archivo).

### 3.5 Ejecutar la interfaz web (Streamlit) — forma principal de usar la solución

La interfaz web consume el mismo código, consolidado en `backend.py`.

**Importante:** Streamlit **debe ejecutarse desde una terminal** (CMD, PowerShell o la
terminal de tu editor), **no** desde una celda de Jupyter ni haciendo doble clic en el
archivo — si lo intentas desde Jupyter, verás advertencias como
`missing ScriptRunContext` y la página no cargará nada.

```bash
streamlit run app.py
```

En la terminal debería aparecer algo como:
```
Local URL: http://localhost:8501
```
El navegador se abre solo con esa dirección; si no ocurre, cópiala y pégala manualmente
en tu navegador.

Ahí puedes:
- Escribir preguntas en un chat.
- Adjuntar una imagen (barra lateral) para probar el agente multimodal.
- Ver qué agentes participaron en cada respuesta (trazabilidad).

Para **detener** la aplicación, vuelve a la terminal donde la lanzaste y presiona `Ctrl+C`.

### 3.6 Preguntas de prueba sugeridas

Para verificar rápidamente que todo funciona, puedes probar estas preguntas en la interfaz:

1. `¿Cuántos días de vacaciones me corresponden al año?`
2. `Voy a tomar mis vacaciones y además quiero agregar a mi pareja al seguro médico. ¿Cuántos días me corresponden, cómo los solicito y qué necesito para inscribir a un dependiente?` (consulta mixta)
3. Adjunta la imagen `formulario_dependiente.png` incluida en el repositorio y pregunta: `¿está completo este formulario y qué datos faltan?`
4. `Registra una solicitud de vacaciones para Juan Pérez.` (con datos incompletos a propósito, para ver que el sistema los solicita)
5. `¿Cuál es el precio de las acciones de Patito S.A. en la bolsa?` (fuera de alcance, para confirmar que responde que no tiene información suficiente)

---

## 4. Decisiones técnicas

| Decisión | Elección | Justificación |
|---|---|---|
| LLM | `gemini-flash-latest` (alias) | Se usa el alias en vez de fijar una versión porque Google descontinuó `gemini-2.0-flash` (shutdown 1 jun 2026) y restringió `gemini-2.5-flash` a cuentas nuevas durante el desarrollo de este proyecto; el alias asegura continuidad sin tener que actualizar el código cada vez que Google retira un modelo. |
| Embeddings | `models/gemini-embedding-001` | `text-embedding-004` fue descontinuado por Google (nov. 2025); `gemini-embedding-001` es el modelo de embeddings vigente y soportado por `GoogleGenerativeAIEmbeddings`. |
| Vector store | Chroma | Simplicidad de uso, persistencia local en disco, sin necesidad de infraestructura externa — adecuado para un prototipo. Cada agente tiene su propia colección/índice independiente. |
| Chunking | Por secciones temáticas de cada documento | Los documentos ficticios están organizados por temas; dividir por sección temática (en vez de por tamaño fijo de caracteres) preserva el contexto semántico completo de cada bloque de información. |
| Orquestación | `langchain.agents.create_agent` + tools que envuelven a cada subagente | Permite que el propio LLM decida, en lenguaje natural, qué agente(s) invocar según el `system_prompt`, sin necesidad de reglas de enrutamiento hardcodeadas. |
| Memoria | `InMemorySaver` (checkpointer de LangGraph) | Permite conversaciones multi-turno (ej. el usuario completa datos de un registro que quedó pendiente en el turno anterior). |
| Interfaz | Streamlit | Interfaz web simple, 100% Python, sin necesidad de HTML/CSS/JS, con soporte nativo para chat y carga de imágenes — ideal para un prototipo de este alcance. |
| Agente de Acción — control de duplicados | Comparación de una "firma" de datos contra el archivo de registro | Evita registrar dos veces la misma solicitud exacta sin necesidad de una base de datos externa. |
| Agente de Acción — confirmación previa | El registro solo ocurre con `confirmar=True` explícito | Cumple el requisito de no ejecutar la acción (efecto real sobre un archivo) sin que el usuario confirme conscientemente. |

---

## 5. Ejemplos de preguntas y respuestas

| # | Pregunta | Agente(s) esperado(s) |
|---|---|---|
| 1 | "¿Cuántos días de vacaciones me corresponden al año?" | Políticas Internas |
| 2 | "¿Qué cubre el seguro médico corporativo y cómo agrego a un familiar como dependiente?" | Beneficios |
| 3 | "¿Cómo funciona el programa de referidos?" | Reclutamiento |
| 4 | **(Mixta)** "Voy a tomar mis vacaciones y además quiero agregar a mi pareja al seguro médico. ¿Cuántos días me corresponden, cómo los solicito y qué necesito para inscribir a un dependiente?" | Políticas Internas **+** Beneficios (consolidado) |
| 5 | "Adjunto el formulario en formulario_dependiente.png: ¿está completo y qué datos faltan?" | Multimodal |
| 6 | "Registra una solicitud de vacaciones para Juan Perez." (datos incompletos) | Acción → solicita los datos faltantes |
| 7 | "¿Cuál es el precio de las acciones de Patito S.A.?" (fuera de alcance) | Ninguno → "No encontré información suficiente en la base documental proporcionada." |

Estas pruebas están reproducidas en la sección final del notebook (`proyectofinal.ipynb`)
y pueden ejecutarse también desde la interfaz Streamlit.

---

## 6. Riesgos y mejoras futuras

### Riesgos identificados
- **Disponibilidad y cuota de la API de Gemini**: el servicio depende 100% de la
  disponibilidad y cuota de la cuenta de Google; se observaron bloqueos de cuota (código
  429) y descontinuación de modelos (`gemini-2.0-flash`, `text-embedding-004`) durante el
  propio desarrollo del proyecto.
- **Alucinaciones**: mitigado parcialmente por el patrón RAG estricto (el LLM solo responde
  con los fragmentos recuperados), pero no es una garantía absoluta.
- **Sin autenticación de usuarios**: cualquiera que acceda a la interfaz puede consultar o
  registrar solicitudes; no hay control de identidad ni de permisos por colaborador.
- **Vector store local (Chroma)**: no está pensado para escalar a un volumen alto de
  documentos ni para un entorno multi-usuario concurrente en producción.
- **El registro de solicitudes es un archivo `.txt` plano**: no es apto para producción
  (sin control de concurrencia, sin backups, sin auditoría real).

### Mejoras futuras propuestas
- Migrar el registro de solicitudes de un `.txt` a una base de datos real (con
  transacciones y control de concurrencia).
- Agregar autenticación de usuarios y control de permisos por documento/agente
  (ej. que un colaborador no pueda registrar solicitudes a nombre de otro).
- Migrar el vector store a una solución más escalable (pgvector) si el volumen de
  documentos crece.
- Agregar monitoreo de: costos (tokens consumidos por Gemini), latencia por consulta,
  tasa de errores y feedback de los usuarios sobre la calidad de las respuestas.
- Agregar reintentos automáticos con backoff ante errores `429 RESOURCE_EXHAUSTED` de
  Gemini, en vez de solo mostrar el error al usuario.
- Versionar los documentos de la base de conocimiento y permitir actualizarlos sin
  tener que regenerar manualmente todo el índice vectorial.

---

## 7. Estructura del repositorio

```
5_RRHH/
├── proyectofinal.ipynb          # Notebook de desarrollo y pruebas (los 5 agentes + orquestador)
├── backend.py                   # Lógica consolidada (agentes + orquestador) como módulo Python
├── app.py                       # Interfaz web (Streamlit)
├── requirements.txt             # Dependencias del proyecto
├── .env.example                 # Plantilla de variables de entorno (sin credenciales reales)
├── documentos/
│   ├── beneficios.txt           # Documento 1: Manual de Beneficios y Compensaciones
│   ├── reglamento.txt           # Documento 2: Reglamento Interno y Código de Conducta
│   └── reclutamiento.txt        # Documento 3: Guía de Reclutamiento, Referidos y Onboarding
├── formulario_dependiente.png   # Imagen de prueba para el agente multimodal
├── registro_solicitudes_rrhh.txt # Archivo de registro generado por el agente de acción
└── README.md
```
