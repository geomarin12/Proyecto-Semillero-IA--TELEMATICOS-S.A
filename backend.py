
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
load_dotenv()  
 
if not os.environ.get("GOOGLE_API_KEY"):
    raise RuntimeError(
        "Falta GOOGLE_API_KEY. Crea un archivo .env con GOOGLE_API_KEY=  "
        "(usa .env.example como plantilla)."
    )
 
MODELO_LLM = "gemini-flash-lite-latest"
MODELO_EMBEDDING = "models/gemini-embedding-001"
llm = ChatGoogleGenerativeAI(model=MODELO_LLM, temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model=MODELO_EMBEDDING)
#-- los agentes racks 
from pathlib import Path 

# 2. creamos las variables de documentos y politicas y ponemos su ruta ademas si no esta el doc cargado , lo escribimos 
DOC_PATH = "01_Beneficios_Compensaciones.txt"
POLITICA = """PATITO S.A.
MANUAL DE BENEFICIOS Y COMPENSACIONES
(Documento ficticio para fines de evaluación del semillero)
Base de conocimiento del Agente de Beneficios y Compensaciones

1. SEGURO MÉDICO CORPORATIVO
1.1 Cobertura: consultas médicas, hospitalización, emergencias, exámenes de laboratorio y
    medicamentos según el plan. Incluye atención ambulatoria y cobertura dental básica.
1.2 Dependientes: el colaborador puede inscribir a cónyuge o pareja e hijos.
1.3 Cómo agregar un dependiente:
    - Completar el formulario de inscripción de dependientes en el portal de RR. HH.
    - Adjuntar el documento que acredite el vínculo (acta de matrimonio/unión o partida de
      nacimiento) y copia del documento de identidad del dependiente.
    - Enviar la solicitud dentro de los primeros 30 días desde el ingreso o desde el evento
      (matrimonio, nacimiento). Fuera de ese plazo, se espera al periodo de inscripción anual.

2. BONOS
- Bono por desempeño anual según evaluación.
- Bono por cumplimiento de metas del área (cuando aplique).

3. OTROS BENEFICIOS
- Día libre de cumpleaños.
- Capacitación y apoyo educativo.
- Modalidad híbrida según el puesto.

4. COMPENSACIÓN
La estructura salarial considera el rol, la banda salarial y el mercado. Las revisiones
salariales se realizan una vez al año."""

# 3.Guardar la instancia path en la variable doc_path_obj
doc_path_obj = Path(DOC_PATH)

# 4. Verificar si el documento existe o crearlo si no está presente
if not doc_path_obj.exists():
    doc_path_obj.write_text(POLITICA, encoding="utf-8")
    print("docpack creado")
else:
    print("El archivo ya existe. Procediendo a la lectura...")

# 5.Utilizar el comando .read_text() para leer el doc y guardarlo en la variable politica
politica = doc_path_obj.read_text(encoding="utf-8")

# 6. Impresión de métricas requeridas de los primeros 400 caracteres
print(f"Caracteres totales: {len(politica)}")
print(" " * 60)
print(politica[:400])

# embeding del primer agente 
import re
from langchain_community.vectorstores import Chroma

def chunkear_por_tema(texto):
    """Divide el manual en un chunk por cada tema numerado (1., 2., 3., etc.)."""
    
    # se crea la variable cabecera donde se guardara la lista de los temas buscados por segmentos 
    cabeceras = list(re.finditer(r"^\d+\.\s+[A-ZÁÉÍÓÚÑ]", texto, flags=re.MULTILINE))

    chunks = []

    for i, m in enumerate(cabeceras):
        ini = m.start()
        # Si no es el último tema, el final es donde empieza el siguiente. Si es el último, va hasta el fin del texto.
        fin = cabeceras[i + 1].start() if i + 1 < len(cabeceras) else len(texto)

        # Extraemos la sección correspondiente y limpiamos espacios en blanco innecesarios
        parte = texto[ini:fin].strip()
        chunks.append(parte)

    return chunks

# 1. Obtenemos los chunks del manual de beneficios y compensaciones y la guardamos en la variable chunk_beneficios 

chunks_beneficios = chunkear_por_tema(politica)

# 2. en la varible chunks adicionales se guardan documentos auxiliares, si llegaran a existir 
chunks_adicionales = [] 

# Unimos los documentos procesados
chunks = chunks_beneficios + chunks_adicionales

# 3. Imprimir métricas de verificación
print(f"Total de chunks creados: {len(chunks)}")
print("=" * 60)

for i, c in enumerate(chunks):
    # Reemplazamos los saltos de línea internos por espacios solo para la previsualización del print
    vista_previa = c.replace('\n', ' ')
    print(f"Chunk {i+1}: {vista_previa[:70]}...")
    
# Cada chunk se embebe con Gemini y se guarda en Chroma
# 1. Creamos la lista de metadatos dinámicamente para cada chunk
metadatos_chunks = [{"seccion": i + 1, "fuente": DOC_PATH} for i in range(len(chunks))]

# 2. Inicializamos el vectorstore en Chroma con la configuración e inserción de datos corregida
vectorstore = Chroma.from_texts(
    texts=chunks,
    embedding=embeddings,
    metadatas=metadatos_chunks,
    collection_name="beneficios_y_compensaciones"
)

# 3. Configuramos retriever para extraer los 2 mejores resultados (k=2)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# rack del primer agente 
PROMPT_CONOCIMIENTO = """Eres el asistente de Recursos Humanos de Patito S.A. Respondes sobre el manual de beneficios y compensaciones.
Reglas estrictas:
- Responde ÚNICAMENTE con base en el CONTEXTO entregado.
- Cita el número de sección cuando sea posible.
- Si la información no está en el contexto, responde exactamente: "No tengo esa información en la política."
- Se breve y directo. No inventes datos."""
def responder_politica(pregunta: str) -> str:
    """Pipeline RAG: recupera contexto de la base de conocimiento y genera la respuesta."""
    docs = retriever.invoke(pregunta)
    contexto = "\n\n---\n\n".join([d.page_content for d in docs])
    
    msg = llm.invoke([
        {"role": "system", "content": PROMPT_CONOCIMIENTO},
        {"role": "user", "content": f"CONTEXTO:\n{contexto}\n\nPREGUNTA: {pregunta}"}
    ])
    contenido = msg.content
    if isinstance(contenido, str):
        return contenido
    if isinstance(contenido, list):
        return "".join(
            bloque.get("text", "")
            for bloque in contenido
            if isinstance(bloque, dict) and bloque.get("type") == "text"
        )
    return str(contenido)

#-- agente 2 

from pathlib import Path 
# 1. Definimos la ruta del archivo y el texto que va a contener
DOC_PATH_REGLAMENTO = "02_Reglamento_Interno.txt"
POLITICA_reglamento = """PATITO S.A.
REGLAMENTO INTERNO DE TRABAJO Y CÓDIGO DE CONDUCTA
(Documento ficticio para fines de evaluación del semillero)
Base de conocimiento del Agente de Políticas Internas

1. JORNADA LABORAL
Jornada de 40 horas semanales. Horario estándar de 8:00 a 17:00 con una hora de almuerzo,
salvo acuerdos de horario flexible o trabajo híbrido.

2. VACACIONES
- Cada colaborador tiene derecho a 15 días hábiles de vacaciones por año cumplido.
- Las vacaciones se solicitan a través del portal de RR. HH. con al menos 15 días de
  anticipación y deben ser aprobadas por el jefe directo.
- Pueden tomarse de forma fraccionada según acuerdo con el área.
- Los días no usados se rigen por la política de acumulación (máximo de un periodo).

3. PERMISOS
- Permisos remunerados: por matrimonio, nacimiento, fallecimiento de familiar directo, según
  la ley y la política interna.
- Permiso no remunerado: se solicita por escrito a través del portal de RR. HH., indicando el
  motivo y el periodo; requiere aprobación del jefe directo y de RR. HH. El tiempo no
  remunerado no genera remuneración durante su duración.

4. CÓDIGO DE CONDUCTA
Respeto, no discriminación, ambiente libre de acoso, cuidado de los recursos de la empresa y
confidencialidad de la información.

5. FALTAS Y SANCIONES
Las faltas se clasifican en leves, graves y muy graves, con medidas que van desde la
amonestación hasta la terminación, según la gravedad y el debido proceso."""

# 2. Instanciamos el objeto Path con la ruta del archivo
doc_reglamento_obj = Path(DOC_PATH_REGLAMENTO)

# 3. GUARDADO DEL DOCUMENTO: Si el archivo no existe en la carpeta, se crea y se escribe el texto
if not doc_reglamento_obj.exists():
    doc_reglamento_obj.write_text(POLITICA_reglamento, encoding="utf-8")
    print(f"Documento {DOC_PATH_REGLAMENTO} creado.")
else:
    print(f"El archivo {DOC_PATH_REGLAMENTO} ya existe. ")

# 4. LECTURA DEL ARCHIVO: Se lee el texto del archivo físico hacia la variable
POLITICA_reglamento = doc_reglamento_obj.read_text(encoding="utf-8")

# 5. IMPRESIÓN DE MÉTRICAS: Total de caracteres y previsualización de los primeros 400
print(f"Caracteres totales: {len(POLITICA_reglamento)}")
print(" " * 60)
print(POLITICA_reglamento[:400])

# embeding del segundo agente 
import re
from langchain_community.vectorstores import Chroma

def chunkear_por_tema(texto):
    """Divide el manual en un chunk por cada tema numerado (1., 2., 3., etc.)."""
    cabeceras = list(re.finditer(r"^\d+\.\s+[A-ZÁÉÍÓÚÑ]", texto, flags=re.MULTILINE))
    chunks = []
    for i, m in enumerate(cabeceras):
        ini = m.start()
        fin = cabeceras[i + 1].start() if i + 1 < len(cabeceras) else len(texto)
        parte = texto[ini:fin].strip()
        chunks.append(parte)
    return chunks

# 1. Obtenemos los chunks del manual de reglamento interno
chunks_reglamento = chunkear_por_tema(POLITICA_reglamento)

# 2. Documentos auxiliares 
chunks_adicionales = []

# Unimos los documentos procesados
chunks_totales_reglamento = chunks_reglamento + chunks_adicionales

# 3. Imprimir métricas de verificación
print(f"Total de chunks creados: {len(chunks_totales_reglamento)}")
print("=" * 60)

for i, c in enumerate(chunks_totales_reglamento):
    vista_previa = c.replace('\n', ' ')
    print(f"Chunk {i+1}: {vista_previa[:70]}...")


# 4. Creamos la lista de metadatos dinámicamente para cada chunk
metadatos_reglamento = [
    {"seccion": i + 1, "fuente": DOC_PATH_REGLAMENTO} 
    for i in range(len(chunks_totales_reglamento))
]

# 5. Inicializamos el vectorstore en Chroma

vectorstore_reglamento = Chroma.from_texts(
    texts=chunks_totales_reglamento,
    embedding=embeddings,            
    metadatas=metadatos_reglamento,
    collection_name="reglamento_interno"
)

# 6. Configuramos retriever para extraer los 2 mejores resultados (k=2)
retriever_reglamento = vectorstore_reglamento.as_retriever(search_kwargs={"k": 2})

# rack del segundo agente 

PROMPT_CONOCIMIENTO_REGLAMENTO = """Eres el asistente de Recursos Humanos de Patito S.A. Respondes sobre el reglamento interno de trabajo y codigo de conducta.

Reglas estrictas:
- Responde ÚNICAMENTE con base en el CONTEXTO entregado.
- Cita el número de sección cuando sea posible.
- Si la información no está en el contexto, responde exactamente: "No tengo esa información en la política."
- Se breve y directo. No inventes datos."""
def responder_POLITICA_reglamento(pregunta: str) -> str:
    """Pipeline RAG: recupera contexto de la base de conocimiento de reglamento y genera la respuesta."""
    docs = retriever_reglamento.invoke(pregunta)
    contexto = "\n\n---\n\n".join([d.page_content for d in docs])  
    msg = llm.invoke([
        {"role": "system", "content": PROMPT_CONOCIMIENTO_REGLAMENTO},
        {"role": "user", "content": f"CONTEXTO:\n{contexto}\n\nPREGUNTA: {pregunta}"}
    ])  
    contenido = msg.content
    if isinstance(contenido, str):
        return contenido
    
    if isinstance(contenido, list):
        return "".join(
            bloque.get("text", "")
            for bloque in contenido
            if isinstance(bloque, dict) and bloque.get("type") == "text"
        )
    return str(contenido)

# agnete 3 
# 1.el primer paso es importar librerias 
from pathlib import Path 

# 2. creamos las variables de documentos y politicas y ponemos su ruta ademas si no esta el doc cargado , lo escribimos 
DOC_PATH_RECLUTAMIENTO = "03_Reclutamiento_Onboarding.txt"
RECLUTAMIENTO = """PATITO S.A.
GUÍA DE RECLUTAMIENTO, REFERIDOS Y ONBOARDING
(Documento ficticio para fines de evaluación del semillero)
Base de conocimiento del Agente de Reclutamiento y Onboarding

1. PROCESO DE SELECCIÓN
Requisición del área -> publicación de la vacante -> revisión de hojas de vida -> entrevistas
(RR. HH. y área solicitante) -> evaluación técnica -> oferta -> contratación.

2. PROGRAMA DE REFERIDOS
- Cualquier colaborador puede referir candidatos para vacantes abiertas a través del portal
  de RR. HH.
- Si el referido es contratado y supera el periodo de prueba (90 días), el colaborador que lo
  refirió recibe un bono de referido.
- No aplica para posiciones de dirección ni para familiares directos del referente
  (para evitar conflicto de interés).

3. ONBOARDING (INDUCCIÓN DE NUEVOS INGRESOS)
Pasos del proceso de onboarding:
3.1 Antes del primer día: TI prepara los accesos y el equipo; RR. HH. envía la bienvenida.
3.2 Primer día: bienvenida, entrega de equipo, firma de documentos y recorrido por la empresa.
3.3 Primera semana: inducción a la cultura, políticas internas y herramientas; asignación de
    un "padrino" o mentor.
3.4 Plan 30-60-90 días: objetivos y seguimiento del nuevo colaborador con su jefe.
3.5 Evaluación de periodo de prueba a los 90 días.

4. DOCUMENTOS DE INGRESO
Identificación, datos bancarios, formularios de beneficios y contrato firmado."""

# Instanciar el objeto Path con el nuevo nombre del archivo
doc_reclutamiento_obj = Path(DOC_PATH_RECLUTAMIENTO)

# 3. Verificar si el documento existe o crearlo si no está presente
if not doc_reclutamiento_obj.exists():
    doc_reclutamiento_obj.write_text(RECLUTAMIENTO, encoding="utf-8")
    print("docpack creado")
else:
    print("El archivo ya existe. Procediendo a la lectura...")

# 4. Leer el texto del archivo utilizando .read_text() y guardarlo en la variable politica
reclutamiento = doc_reclutamiento_obj.read_text(encoding="utf-8")

# 5. Impresión de métricas requeridas e inspección de los primeros 400 caracteres
print(f"Caracteres totales: {len(reclutamiento)}")
print(" " * 60)
print(reclutamiento[:400])

# embdeing del 3er agente 
import re
from langchain_community.vectorstores import Chroma

def chunkear_por_tema(texto):
    """Divide el manual en un chunk por cada tema numerado (1., 2., 3., etc.)."""
    cabeceras = list(re.finditer(r"^\d+\.\s+[A-ZÁÉÍÓÚÑ]", texto, flags=re.MULTILINE))
    chunks = []
    for i, m in enumerate(cabeceras):
        ini = m.start()
        fin = cabeceras[i + 1].start() if i + 1 < len(cabeceras) else len(texto)
        parte = texto[ini:fin].strip()
        chunks.append(parte)
    return chunks

# 1. Obtenemos los chunks del manual de reglamento interno
chunks_reclutamiento = chunkear_por_tema(RECLUTAMIENTO)

# 2. Documentos auxiliares (Vacío por ahora)
chunks_adicionales = []

# Unimos los documentos procesados
chunks_totales_reclutamiento = chunks_reclutamiento + chunks_adicionales

# 3. Imprimir métricas de verificación
print(f"Total de chunks creados: {len(chunks_totales_reclutamiento)}")
print("=" * 60)

for i, c in enumerate(chunks_totales_reclutamiento):
    vista_previa = c.replace('\n', ' ')
    print(f"Chunk {i+1}: {vista_previa[:70]}...")

# 4. Creamos la lista de metadatos dinámicamente para cada chunk
metadatos_reclutamiento = [
    {"seccion": i + 1, "fuente": DOC_PATH_RECLUTAMIENTO} 
    for i in range(len(chunks_totales_reclutamiento))
]

# 5. Inicializamos el vectorstore en Chroma
vectorstore_reclutamiento = Chroma.from_texts(
    texts=chunks_totales_reclutamiento,
    embedding=embeddings,            
    metadatas=metadatos_reclutamiento,
    collection_name="reclutamiento_interno"
)

# 6. utilizamos retriever para obtener mejor resultados 
retriever_reclutamiento= vectorstore_reclutamiento.as_retriever(search_kwargs={"k": 2})

# rack del 3r agente 
PROMPT_CONOCIMIENTO_RECLUTAMIENTO = """Eres el asistente de Recursos Humanos de Patito S.A. Respondes sobre GUÍA DE RECLUTAMIENTO, REFERIDOS Y ONBOARDING
Reglas estrictas:
- Responde ÚNICAMENTE con base en el CONTEXTO entregado.
- Cita el número de sección cuando sea posible.
- Si la información no está en el contexto, responde exactamente: "No tengo esa información en la política."
- Se breve y directo. No inventes datos."""

def responder_RECLUTAMIENTO(pregunta: str) -> str:
    """Pipeline RAG: recupera contexto de la base de conocimiento de reglamento y genera la respuesta."""
    docs = retriever_reclutamiento.invoke(pregunta)
    contexto = "\n\n---\n\n".join([d.page_content for d in docs])

    msg = llm.invoke([
        {"role": "system", "content": PROMPT_CONOCIMIENTO_RECLUTAMIENTO},
        {"role": "user", "content": f"CONTEXTO:\n{contexto}\n\nPREGUNTA: {pregunta}"}
    ])
    return msg.content

# agente multimodal - crear formulario 
from PIL import Image, ImageDraw
import base64
def crear_formulario_demo(ruta="formulario_dependiente.png"):
    """Genera una imagen simple de formulario para probar el agente multimodal."""
    img = Image.new("RGB", (500, 400), "white")
    d = ImageDraw.Draw(img)
    lineas = [
        "PATITO S.A. - RECURSOS HUMANOS",
        "FORMULARIO DE INSCRIPCION DE DEPENDIENTE",
        "----------------------------------------",
        "Nombre del colaborador: Juan Perez",
        "Nombre del dependiente: Maria Perez",
        "Vinculo: Conyuge",
        "Fecha de nacimiento dependiente: ",      
        "Documento de respaldo adjunto: NO",      
        "----------------------------------------",
        "Fecha de solicitud: 2026-07-20",
        "Firma: ________________",
    ]
    y = 20
    for ln in lineas:
        d.text((20, y), ln, fill="black")
        y += 25
    img.save(ruta)
    return ruta
ruta_formulario = crear_formulario_demo()

# registro formulario - multimodal 
from langchain_core.messages import HumanMessage

def analizar_formulario(ruta_imagen: str) -> str:
    """Agente multimodal: envía la imagen a Gemini (vision) y valida/extrae los datos del formulario."""
    try:
        with open(ruta_imagen, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return f"No se encontró la imagen '{ruta_imagen}'."

    prompt = (
        "Analiza esta imagen de un formulario de RR.HH. de Patito S.A. "
        "Extrae y devuelve en líneas separadas: nombre del colaborador, nombre del dependiente, "
        "vínculo, fecha de nacimiento del dependiente, si tiene documento de respaldo adjunto (SI/NO), "
        "y fecha de solicitud. "
        "Si algún dato no aparece o está vacío, escribe 'no visible' o 'falta'. "
        "Al final, indica explícitamente si el formulario está COMPLETO o INCOMPLETO, "
        "y en caso de estar incompleto, lista qué datos faltan."
    )
    msg = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": f"data:image/png;base64,{b64}"},
    ])
    return llm.invoke([msg]).content

    #-- agente de accion 
   
from pathlib import Path
from datetime import datetime
from langchain.tools import tool

REGISTRO_PATH = "registro_solicitudes_rrhh.txt"

CAMPOS_VACACIONES = ["nombre_solicitante", "fecha_inicio", "fecha_fin", "dias", "jefe_aprueba"]
CAMPOS_DEPENDIENTE = ["nombre_solicitante", "nombre_dependiente", "vinculo", "documento_respaldo"]
DIAS_ANTICIPACION_MINIMOS = 15

def _siguiente_id(prefijo: str) -> str:
    if not Path(REGISTRO_PATH).exists():
        return f"{prefijo}-0001"
    n = sum(1 for l in open(REGISTRO_PATH, encoding="utf-8") if l.strip().startswith(prefijo))
    return f"{prefijo}-{n + 1:04d}"


@tool
def registrar_solicitud_rrhh(tipo: str = "", nombre_solicitante: str = "",
                              fecha_inicio: str = "", fecha_fin: str = "", dias: int = 0,
                              jefe_aprueba: str = "", nombre_dependiente: str = "",
                              vinculo: str = "", documento_respaldo: str = "",
                              confirmar: bool = False) -> str:
    """Registra una solicitud de RR.HH. en un archivo de texto. El parametro 'tipo' debe ser
    'vacaciones' o 'dependiente'.
    Si tipo='vacaciones', requiere TODOS estos datos: nombre_solicitante, fecha_inicio (YYYY-MM-DD),
    fecha_fin (YYYY-MM-DD), dias (numero de dias) y jefe_aprueba. La fecha_inicio debe tener al
    menos 15 dias de anticipacion respecto a hoy.
    Si tipo='dependiente', requiere TODOS estos datos: nombre_solicitante, nombre_dependiente,
    vinculo (ej. conyuge, hijo) y documento_respaldo.
    Si falta algun dato obligatorio o no cumple la anticipacion, NO registra y devuelve que datos
    faltan o que corregir. Solo escribe en el archivo cuando confirmar=True; si confirmar=False,
    devuelve un resumen pidiendo confirmacion explicita antes de registrar."""

    tipo = tipo.strip().lower()

    if tipo == "vacaciones":
        datos = {"nombre_solicitante": nombre_solicitante, "fecha_inicio": fecha_inicio,
                  "fecha_fin": fecha_fin, "dias": dias, "jefe_aprueba": jefe_aprueba}

        # --- SISTEMA DE CONTROL: validar campos obligatorios ---
        faltantes = [k for k in CAMPOS_VACACIONES
                     if not str(datos[k]).strip() or (k == "dias" and int(dias) <= 0)]
        if faltantes:
            return "No se registro la solicitud. Faltan datos obligatorios: " + ", ".join(faltantes) + "."

        #  validar anticipacion minima (15 dias) ---
        try:
            fecha_inicio_dt = datetime.strptime(fecha_inicio, "%Y-%m-%d")
        except ValueError:
            return f"No se registro la solicitud: fecha_inicio '{fecha_inicio}' no tiene formato valido (YYYY-MM-DD)."

        dias_restantes = (fecha_inicio_dt - datetime.now()).days
        if dias_restantes < DIAS_ANTICIPACION_MINIMOS:
            return (f"No se registro la solicitud: se requieren al menos {DIAS_ANTICIPACION_MINIMOS} "
                    f"dias de anticipacion (faltan {DIAS_ANTICIPACION_MINIMOS - dias_restantes}).")

        firma = f"VAC|{nombre_solicitante}|{fecha_inicio}|{fecha_fin}|{dias}|{jefe_aprueba}"
        resumen = (f"{nombre_solicitante}, del {fecha_inicio} al {fecha_fin} ({dias} dias), "
                   f"aprobado por {jefe_aprueba}")
        linea_datos = (f"{nombre_solicitante} | {fecha_inicio} a {fecha_fin} | {dias} dias | "
                       f"Aprueba: {jefe_aprueba}")
        prefijo_id = "VAC"

    elif tipo == "dependiente":
        datos = {"nombre_solicitante": nombre_solicitante, "nombre_dependiente": nombre_dependiente,
                  "vinculo": vinculo, "documento_respaldo": documento_respaldo}

        # validar campos obligatorios 
        faltantes = [k for k in CAMPOS_DEPENDIENTE if not str(datos[k]).strip()]
        if faltantes:
            return "No se registro la solicitud. Faltan datos obligatorios: " + ", ".join(faltantes) + "."

        firma = f"DEP|{nombre_solicitante}|{nombre_dependiente}|{vinculo}|{documento_respaldo}"
        resumen = (f"dependiente {nombre_dependiente} ({vinculo}) de {nombre_solicitante}, "
                   f"con respaldo: {documento_respaldo}")
        linea_datos = (f"Solicitante: {nombre_solicitante} | Dependiente: {nombre_dependiente} "
                       f"({vinculo}) | Respaldo: {documento_respaldo}")
        prefijo_id = "DEP"

    else:
        return "Tipo de solicitud no reconocido. Usa 'vacaciones' o 'dependiente'."

    #  evitar duplicados 
    if Path(REGISTRO_PATH).exists():
        with open(REGISTRO_PATH, encoding="utf-8") as f:
            if any(firma in l for l in f):
                return "Esta solicitud ya habia sido registrada previamente (duplicado). No se volvio a registrar."

    #  pedir confirmacion antes de escribir 
    if not confirmar:
        return f"Datos completos y validos: {resumen}. ¿Confirmas el registro? Vuelve a invocar con confirmar=True para finalizar."

    rid = _siguiente_id(prefijo_id)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    linea = f"{rid} | {ts} | {linea_datos} | firma:{firma}"

    try:
        with open(REGISTRO_PATH, "a", encoding="utf-8") as f:
            f.write(linea + "\n")
    except Exception as e:
        return f"Error al registrar: {e}"

    return f"Solicitud registrada con ID {rid}.  ->  {linea}"

if __name__ == "__main__":
    # Prueba 1: faltan datos -> el control lo impide
    print(registrar_solicitud_rrhh.invoke({
        "tipo": "vacaciones", "nombre_solicitante": "Juan Perez", "fecha_inicio": "2026-08-20"}))

    # Prueba 2: datos completos pero SIN confirmar -> pide confirmacion
    print(registrar_solicitud_rrhh.invoke({
        "tipo": "vacaciones", "nombre_solicitante": "Juan Perez", "fecha_inicio": "2026-08-20",
        "fecha_fin": "2026-08-27", "dias": 5, "jefe_aprueba": "Maria Gomez"}))

    # Prueba 3: datos completos y CONFIRMANDO -> registra
    print(registrar_solicitud_rrhh.invoke({
        "tipo": "vacaciones", "nombre_solicitante": "Juan Perez", "fecha_inicio": "2026-08-20",
        "fecha_fin": "2026-08-27", "dias": 5, "jefe_aprueba": "Maria Gomez", "confirmar": True}))

    # Prueba 4: mismo registro otra vez -> detecta duplicado
    print(registrar_solicitud_rrhh.invoke({
        "tipo": "vacaciones", "nombre_solicitante": "Juan Perez", "fecha_inicio": "2026-08-20",
        "fecha_fin": "2026-08-27", "dias": 5, "jefe_aprueba": "Maria Gomez", "confirmar": True}))

    # Prueba 5: dependiente incompleto
    print(registrar_solicitud_rrhh.invoke({
        "tipo": "dependiente", "nombre_solicitante": "Juan Perez", "nombre_dependiente": "Ana Perez"}))

    # Prueba 6: dependiente completo y confirmado
    print(registrar_solicitud_rrhh.invoke({
        "tipo": "dependiente", "nombre_solicitante": "Juan Perez", "nombre_dependiente": "Ana Perez",
        "vinculo": "conyuge", "documento_respaldo": "cedula.pdf", "confirmar": True}))
    #-- agente orquestador 
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
import uuid

@tool
def consultar_beneficios(pregunta: str) -> str:
    """Responde preguntas sobre el Manual de Beneficios y Compensaciones: seguro medico,
    dependientes, bonos, otros beneficios y estructura de compensacion. Usa la base de
    conocimiento embebida del Agente de Beneficios."""
    return responder_politica(pregunta)


@tool
def consultar_politicas_internas(pregunta: str) -> str:
    """Responde preguntas sobre el Reglamento Interno de Trabajo y Codigo de Conducta:
    jornada laboral, vacaciones, permisos, codigo de conducta y faltas/sanciones. Usa la
    base de conocimiento embebida del Agente de Politicas Internas."""
    return responder_POLITICA_reglamento(pregunta)


@tool
def consultar_reclutamiento(pregunta: str) -> str:
    """Responde preguntas sobre el proceso de seleccion, el programa de referidos y el
    onboarding de nuevos colaboradores. Usa la base de conocimiento embebida del Agente
    de Reclutamiento y Onboarding."""
    return responder_RECLUTAMIENTO(pregunta)


@tool
def analizar_formulario_tool(ruta_imagen: str) -> str:
    """Agente multimodal: analiza la imagen de un formulario de RR.HH. (por ejemplo, el
    formulario de inscripcion de dependiente) y extrae sus datos, indicando si esta
    completo o que informacion falta. Recibe la RUTA del archivo de imagen (ej. 'formulario_dependiente.png')."""
    return analizar_formulario(ruta_imagen)


tools_orquestador = [
    consultar_beneficios,
    consultar_politicas_internas,
    consultar_reclutamiento,
    analizar_formulario_tool,
    registrar_solicitud_rrhh,
]


# Prompt del orquestador

SYSTEM_PROMPT = """Eres el orquestador de la Mesa de Ayuda IA de Recursos Humanos de Patito S.A.
Coordinas cinco capacidades (tools). NUNCA respondas de memoria: siempre usa la tool
correspondiente para obtener la informacion antes de responder.

- consultar_beneficios: preguntas sobre seguro medico, dependientes, bonos y compensacion.
- consultar_politicas_internas: preguntas sobre vacaciones, permisos, jornada laboral,
  codigo de conducta y sanciones.
- consultar_reclutamiento: preguntas sobre proceso de seleccion, programa de referidos
  y onboarding.
- analizar_formulario_tool: cuando el usuario mencione o adjunte la RUTA de una imagen
  de un formulario (ej. formulario_dependiente.png).
- registrar_solicitud_rrhh: para REGISTRAR una solicitud de vacaciones o de inscripcion
  de dependiente (tipo='vacaciones' o tipo='dependiente').

Reglas de ruteo:
- Si la pregunta toca mas de un tema (ej. vacaciones Y beneficios), DEBES invocar todas
  las tools de conocimiento relevantes y consolidar ambas respuestas en una sola, clara
  y ordenada por tema.
- Si el usuario da la ruta de una imagen, usa analizar_formulario_tool primero. Si de esa
  imagen surge un registro pendiente (ej. datos de un dependiente), puedes complementarlo
  con consultar_beneficios antes de responder.
- Si el usuario pide registrar/guardar una solicitud, usa registrar_solicitud_rrhh.
  Necesitas, segun el tipo:
  - vacaciones: nombre_solicitante, fecha_inicio, fecha_fin, dias, jefe_aprueba
    (fecha_inicio con al menos 15 dias de anticipacion).
  - dependiente: nombre_solicitante, nombre_dependiente, vinculo, documento_respaldo.
  Si falta algun dato, PIDESELO al usuario y espera su respuesta; nunca registres con
  datos incompletos ni sin que el usuario confirme explicitamente (confirmar=True solo
  despues de que el usuario diga que si).
- Si ninguna tool de conocimiento devuelve informacion relevante, responde exactamente:
  "No encontre informacion suficiente en la base documental proporcionada." No inventes
  datos que no esten en el contexto recuperado.
- Al final de cada respuesta, agrega una linea "Agentes utilizados: ..." indicando que
  tool(s) invocaste, para dar trazabilidad."""


# Memoria: permite conversaciones multi-turno 

memoria = InMemorySaver()
orquestador = create_agent(
    model=llm,
    tools=tools_orquestador,
    system_prompt=SYSTEM_PROMPT,
    checkpointer=memoria,
)

print("Tools registradas en el orquestador:")
for t in tools_orquestador:
    print("  -", t.name)


def _imprimir_pasos(resultado):
    """Muestra que tools se invocaron y su resultado (trazabilidad)."""
    for m in resultado["messages"]:
        for tc in (getattr(m, "tool_calls", None) or []):
            print(f"[TOOL] {tc['name']}({tc['args']})")
        if m.__class__.__name__ == "ToolMessage":
            print(f"[RESPONSE] {str(m.content)[:300]}\n")


def extraer_texto(content):
    """Gemini a veces devuelve el content como una LISTA de bloques
    (texto + firmas de 'thinking'). Esta funcion devuelve solo el texto plano."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        partes = []
        for b in content:
            if isinstance(b, dict):
                partes.append(b.get("text", ""))
            elif isinstance(b, str):
                partes.append(b)
        return "".join(partes).strip()
    return str(content)


def consultar(pregunta: str, thread_id: str = None):
    """Invoca al orquestador (una consulta suelta) e imprime tools + respuesta final."""
    thread_id = thread_id or f"demo-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}
    print(f">>> Usuario: {pregunta}\n")
    resultado = orquestador.invoke({"messages": [{"role": "user", "content": pregunta}]}, config)
    _imprimir_pasos(resultado)
    print("=== Respuesta final ===")
    print(extraer_texto(resultado["messages"][-1].content))
    return resultado
if __name__ == "__main__":
    # Prueba 1: agente de conocimiento simple
    consultar("¿Cuántos días de vacaciones me corresponden al año?")
 
    # Prueba 2: consulta mixta (vacaciones + beneficios) -> obliga a invocar 2 tools
    consultar(
        "Voy a tomar mis vacaciones y además quiero agregar a mi pareja al seguro médico. "
        "¿Cuántos días me corresponden, cómo los solicito y qué necesito para inscribir "
        "a un dependiente en el beneficio?"
    )
 
    # Prueba 3: multimodal
    consultar("Adjunto el formulario en formulario_dependiente.png: ¿está completo y qué datos faltan?")
 
    # Prueba 4: registrar con datos incompletos -> el sistema de control debe pedir lo que falta
    consultar("Registra una solicitud de vacaciones para Juan Perez.")
 
    # Prueba 5: fuera de alcance -> debe admitir que no tiene informacion
    consultar("¿Cuál es el precio de las acciones de Patito S.A. en la bolsa?")
    
