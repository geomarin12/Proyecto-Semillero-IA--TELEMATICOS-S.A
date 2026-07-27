"""
app.py — Interfaz web (Streamlit) para la Mesa de Ayuda IA de Patito S.A.
Ejecutar con: streamlit run app.py
"""

import uuid
import streamlit as st

st.set_page_config(page_title="Mesa de Ayuda IA — Patito S.A.", page_icon="🦆", layout="centered")

# ---------------------------------------------------------
# Cargar el backend (orquestador) con manejo de errores
# ---------------------------------------------------------
try:
    from backend import consultar
    backend_disponible = True
    error_backend = None
except Exception as e:
    backend_disponible = False
    error_backend = str(e)

st.title("🦆 Mesa de Ayuda IA — Recursos Humanos")
st.caption("Patito S.A. · Beneficios · Políticas Internas · Reclutamiento · Formularios · Registro de solicitudes")

if not backend_disponible:
    st.error(
        "No se pudo cargar el backend (orquestador). Revisa que backend.py esté completo "
        "y que la GOOGLE_API_KEY esté configurada en tu archivo .env.\n\n"
        f"Detalle del error: {error_backend}"
    )
    st.stop()

# ---------------------------------------------------------
# Memoria de la conversación (thread_id fijo por sesión de navegador)
# ---------------------------------------------------------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"streamlit-{uuid.uuid4().hex[:8]}"

if "historial" not in st.session_state:
    st.session_state.historial = []  # lista de (pregunta, respuesta, agentes_usados)

# ---------------------------------------------------------
# Barra lateral: subir imagen (para el agente multimodal)
# ---------------------------------------------------------
with st.sidebar:
    st.header("📎 Adjuntar formulario (opcional)")
    st.caption("Para el agente multimodal: sube una imagen y su ruta se agregará a tu pregunta.")
    imagen = st.file_uploader("Formulario / comprobante", type=["png", "jpg", "jpeg"])
    ruta_imagen_actual = None
    if imagen is not None:
        ruta_imagen_actual = f"_subida_{imagen.name}"
        with open(ruta_imagen_actual, "wb") as f:
            f.write(imagen.getbuffer())
        st.image(imagen, caption="Vista previa", use_container_width=True)
        st.success(f"Imagen guardada como: {ruta_imagen_actual}")

    st.divider()
    if st.button("🗑️ Reiniciar conversación"):
        st.session_state.historial = []
        st.session_state.thread_id = f"streamlit-{uuid.uuid4().hex[:8]}"
        st.rerun()

# ---------------------------------------------------------
# Mostrar historial de la conversación
# ---------------------------------------------------------
for pregunta, respuesta, agentes in st.session_state.historial:
    with st.chat_message("user"):
        st.write(pregunta)
    with st.chat_message("assistant"):
        st.write(respuesta)
        if agentes:
            st.caption(f"🔎 Agentes usados: {', '.join(agentes)}")

# ---------------------------------------------------------
# Entrada de la pregunta (estilo chat)
# ---------------------------------------------------------
pregunta_usuario = st.chat_input("Escribe tu pregunta para RR.HH...")

if pregunta_usuario:
    # Si hay imagen subida, se agrega la ruta a la pregunta para que el orquestador la detecte
    pregunta_final = pregunta_usuario
    if ruta_imagen_actual:
        pregunta_final = f"{pregunta_usuario} (imagen adjunta en la ruta: {ruta_imagen_actual})"

    with st.chat_message("user"):
        st.write(pregunta_usuario)

    with st.chat_message("assistant"):
        with st.spinner("Consultando a los agentes..."):
            try:
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                resultado = consultar(pregunta_final, thread_id=st.session_state.thread_id)

                # Extraer texto final y tools usadas para trazabilidad
                mensajes = resultado["messages"]
                agentes_usados = []
                for m in mensajes:
                    for tc in (getattr(m, "tool_calls", None) or []):
                        agentes_usados.append(tc["name"])

                contenido = mensajes[-1].content
                if isinstance(contenido, list):
                    texto_final = "".join(
                        b.get("text", "") for b in contenido if isinstance(b, dict)
                    ).strip()
                else:
                    texto_final = str(contenido)

            except Exception as e:
                texto_final = f"Ocurrió un error al consultar los agentes: {e}"
                agentes_usados = []

        st.write(texto_final)
        if agentes_usados:
            st.caption(f"🔎 Agentes usados: {', '.join(agentes_usados)}")

    st.session_state.historial.append((pregunta_usuario, texto_final, agentes_usados))