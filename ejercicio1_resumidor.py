"""
Construye una cadena (LCEL) que reciba un texto largo y devuelva un resumen en 3 viñetas. Debe usar el patrón prompt | modelo | parser de salida.
1) Escribe el prompt con una instrucción clara.
2) Conecta un modelo gratis (Ollama o Gemini free tier).
3) Añade un parser que fuerce 3 viñetas.
4) Pruébalo con 2 textos distintos.

Entregable: Repositorio con el script + captura de las 2 ejecuciones de ejemplo.

requirements
La cadena mantiene memoria persistente entre turnos de conversación.

Al menos una tool está integrada.

La cadena se ejecuta de principio a fin sin errores.

"""

from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, ToolMessage


@tool
def format_texto(texto: str) -> str:
    """Formatea el texto en mayúsculas."""
    return texto.upper()


tools = [format_texto]
tools_por_nombre = {t.name: t for t in tools}


modelo_base = ChatOllama(model="llama3.1", temperature=0)
modelo = modelo_base.bind_tools(tools)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Resume únicamente el texto del último mensaje del usuario "
            "en exactamente tres viñetas breves. "
            "Selecciona las ideas más relevantes y usa tus propias palabras. "
            "Devuelve exactamente tres líneas, cada una empezando por '- '. "
            "No añadas introducción, títulos ni cierre.",
        ),
        MessagesPlaceholder(variable_name="historial"),
        ("human", "{texto}"),
    ]
)


def ejecutar_modelo_con_tools(entrada, config):
    mensajes = entrada.to_messages()

    ai = modelo.invoke(mensajes, config=config)

    if not ai.tool_calls:
        return ai

    mensajes.append(ai)

    for llamada in ai.tool_calls:
        herramienta = tools_por_nombre[llamada["name"]]

        resultado = herramienta.invoke(
            llamada["args"],
            config=config,
        )

        mensajes.append(
            ToolMessage(
                content=str(resultado),
                tool_call_id=llamada["id"],
            )
        )

    return modelo_base.invoke(mensajes, config=config)


def validar_vinetas(texto: str) -> str:
    lineas = [linea.strip() for linea in texto.splitlines() if linea.strip()]

    if len(lineas) != 3 or not all(
        linea.startswith(("- ", "* ", "• ")) and linea[2:].strip() for linea in lineas
    ):
        raise OutputParserException(
            "Se esperaban exactamente tres viñetas. " f"Respuesta recibida: {texto!r}"
        )

    return "\n".join(f"- {linea[2:].strip()}" for linea in lineas)


cadena = (
    prompt
    | RunnableLambda(ejecutar_modelo_con_tools)
    | StrOutputParser()
    | RunnableLambda(validar_vinetas)
    | RunnableLambda(lambda resumen: format_texto.invoke({"texto": resumen}))
)

historial: dict[str, InMemoryChatMessageHistory] = {}


def obtener_historial(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in historial:
        historial[session_id] = InMemoryChatMessageHistory()
    return historial[session_id]


chat = RunnableWithMessageHistory(
    cadena,
    obtener_historial,
    input_messages_key="texto",
    history_messages_key="historial",
)

entradaSesion1 = {
    "Texto 1": (
        """
    El pueblo de Valdeluz ha inaugurado una biblioteca en una antigua
    estación de tren que llevaba veinte años abandonada. Los vecinos
    donaron más de dos mil libros y participaron en la restauración
    del edificio, conservando el reloj y los bancos originales.

    La biblioteca abrirá de martes a sábado y contará con una sala
    de estudio, ordenadores con acceso a internet y una zona infantil.
    Cada mes se organizarán talleres gratuitos de escritura y un club
    de lectura. El ayuntamiento también pondrá en marcha un servicio
    de préstamo a domicilio para las personas con movilidad reducida.
    """
    ),
    "Texto 2": (
        """
    Una empresa ha diseñado un robot que recoge basura en las playas.
    Funciona con energía solar y utiliza cámaras para distinguir los
    residuos de las piedras, las algas y los animales pequeños.
    Su depósito puede almacenar hasta cincuenta kilos de basura.

    Durante su primera prueba recogió treinta kilos de plástico
    y varias latas, aunque tuvo dificultades para avanzar por la
    arena húmeda. Los ingenieros modificarán las ruedas para mejorar
    su movilidad. El municipio probará dos unidades durante el verano
    y evaluará su rendimiento antes de comprar más equipos.
    """
    ),
}

entradaSesion2 = """
    Un grupo de estudiantes ha creado un huerto en la azotea de su colegio.
    Cultivan tomates, lechugas y hierbas aromáticas en mesas de cultivo
    fabricadas con materiales reutilizados. Para regar utilizan agua
    de lluvia almacenada en un depósito y un sistema de riego por goteo.

    Los alumnos se reparten las tareas y registran cada semana el
    crecimiento de las plantas. La cosecha se empleará en el comedor
    escolar y los restos vegetales servirán para elaborar compost.
    El proyecto también permitirá estudiar cómo influyen la luz y
    la temperatura en los cultivos. Si funciona bien, el próximo curso
    se ampliará con una zona de flores para atraer insectos polinizadores.
"""


if __name__ == "__main__":
    config = {"configurable": {"session_id": "1"}}

    for i, texto in enumerate(entradaSesion1.values(), 1):
        print(f"[Sesión 1: Turno {i}] Yo:\n{texto}\n")

        resumen = chat.invoke({"texto": texto}, config=config)

        print(f"[Sesión 1: Turno {i}] Bot:\n{resumen}\n")

    print(f"[Sesión 2] Yo:\n{entradaSesion2}\n")

    resumenChatSinMemoriaAnterior = chat.invoke(
        {"texto": entradaSesion2},
        config={"configurable": {"session_id": "2"}},
    )

    print(f"[Sesión 2] Bot:\n{resumenChatSinMemoriaAnterior}\n")

    print(
        "Mensajes guardados en 'Sesión 1':",
        len(historial["1"].messages),
    )
    print(
        "Mensajes guardados en 'Sesión 2':",
        len(historial["2"].messages),
    )
    print(
        "Mensajes guardados en total:",
        sum(len(h.messages) for h in historial.values()),
    )
