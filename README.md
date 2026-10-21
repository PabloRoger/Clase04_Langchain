# Ejercicio 1 - Resumidor con LCEL

Cadena de LangChain que recibe un texto largo y devuelve un resumen en **tres viñetas**.

## Qué incluye el script

- **Prompt → modelo → parser** con LCEL (`prompt | modelo | parser`).
- **Modelo gratuito en local** con Ollama (`llama3.1`), sin API key.
- **Parser que fuerza tres viñetas**: si la respuesta no tiene exactamente tres líneas que empiecen por `- `, lanza un error.
- **Memoria persistente** entre turnos con `RunnableWithMessageHistory`, separada por sesiones (`session_id`).
- **Una tool**: `format_texto`, que pasa el resumen a mayúsculas.
- Se ejecuta con dos textos en la sesión 1 y un texto en la sesión 2 (la sesión 2 no ve el historial de la 1).

## Requisitos

- Python 3.10 o superior.
- [Ollama](https://ollama.com) instalado y en ejecución.

## Instalación y ejecución (Windows PowerShell)

```powershell
# 1. Descargar el modelo que usa el script (solo una vez)
ollama pull llama3.1

# 2. Crear y activar un entorno virtual
python -m venv .venv
.venv\Scripts\Activate.ps1

# 3. Instalar las dependencias
pip install -r requirements.txt

# 4. Ejecutar el script
python ejercicio1_resumidor.py
```

En macOS o Linux, el paso 2 es `python -m venv .venv` seguido de `source .venv/bin/activate`.

## Salida esperada

La captura `solucion_terminal.png` muestra una ejecución completa. Al final se imprime el número de mensajes guardados por sesión:

```
Mensajes guardados en 'Sesión 1': 4
Mensajes guardados en 'Sesión 2': 2
Mensajes guardados en total: 6
```

Al principio puede aparecer un `LangChainDeprecationWarning` sobre `RunnableWithMessageHistory` e `InMemoryChatMessageHistory`. Es un aviso de obsolescencia y no impide que el script funcione.

## Archivos

| Archivo | Descripción |
|---|---|
| `ejercicio1_resumidor.py` | Script principal del ejercicio |
| `requirements.txt` | Dependencias de Python |
| `solucion_terminal.png` | Captura de una ejecución de ejemplo |
| `README.md` | Este documento |
