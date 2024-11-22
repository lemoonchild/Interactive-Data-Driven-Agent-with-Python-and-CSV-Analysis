import os
import datetime
from typing import Any, Dict

import streamlit as st
from dotenv import load_dotenv
from langchain import hub
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_experimental.tools import PythonREPLTool
from langchain_experimental.agents.agent_toolkits import create_csv_agent

load_dotenv()


def save_history(question, answer):
    with open("history.txt", "a") as f:
        f.write(f"{datetime.datetime.now()}: {question}-->{answer}\n")


def load_history():
    if os.path.exists("history.txt"):
        with open("history.txt", "r") as f:
            return f.readlines()
    return []


def main():
    st.set_page_config(page_title="Agente Interactivo", page_icon="🤖", layout="wide")

    st.title("🤖 Agente Interactivo")
    st.markdown("### Descripción del Bot")
    st.markdown("""
    Este bot tiene la capacidad de ejecutar código Python para responder preguntas y también de responder preguntas basadas en datos de diferentes archivos CSV. 
    Utiliza el input para escribir preguntas específicas o selecciona una de las opciones para ejecutar ejemplos predefinidos.
    """)

    instructions = """You are an agent designed to write and execute python code to answer questions.
    You have access to a python REPL, which you can use to execute python code.
    You have qrcode package installed.
    If you get an error, debug your code and try again.
    Only use the output of your code to answer the question.
    You might know the answer without running any code, but you should still run the code to get the answer.
    If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
    """

    base_prompt = hub.pull("langchain-ai/react-agent-template")
    prompt = base_prompt.partial(instructions=instructions)

    tools = [PythonREPLTool()]
    python_agent = create_react_agent(
        prompt=prompt,
        llm=ChatOpenAI(temperature=0, model="gpt-4o"),
        tools=tools,
    )

    python_agent_executor = AgentExecutor(agent=python_agent, tools=tools, verbose=True)

    # Definición de agentes para cada archivo CSV
    cats_agent = create_csv_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4"),
        path="cats_dataset.csv",
        verbose=True,
        allow_dangerous_code=True,
    )
    dogs_agent = create_csv_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4"),
        path="dogs_breads_around_world.csv",
        verbose=True,
        allow_dangerous_code=True,
    )
    sleep_patterns_agent = create_csv_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4"),
        path="student_sleep_patterns.csv",
        verbose=True,
        allow_dangerous_code=True,
    )
    healthy_food_agent = create_csv_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4"),
        path="top_100_healthiest_food_in_the_world.csv",
        verbose=True,
        allow_dangerous_code=True,
    )

    def python_agent_executor_wrapper(original_prompt: str) -> dict[str, Any]:
        return python_agent_executor.invoke({"input": original_prompt})

    # Funciones de envoltura para cada agente CSV
    def cats_agent_wrapper(input_prompt: str) -> dict[str, Any]:
        return cats_agent.invoke({"input": input_prompt})

    def dogs_agent_wrapper(input_prompt: str) -> dict[str, Any]:
        return dogs_agent.invoke({"input": input_prompt})

    def sleep_patterns_wrapper(input_prompt: str) -> dict[str, Any]:
        return sleep_patterns_agent.invoke({"input": input_prompt})

    def healthy_food_wrapper(input_prompt: str) -> dict[str, Any]:
        return healthy_food_agent.invoke({"input": input_prompt})

    tools = [
        Tool(name="Python Agent", func=python_agent_executor_wrapper,
             description="""Useful when you need to transform natural language to python and execute the python code, 
             returning the results of the code execution DOES NOT ACCEPT CODE AS INPUT"""),
        Tool(name="Cats Data Agent", func=cats_agent_wrapper,
             description="Provides detailed answers about cat breeds, including age, weight, color, and gender based "
                         "on the cats_dataset.csv."),
        Tool(name="Dogs Data Agent", func=dogs_agent_wrapper,
             description="Enables queries about various dog breeds from around the world, offering insights into "
                         "breed specifics directly from the dogs_breads_around_world.csv."),
        Tool(name="Sleep Patterns Agent", func=sleep_patterns_wrapper,
             description="Analyzes and responds to queries on student sleep patterns, including sleep duration, "
                         "study hours, screen time, and more, utilizing data from student_sleep_patterns.csv."),
        Tool(name="Healthy Food Agent", func=healthy_food_wrapper,
             description="Answers questions related to nutritional values, origin, and health benefits of foods "
                         "listed in the top_100_healthiest_food_in_the_world.csv, focusing on aspects like calories, "
                         "protein content, fiber, and vitamins."),
    ]

    prompt = base_prompt.partial(instructions="")
    grand_agent = create_react_agent(
        prompt=prompt,
        llm=ChatOpenAI(temperature=0, model="gpt-4-turbo"),
        tools=tools
    )

    grand_agent_executor = AgentExecutor(
        agent=grand_agent,
        tools=tools,
        verbose=True,
    )

    # Interfaz de Streamlit

    # Sección de Python REPL
    st.markdown("## Python REPL")
    python_options = [
        "La división de los números 4000 sobre 40",
        "Crea un juego básico de snake con la librería pygame",
        "Genera un patrón piramidal de asteriscos de tamaño 5",
    ]
    python_example = st.selectbox("Ejemplos de Python", python_options, key="python_example")
    if st.button("Ejecutar Python", key="execute_python"):
        loading_placeholder = st.empty()
        loading_placeholder.markdown("### Procesando tu respuesta...")

        try:
            response = python_agent_executor_wrapper(python_example)
            if 'output' in response:
                st.markdown("### Respuesta del agente:")
                st.code(response['output'], language="python")
                save_history(python_example, response['output'])
            else:
                st.error("No hubo salida del agente.")
        except Exception as e:
            st.error(f"Error al ejecutar el agente: {str(e)}")

    st.markdown("---")  # Separador

    # Sección de Interacción con archivos CSV
    st.markdown("## Interacción con archivos CSV")
    st.markdown("### Descripción de los archivos CSV disponibles")
    st.markdown("""
    - **cats_dataset.csv**: Contiene información sobre razas de gatos, incluyendo edad, peso, color y género.
    - **dogs_breads_around_world.csv**: Detalles sobre razas de perros de todo el mundo.
    - **student_sleep_patterns.csv**: Patrones de sueño de estudiantes, incluyendo duración del sueño y hábitos de estudio.
    - **top_100_healthiest_food_in_the_world.csv**: Información nutricional sobre los alimentos más saludables.
    """)

    csv_question = st.text_input("Escribe tu pregunta", key="csv_input")
    if st.button("Ejecutar pregunta", key="execute_query"):
        loading_placeholder = st.empty()
        loading_placeholder.markdown("### Procesando tu respuesta...")
        try:
            response = grand_agent_executor.invoke({"input": csv_question})
            if 'output' in response:
                st.markdown("### Respuesta del agente:")
                st.write(response['output'])
                save_history(csv_question, response['output'])
            else:
                st.error("No se pudo obtener una respuesta adecuada para la pregunta.")
        except Exception as e:
            st.error(f"Error al procesar la pregunta: {str(e)}")

    st.markdown("## Historial")
    if st.button("Cargar Historial", key="load_history"):
        history = load_history()
        for entry in history:
            st.text(entry)

    st.write("")  #
    st.write("")


if __name__ == "__main__":
    main()
