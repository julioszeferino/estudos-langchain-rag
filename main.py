from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")

if api_key is None:
    raise ValueError("A variável de ambiente ANTHROPIC_API_KEY não está definida.")

numero_dias = 7
numero_criancas = 2
atividade = "praia"
modelo_de_prompt = PromptTemplate(
    template = """
    Crie um roteiro de viagem de {numero_dias} dias,
    para {numero_criancas} crianças,
    com foco em atividades de {atividade}.
    """
)

# prompt = f"""
# Crie um roteiro de viagem de {numero_dias} dias para {numero_criancas} crianças, 
# com foco em atividades de {atividade}.
# """

prompt = modelo_de_prompt.format(
    numero_dias=numero_dias,
    numero_criancas=numero_criancas,
    atividade=atividade
)

# Podemos ter vários modelos intanciados
# inclusive de outros provedores, como OpenAI, Google, etc.
# o objetivo do langchain é permitir que você troque de 
# modelo facilmente, sem precisar mudar o código. Permitindo
# que cada modelo execute a tarefa na qual ele é mais 
# eficiente.
modelo_claude = ChatAnthropic(
    anthropic_api_key=api_key,
    model="claude-haiku-5-5",
)

resposta = modelo_claude.invoke(prompt)

print(resposta.content)
