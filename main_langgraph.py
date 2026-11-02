""" "
O langchain permite cadeias lineares de execução, onde a
saída de um componente é a entrada do próximo.

O langgraph permite a criação de cadeias mais complexas,
com ramificações, e paralelismo. Importante em processos
que sao independentes e que precisam decidir uma rota de
execução.
"""
import asyncio
import os
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

load_dotenv()

api_key = os.getenv('ANTHROPIC_API_KEY')
if api_key is None:
    raise ValueError(
        'A variável de ambiente ANTHROPIC_API_KEY não está definida.'
    )

modelo = ChatAnthropic(
    anthropic_api_key=api_key,
    model='claude-haiku-5-5',
)

prompt_consultor_praia = ChatPromptTemplate.from_messages(
    [
        (
            'system',
            'Apresente-se como Sra Praia. Você é uma especialista em \
         viagens com destinos para praia.',
        ),
        ('human', '{query}'),
    ]
)

prompt_consultor_montanha = ChatPromptTemplate.from_messages(
    [
        (
            'system',
            'Apresente-se como Sr Montanha. Você é uma especialista em \
         viagens com destinos para montanhas e atividades radicais.',
        ),
        ('human', '{query}'),
    ]
)

cadeia_praia = prompt_consultor_praia | modelo | StrOutputParser()
cadeia_montanha = prompt_consultor_montanha | modelo | StrOutputParser()


class Rota(TypedDict):
    destino: Literal['praia', 'montanha']


# Criando um roteador que toma decisao
prompt_roteador = ChatPromptTemplate.from_messages(
    [
        ('system', "Responda apenas com 'praia' ou 'montanha'"),
        ('human', '{query}'),
    ]
)

roteador = prompt_roteador | modelo.with_structured_output(Rota)


# Como vamos executar as acoes em paralelo e nao sabemos
# quanto tempo cada uma vai levar, vamos criar metodos assincronos
# para cada acao.
class Estado(TypedDict):
    query: str
    destino: Rota
    resposta: str


async def no_roteador(estado: Estado, config: RunnableConfig):
    """Executa o roteador para decidir qual cadeia executar."""
    return {
        'destino': await roteador.ainvoke(
            {'query': estado['query']}, config=config
        )
    }


async def no_praia(estado: Estado, config: RunnableConfig):
    """Executa a cadeia de praia."""
    return {
        'resposta': await cadeia_praia.ainvoke(
            {'query': estado['query']}, config=config
        )
    }


async def no_montanha(estado: Estado, config: RunnableConfig):
    """Executa a cadeia de montanha."""
    return {
        'resposta': await cadeia_montanha.ainvoke(
            {'query': estado['query']}, config=config
        )
    }


def escolher_no(estado: Estado) -> Literal['praia', 'montanha']:
    """Escolhe qual nó executar com base na decisão do roteador."""
    return 'praia' if estado['destino']['destino'] == 'praia' else 'montanha'


# Monta o grafo de estados para
# decidir qual cadeia executar.
grafo = StateGraph(Estado)
grafo.add_node('rotear', no_roteador)
grafo.add_node('praia', no_praia)
grafo.add_node('montanha', no_montanha)
# Adiciona regra de destino
grafo.add_edge(START, 'rotear')
grafo.add_conditional_edges('rotear', escolher_no)
# Adiciona regras de finalizacao
grafo.add_edge('praia', END)
grafo.add_edge('montanha', END)

# Compila o grafo em um Runnable
app = grafo.compile()


async def main():
    resposta = await app.ainvoke(
        {
            'query': 'Quero visitar um lugar no Brasil famoso por praias\
                  e cultura'
        }
    )
    print(resposta['resposta'])


asyncio.run(main())
