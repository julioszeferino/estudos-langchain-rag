""" "
O langchain permite cadeias lineares de execução, onde a
saída de um componente é a entrada do próximo.

O langgraph permite a criação de cadeias mais complexas,
com ramificações, e paralelismo. Importante em processos
que sao independentes e que precisam decidir uma rota de
execução.
"""
import os
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

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


# fazendo a orquestracao das cadeias de forma manual
prompt_roteador = ChatPromptTemplate.from_messages(
    [
        ('system', "Responda apenas com 'praia' ou 'montanha'"),
        ('human', '{query}'),
    ]
)

roteador = prompt_roteador | modelo.with_structured_output(Rota)


def responder(pergunta: str):
    rota = roteador.invoke({'query': pergunta})['destino']
    if rota == 'praia':
        return cadeia_praia.invoke({'query': pergunta})
    elif rota == 'montanha':
        return cadeia_montanha.invoke({'query': pergunta})
    else:
        raise ValueError(f'Rota desconhecida: {rota}')


print(
    responder(
        'Quero visitar um lugar no Brasil, famoso por praias e cultura. \
                Pode sugerir?'
    )
)
