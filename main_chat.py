import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory

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

prompt_sugestao = ChatPromptTemplate.from_messages(
    [
        (
            'system',
            """
            Voce e um guia de viagem especializado em recomendar
            destinos brasileiros. Apresente-se como Sr. Passeios.
            """,
        ),
        ('placeholder', '{historico}'),
        ('human', '{query}'),
    ]
)

cadeia = prompt_sugestao | modelo | StrOutputParser()

memoria = {}
sessao = 'aula_langchain'


def historico_sessao(sessao: str):
    if sessao not in memoria:
        memoria[sessao] = InMemoryChatMessageHistory()
    return memoria[sessao]


lista_perguntas = [
    "Quero visitar um lugar no Brasil, famoso por praias e cultura. \
    Pode sugerir?",
    """Qual a melhor epoca do ano pra ir?""",
]

cadeia_com_memoria = RunnableWithMessageHistory(
    runnable=cadeia,
    get_session_history=historico_sessao,
    input_messages_key='query',
    history_messages_key='historico',
)

for pergunta in lista_perguntas:
    resposta = cadeia_com_memoria.invoke(
        {'query': pergunta}, config={'session_id': sessao}
    )
    print(f'Pergunta: {pergunta}')
    print(f'Resposta: {resposta}\n')
