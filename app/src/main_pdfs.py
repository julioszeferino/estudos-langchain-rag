import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

embeddings = HuggingFaceEmbeddings(
    model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
)

arquivos = [
    'app/gtb_docs/GTB_standard_Nov23.pdf',
    'app/gtb_docs/GTB_gold_Nov23.pdf',
    'app/gtb_docs/GTB_platinum_Nov23.pdf',
]

documentos = sum([PyPDFLoader(arquivo).load() for arquivo in arquivos], [])

chunks = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100  # sobreposicao
).split_documents(documentos)

dados_recuperados = FAISS.from_documents(chunks, embeddings).as_retriever(
    search_kwargs={'k': 2}
)


prompt_consulta_seguro = ChatPromptTemplate.from_messages(
    [
        ('system', 'Responda usando exclusivamente o conteúdo fornecido'),
        ('human', '{query}\n\nContexto: \n{contexto}\n\nResposta:'),
    ]
)

cadeia = prompt_consulta_seguro | modelo | StrOutputParser()


def responder(pergunta: str):
    trechos = dados_recuperados.invoke(pergunta)
    contexto = '\n\n'.join(trecho.page_content for trecho in trechos)
    return cadeia.invoke({'query': pergunta, 'contexto': contexto})


print(
    responder(
        'Como devo proceder caso tenha um item comprado roubado e caso \
            eu tenha o cartão gold'
    )
)
print(
    responder(
        'Como devo proceder caso tenha um item comprado roubado e caso \
            eu tenha o cartão platinum'
    )
)
