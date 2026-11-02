import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

SRC_DIR = Path(__file__).resolve().parents[1] / 'src'


@pytest.fixture
def carregar(monkeypatch):
    """Importa um módulo de app/src com as dependências externas mockadas."""

    def _carregar(nome, api_key='chave-teste'):
        if api_key:
            monkeypatch.setenv('ANTHROPIC_API_KEY', api_key)
        else:
            monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)

        mocks = MagicMock()
        mocks.chat.return_value = RunnableLambda(
            lambda _: AIMessage(content='resposta do modelo')
        )
        mocks.loader.return_value.load.return_value = [
            Document(page_content='conteudo')
        ]
        mocks.retriever = (
            mocks.faiss.from_documents.return_value.as_retriever.return_value
        )
        mocks.retriever.invoke.return_value = [Document(page_content='trecho')]

        with patch('dotenv.load_dotenv'), patch(
            'langchain_anthropic.ChatAnthropic', mocks.chat
        ), patch('langchain_huggingface.HuggingFaceEmbeddings'), patch(
            'langchain_community.vectorstores.FAISS', mocks.faiss
        ), patch(
            'langchain_community.document_loaders.TextLoader', mocks.loader
        ), patch(
            'langchain_community.document_loaders.PyPDFLoader', mocks.loader
        ):
            spec = importlib.util.spec_from_file_location(
                nome, SRC_DIR / f'{nome}.py'
            )
            modulo = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(modulo)

        return modulo, mocks

    return _carregar
