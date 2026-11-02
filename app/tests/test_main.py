import pytest


def test_sem_api_key_levanta_erro(carregar):
    # Arrange
    api_key = None

    # Act / Assert
    with pytest.raises(ValueError):
        carregar('main', api_key=api_key)


def test_modelo_criado_com_api_key(carregar):
    # Arrange
    api_key = 'minha-chave'

    # Act
    _, mocks = carregar('main', api_key=api_key)

    # Assert
    mocks.chat.assert_called_once_with(
        anthropic_api_key='minha-chave', model='claude-haiku-5-5'
    )


def test_carrega_documento_txt(carregar):
    # Arrange
    caminho = 'app/gtb_docs/GTB_gold_Nov23.txt'

    # Act
    _, mocks = carregar('main')

    # Assert
    mocks.loader.assert_called_once_with(file_path=caminho, encoding='utf-8')


def test_responder_retorna_resposta_do_modelo(carregar):
    # Arrange
    modulo, mocks = carregar('main')

    # Act
    resposta = modulo.responder('Minha pergunta')

    # Assert
    assert resposta == 'resposta do modelo'
    mocks.retriever.invoke.assert_called_with('Minha pergunta')
