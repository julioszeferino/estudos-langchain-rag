import pytest


def test_sem_api_key_levanta_erro(carregar):
    # Arrange
    api_key = None

    # Act / Assert
    with pytest.raises(ValueError):
        carregar('main_pdfs', api_key=api_key)


def test_carrega_os_tres_pdfs(carregar):
    # Arrange
    arquivos = [
        'app/gtb_docs/GTB_standard_Nov23.pdf',
        'app/gtb_docs/GTB_gold_Nov23.pdf',
        'app/gtb_docs/GTB_platinum_Nov23.pdf',
    ]

    # Act
    _, mocks = carregar('main_pdfs')

    # Assert
    chamados = [c.args[0] for c in mocks.loader.call_args_list]
    assert chamados == arquivos


def test_responder_retorna_resposta_do_modelo(carregar):
    # Arrange
    modulo, mocks = carregar('main_pdfs')

    # Act
    resposta = modulo.responder('Minha pergunta')

    # Assert
    assert resposta == 'resposta do modelo'
    mocks.retriever.invoke.assert_called_with('Minha pergunta')
