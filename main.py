import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.globals import set_debug
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

load_dotenv()
api_key = os.getenv('ANTHROPIC_API_KEY')
if api_key is None:
    raise ValueError(
        'A variável de ambiente ANTHROPIC_API_KEY não está definida.'
    )

# ativando o debug
set_debug(True)


# definindo o formato da saida
# permite que a saida seja validada e estruturada de acordo com
# o modelo Pydantic
class Destino(BaseModel):
    cidade: str = Field(description='A cidade recomendada para visitar')
    motivo: str = Field(
        description='O motivo pelo qual a cidade foi recomendada'
    )


class Restaurantes(BaseModel):
    cidade: str = Field(description='A cidade recomendada para visitar')
    restaurantes: str = Field(
        description='Sugestões de restaurantes na cidade recomendada'
    )


parseador_destino = JsonOutputParser(pydantic_object=Destino)
parseador_restaurantes = JsonOutputParser(pydantic_object=Restaurantes)

prompt_cidade = PromptTemplate(
    template="""
    Sugira uma cidade dado o meu interesse por {interesse}.
    {formato_de_saida}
    """,
    input_variables=['interesse'],
    partial_variables={
        'formato_de_saida': parseador_destino.get_format_instructions()
    },
)

prompt_restaurantes = PromptTemplate(
    template="""
    Sugira restaurantes na cidade {cidade}.
    {formato_de_saida}
    """,
    input_variables=['cidade'],
    partial_variables={
        'formato_de_saida': parseador_restaurantes.get_format_instructions()
    },
)

prompt_cultural = PromptTemplate(
    template="""
    Sugira atividades culturais na cidade {cidade}.
    """,
    input_variables=['cidade'],
)


modelo_claude = ChatAnthropic(
    anthropic_api_key=api_key,
    model='claude-haiku-5-5',
)

# Chain sao uma boa opcao quando queremos encadear varios modelos ou prompts,
# ou quando queremos fazer um fluxo de trabalho mais complexo.

# o resultado de uma chain pode ser passado como entrada para outra chain,
# permitindo a criação de fluxos de trabalho mais complexos e sofisticados.

cadeia_1 = prompt_cidade | modelo_claude | parseador_destino
cadeia_2 = prompt_restaurantes | modelo_claude | parseador_restaurantes
cadeia_3 = prompt_cultural | modelo_claude | StrOutputParser()

cadeia = cadeia_1 | cadeia_2 | cadeia_3

resposta = cadeia.invoke({'interesse': 'praias'})

print(resposta)
