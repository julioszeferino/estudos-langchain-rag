from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import Field, BaseModel
from langchain_core.globals import set_debug
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")
if api_key is None:
    raise ValueError("A variável de ambiente ANTHROPIC_API_KEY não está definida.")

# ativando o debug
set_debug(True)


# definindo o formato da saida
# permite que a saida seja validada e estruturada de acordo com 
# o modelo Pydantic
class Destino(BaseModel):
    cidade: str = Field(description="A cidade recomendada para visitar")
    motivo: str = Field(description="O motivo pelo qual a cidade foi recomendada")


parseador = JsonOutputParser(pydantic_object=Destino)

prompt_cidade = PromptTemplate(
    template="""
    Sugira uma cidade dado o meu interesse por {interesse}.
    {formato_de_saida}
    """,
    input_variables=["interesse"],
    partial_variables={
        "formato_de_saida": parseador.get_format_instructions()
    },
)

modelo_claude = ChatAnthropic(
    anthropic_api_key=api_key,
    model="claude-haiku-5-5",
)

# Chain sao uma boa opcao quando queremos encadear varios modelos ou prompts,
# ou quando queremos fazer um fluxo de trabalho mais complexo.
cadeia = prompt_cidade | modelo_claude | parseador

resposta = cadeia.invoke(
    {
        "interesse": "praias"
    }
)

print(resposta)
