from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")

if api_key is None:
    raise ValueError("A variável de ambiente ANTHROPIC_API_KEY não está definida.")


prompt_cidade = PromptTemplate(
    template="""
    Sugira uma cidade dado o meu interesse por {interesse}.
    """,
    input_variables=["interesse"],
)

modelo_claude = ChatAnthropic(
    anthropic_api_key=api_key,
    model="claude-haiku-5-5",
)

# Chain sao uma boa opcao quando queremos encadear varios modelos ou prompts,
# ou quando queremos fazer um fluxo de trabalho mais complexo.
cadeia = prompt_cidade | modelo_claude | StrOutputParser()

resposta = cadeia.invoke(
    {
        "interesse": "praias"
    }
)

print(resposta)
