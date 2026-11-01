from anthropic import Anthropic
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")

numero_dias = 7
numero_criancas = 2
atividade = "musica"

prompt = f"""
    Crie um roteiro de viagem de {numero_dias} dias, para uma família 
    com {numero_criancas} crianças, que gosta de {atividade}.
"""

cliente = Anthropic(api_key=api_key)

resposta = cliente.messages.create(
    model="claude-haiku-5-5",
    max_tokens=1000,
    system="Você e um assistente de roteiro de viagens.",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ],
)

if resposta.stop_reason == "max_tokens":
    print("Resposta cortada — aumente max_tokens")

for bloco in resposta.content:
    if bloco.type == "text":
        print(bloco.text)
    



