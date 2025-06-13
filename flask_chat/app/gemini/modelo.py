# app/gemini/agente.py

from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)
juiz = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3, google_api_key=api_key)

# Função auxiliar
def ler_arquivo(nome_arquivo) -> str:
    try:
        with open(nome_arquivo, 'r', encoding='utf-8') as arquivo:
            return arquivo.read()
    except Exception as erro:
        return f"Erro ao ler o arquivo: {erro}"

tools = [
    Tool(
        name="ler_arquivo",
        func=ler_arquivo,
        description="Lê o conteúdo de um arquivo de texto para obter mais informações."
    )
]

system_prompt = PromptTemplate.from_template("""
Você é um especialista na luta sobre o gorila César e os 100 homens. Sua tarefa é analisar o conteúdo de arquivos e responder de forma completa e inteligente a pergunta do usuário.
Não se esqueça que tem que responder no mesmo idioma em que a pessoa perguntou. Não utiliza o nome dos arquivos para responder, apenas o conteúdo dos arquivos. Resposta completa e se preciso diga um pouco quem são cada um dos homens.
Use a ferramenta: {tool_names}

Pergunta do usuário: {input}
""")

prompt_juiz = '''
Você é um avaliador imparcial. Sua tarefa é revisar a resposta de um tutor de IA para uma pergunta de um usuario que quer saber sobre a batalha do Gorila César contra os 100 homens.

Critérios:
- A resposta está tecnicamente correta?
- Está clara para o nível médio técnico?

Se a resposta for boa, diga “✅ Aprovado” e explique por quê.
Se tiver problemas, diga “⚠️ Reprovado” e proponha uma versão melhorada.
'''

memory = ConversationBufferMemory(memory_key="chat_history")

agent = initialize_agent(
    llm=llm,
    tools=tools,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    prompt=system_prompt.partial(system_message=system_prompt),
    memory=memory,
    verbose=False
)

# Caminhos dos arquivos
# Obter o diretório do arquivo atual (modelo.py)
script_dir = os.path.dirname(__file__)

# Construir o caminho para a pasta 'gorila-arquivos'
# Assumindo que 'gorila-arquivos' está no mesmo nível de 'modelo.py'
pasta = os.path.join(script_dir, "gorila-arquivos")

# Agora, liste os arquivos dentro dessa pasta
arquivos = [arquivo for arquivo in os.listdir(pasta) if arquivo.endswith(".txt")]
caminhos = [os.path.join(pasta, arquivo) for arquivo in arquivos]

def avaliar_resposta(pergunta, resposta_tutor):
    mensagens = [
        SystemMessage(content=prompt_juiz),
        HumanMessage(content=f"Pergunta do aluno: {pergunta}\n\nResposta do tutor: {resposta_tutor}")
    ]
    return juiz.invoke(mensagens).content

def responder_com_agente(pergunta: str) -> tuple[str, str]:
    prompt_usuario = f"Analise os seguintes arquivos: {', '.join(caminhos)}. Pergunta: {pergunta}"
    resposta = agent.run(prompt_usuario)
    avaliacao = avaliar_resposta(prompt_usuario, resposta)
    return resposta.strip(), avaliacao.strip()
