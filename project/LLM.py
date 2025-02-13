import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
import streamlit as st

# Carregar variáveis de ambiente
load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Inicializar FastAPI
app = FastAPI()

# Inicializar modelo de embedding
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Conectar ao ChromaDB
vector_db = Chroma(persist_directory='vectorstore', 
                    embedding_function=embeddings)

retriever = vector_db.as_retriever(search_type='similarity',
                                   search_kwargs={'k': 5})

# Configuração do modelo Groq
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name='llama-3.3-70b-versatile', 
    temperature=0,
    top_p=0.9
)

# # Criar um prompt personalizado
# prompt_template = PromptTemplate(
#     template='''
#     Você é um assistente especializado em responder perguntas sobre a Hotmart,
#       e se baseia apenas nos dados presentes no vector database da pasta /vectorstore.
    
#     **Contexto**: 
#     {context}

#     **Pergunta**: 
#     {question}

#     **Instruções**:
#     - Se a resposta não estiver no contexto, diga que não sabe.
#     - Explique de forma clara e objetiva.
#     - Seja breve, mas informativo.

#     **Resposta**:
#     ''',
#     input_variables=['context', 'question']
# )

# Configurar a cadeia de recuperação (QA)
qa_chain = RetrievalQA.from_chain_type(llm=llm, 
                                       chain_type='stuff', 
                                       retriever=retriever,
                                       return_source_documents=True)

# Criar memória para o agente (opcional)
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

# # Criar ferramenta de recuperação
# retrieval_tool = Tool(
#     name='Hotmart Knowledge Base',
#     func=vector_db.similarity_search,
#     description='Busca informações relevantes na base de conhecimento da Hotmart.'
# )

# # Criar agente que usa a ferramenta de busca
# agent = initialize_agent(
#     tools=[qa_chain],
#     llm=llm,
#     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#     memory=memory,
#     verbose=True
# )

# # Modelo para requisição
# class QueryRequest(BaseModel):
#     question: str

# @app.post('/ask')
# def ask_question(query: QueryRequest):
#     '''Recebe uma pergunta, busca no VectorDB e responde com o agente.'''
#     # Recuperar trechos relevantes da base vetorial
#     matching_docs = vector_db.similarity_search(query.question, k=3)
#     context = '\n'.join([doc.page_content for doc in matching_docs])

#     # # Preencher o template com contexto e pergunta
#     # prompt_filled = prompt_template.format(context=context, 
#     #                                        question=query.question)

#     # Chamar a LLM para gerar a resposta
#     response = llm.invoke(prompt_filled)

    # return {'answer': response}


# Interface com Streamlit
st.title('Perguntas sobre a Hotmart')
st.write('Digite sua pergunta abaixo para obter uma resposta baseada no conteúdo.')

# Caixa de entrada para o usuário digitar a pergunta
user_question = st.text_input('Qual é a sua pergunta?', '')

# Se houver uma pergunta, processe a resposta
if user_question:
    # Recuperando resposta do LLM
    resposta = qa_chain.run(user_question)
    st.write(f'Resposta: {resposta}')