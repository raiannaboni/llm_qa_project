import torch
import os
import getpass
import bs4
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import HuggingFaceHub
from dotenv import load_dotenv
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

load_dotenv()
HF_API_TOKEN = os.getenv('HF_TOKEN')


loader = WebBaseLoader(web_paths = ('https://hotmart.com/pt-br/blog/como-funciona-hotmart',),)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, 
                                               chunk_overlap=200, 
                                               add_start_index=True)
splits = text_splitter.split_documents(docs)

hf_embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')

vectorstore = Chroma.from_documents(documents=splits, embedding=hf_embeddings)

# Recuperação dos arquivos
retriever = vectorstore.as_retriever(search_type='similarity', 
                                     search_kwargs={'k': 6})

model_id = 'meta-llama/Meta-Llama-3-8B-Instruct'


llm = HuggingFaceHub(
      huggingfacehub_api_token=HF_API_TOKEN,
      repo_id=model_id,
      model_kwargs={
          'temperature': 0.2,
          'return_full_text': False,
          'max_new_tokens': 300,
          'stop': ['<|eot_id|>']
      }
  )


# Função para configurar a RAG Chain
def config_rag_chain(llm, retriever):
    # Definição de tokens especiais do modelo
    token_s, token_e = '<|begin_of_text|><|start_header_id|>system<|end_header_id|>', '<|eot_id|><|start_header_id|>assistant<|end_header_id|>'
    
    # Template para Q&A
    qa_prompt_template = '''Você é um assistente virtual prestativo e está respondendo perguntas sobre a empresa Hotmart. 
    Use os seguintes pedaços de contexto recuperados para responder à pergunta. 
    Se você não sabe a resposta ou se ela não está no contexto, apenas diga que não sabe. Mantenha a resposta concisa. 
    Responda em português. \n\n
    Pergunta: {input} \n
    Contexto: {context}'''
    
    # Criando o prompt corretamente
    qa_prompt = PromptTemplate.from_template(token_s + qa_prompt_template + token_e)

    # Criando retriever baseado no histórico
    history_aware_retriever = create_history_aware_retriever(llm=llm, retriever=retriever, prompt=qa_prompt)

    # Criando a cadeia de perguntas e respostas (Q&A)
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Criando a cadeia completa de recuperação (RAG)
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    return rag_chain

# Configuração da RAG Chain
rag_chain = config_rag_chain(llm, retriever)

# Teste
print(rag_chain.invoke({'input': 'O que são produtos digitais?'}))

# # Interface com Streamlit
# st.title('Perguntas sobre a Hotmart')
# st.write('Digite sua pergunta abaixo para obter uma resposta baseada no conteúdo.')

# # Caixa de entrada para o usuário digitar a pergunta
# user_question = st.text_input('Qual é a sua pergunta?')

# # Se houver uma pergunta, processe a resposta
# if user_question:
#     with st.spinner('Gerando resposta...'):
#         resposta = chain_rag.invoke(user_question)  
    
#     st.write('### Resposta:')
#     st.write(resposta)