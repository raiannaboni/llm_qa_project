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
          'temperature': 0.1,
          'return_full_text': False,
          'max_new_tokens': 500,
          # 'stop': ['<|eot_id|>']
      }
  )


template_rag = '''
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
Você é uma assistente virtual prestativa e está respondendo perguntas sobre a empresa Hotmart baseado nos textos do vectorstorage.
Use os pedaços de contexto recuperados para responder às perguntas.
Se você não sabe a resposta, apenas diga que não sabe. Mantenha a resposta concisa.
Responda em português.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Pergunta: {pergunta}
Contexto: {contexto}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
'''

prompt_rag = PromptTemplate(
    input_variables=['contexto', 'pergunta'],
    template=template_rag,
)

def format_docs(docs):
  return '\n\n'.join(doc.page_content for doc in docs)

chain_rag = ({'contexto': retriever
             | format_docs, 'pergunta': RunnablePassthrough()}
             | prompt_rag
             | llm
             | StrOutputParser())

# Teste
print(chain_rag.invoke('o que sao produtos digtais?'))

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