import torch
import os
import getpass
import bs4
import time
import streamlit as st
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline
from langchain_ollama import ChatOllama
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
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import AIMessage, HumanMessage


load_dotenv()
HF_API_TOKEN = os.getenv('HF_TOKEN')
model_class = 'hf_hub' 

# LLM
def model_hf_hub(model='meta-llama/Meta-Llama-3-8B-Instruct', temperature=0.1):
  llm = HuggingFaceHub(
      huggingfacehub_api_token=HF_API_TOKEN,
      repo_id=model,
      model_kwargs={
          'temperature': temperature,
          'return_full_text': False,
          'max_new_tokens': 500,
          'stop': ['<|eot_id|>']
      }
  )
  return llm

def model_ollama(model='phi3', temperature=0.1):
    llm = ChatOllama(
        model=model,
        temperature=temperature,
    )
    return llm

# Indexação de texto no VectorDB
def vectorstorage():
    loader = WebBaseLoader(
            web_paths='https://hotmart.com/pt-br/blog/como-funciona-hotmart')
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, 
                                                chunk_overlap=200, 
                                                add_start_index=True)
    splits = text_splitter.split_documents(docs)

    hf_embeddings = HuggingFaceEmbeddings(
                    model_name='sentence-transformers/all-mpnet-base-v2')

    vectorstore = Chroma.from_documents(documents=splits, 
                                        embedding=hf_embeddings)

    return vectorstore

# Recuperação dos arquivos
def retrieve_text(vectorstore):
    retriever = vectorstore.as_retriever(search_type = 'mmr', 
                                        search_kwargs={'k': 3, 
                                                       'fetch_k': 4})
    return retriever

# Configuração da chain
def config_rag_chain(model_class, retriever):
    # Carregamento da LLM
    if model_class == 'hf_hub':
        llm = model_hf_hub()
    elif model_class == 'ollama':
        llm = model_ollama()

    # Para definição dos prompts (usa o histórico para responder as perguntas seguintes)
    if model_class.startswith('hf'):
        token_s, token_e = '<|begin_of_text|><|start_header_id|>system<|end_header_id|>', '<|eot_id|><|start_header_id|>assistant<|end_header_id|>'
    else:
        token_s, token_e = '', ''

    # Prompt de contextualização
    # consulta -> retriever
    # (consulta, histórico do chat) -> LLM -> consulta reformulada -> retriever
    context_q_system_prompt = 'Given the following chat history and the follow-up question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is.'
    context_q_system_prompt = token_s + context_q_system_prompt
    context_q_user_prompt = 'Question: {input}' + token_e
    context_q_prompt = ChatPromptTemplate.from_messages(
        [
            ('system', context_q_system_prompt),
            MessagesPlaceholder('chat_history'),
            ('human', context_q_user_prompt)
        ]
    )

    # Chain para contextualização
    print('\n\n coisaaa', llm, retriever, context_q_prompt)

    history_aware_retriever = create_history_aware_retriever(llm=llm,
                                                             retriever=retriever,
                                                             prompt=context_q_prompt)
    
    # Prompt para perguntas e respostas (Q&A)
    qa_prompt_template = '''Você é um assistente virtual prestativo e está respondendo perguntas sobre a empresa Hotmart. 
    Use os seguintes pedaços de contexto recuperado para responder à pergunta. 
    Se você não sabe a resposta, apenas diga que não sabe. Mantenha a resposta concisa. 
    Responda em português. \n\n
    Pergunta: {input} \n
    Contexto: {context}'''

    qa_prompt = PromptTemplate.from_template(token_s + qa_prompt_template + token_e)

    # Configurar LLM e Chain para perguntas e respostas (Q&A)
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    return rag_chain
  
vectorstore = vectorstorage()
retriever = retrieve_text(vectorstore)
rag_chain = config_rag_chain(model_class, retriever)




# # Configurações do Streamlit
# st.set_page_config(page_title='Saiba tudo sobre a Hotmart ', page_icon='📚')
# st.title('O que você quer saber sobre a Hotmart?')

# if 'chat_history' not in st.session_state:
#     st.session_state.chat_history = [
#         AIMessage(content='Olá, sou a sua assistente virtual! O que você quer saber sobre a Hotmart?'),
#     ]

# if 'retriever' not in st.session_state:
#     st.session_state.retriever = None

# for message in st.session_state.chat_history:
#     if isinstance(message, AIMessage):
#         with st.chat_message('AI'):
#             st.write(message.content)
#     elif isinstance(message, HumanMessage):
#         with st.chat_message('Human'):
#             st.write(message.content)

# start = time.time()
# user_query = st.chat_input('Digite sua mensagem aqui')

# if user_query is not None and user_query != '':
#     st.session_state.chat_history.append(HumanMessage(content=user_query))

#     with st.chat_message('Human'):
#         st.markdown(user_query)

#     with st.chat_message('AI'):      
#         rag_chain = config_rag_chain(model_class, 
#                                      st.session_state.retriever)

#         result = rag_chain.invoke({'input': user_query,
#                                    'chat_history': st.session_state.chat_history})

#         resp = result['answer']
#         st.write(resp)

#     st.session_state.chat_history.append(AIMessage(content=resp))

# end = time.time()
# print('Tempo: ', end - start)

