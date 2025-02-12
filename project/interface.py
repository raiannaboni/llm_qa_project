import streamlit as st
from fastapi import HTTPException
import requests

# URL do FastAPI
FASTAPI_URL = 'http://127.0.0.1:8000'

# Função para processar o documento
def process_document():
    '''Chama o endpoint do FastAPI para processar o documento'''
    response = requests.post(f'{FASTAPI_URL}/process_document')
    if response.status_code == 200:
        st.success('Documento processado com sucesso!')
    else:
        st.error('Erro ao processar o documento.')

# Função para realizar a busca
def search_answer(query):
    '''Chama o endpoint do FastAPI para buscar a resposta'''
    response = requests.get(f'{FASTAPI_URL}/search', params={'query': query})
    if response.status_code == 200:
        result = response.json()
        st.write(f'Resposta: {result["message"]}')
    else:
        st.error('Erro ao buscar a resposta.')

# Interface do Streamlit
st.title('Sistema de Perguntas e Respostas sobre o Hotmart')

# Botão para processar o documento
if st.button('Processar Documento'):
    process_document()

# Campo de texto para a pergunta
query = st.text_input('Digite sua pergunta:')

# Botão para buscar resposta
if st.button('Buscar Resposta'):
    if query:
        search_answer(query)
    else:
        st.error('Digite uma pergunta para buscar a resposta.')
