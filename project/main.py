import os
from langchain.prompts import PromptTemplate
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

def get_hotmart_data():
    loader = WebBaseLoader(web_paths = ('https://hotmart.com/pt-br/blog/como-funciona-hotmart',),)
    hotmart_data = loader.load()
    return hotmart_data

def text_splitter(hotmart_data):
    splitter_config = RecursiveCharacterTextSplitter(chunk_size=1000, 
                                               chunk_overlap=200, 
                                               add_start_index=True)
    splitted_texts = splitter_config.split_documents(hotmart_data)
    return splitted_texts

def create_vector_store(splitted_texts):
    hf_embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
    vectorstore = Chroma.from_documents(documents=splitted_texts, embedding=hf_embeddings)
    return vectorstore

hotmart_data = get_hotmart_data()
splitted_texts = text_splitter(hotmart_data)

vectorstore = create_vector_store(splitted_texts)
retriever = vectorstore.as_retriever(search_type='similarity', 
                                     search_kwargs={'k': 6})

llm = HuggingFaceHub(
    huggingfacehub_api_token=HF_API_TOKEN,
    repo_id='meta-llama/Meta-Llama-3-8B-Instruct',
    model_kwargs={
        'temperature': 0.1,
        'max_new_tokens': 500,
        'return_full_text': False, 
        'do_sample': False,
    }
)


template_rag = '''
Você é uma assistente virtual prestativa e está respondendo perguntas sobre a empresa Hotmart baseado nos textos do vectorstorage.
Use os pedaços de contexto recuperados para responder às perguntas.
Se você não sabe a resposta, apenas diga que não sabe. Mantenha a resposta concisa.
Responda em português.

### Contexto:
{contexto}

### Pergunta:
{pergunta}

### Resposta:
'''

prompt_rag = PromptTemplate(
    input_variables=['contexto', 'pergunta'],
    template=template_rag,
)

def format_docs(docs):
  return '\n\n'.join(doc.page_content for doc in docs)

chain_rag = ({'contexto': retriever | format_docs, 'pergunta': RunnablePassthrough()}
             | prompt_rag
             | llm
             | StrOutputParser())

# Streamlit
def interface():
    st.title('Perguntas sobre a Hotmart')
    st.write('Digite sua pergunta abaixo para obter uma resposta baseada no conteúdo.')

    user_question = st.text_input('Qual é a sua pergunta?')

    if user_question:
        with st.spinner('Gerando resposta...'):
            resposta = chain_rag.invoke(user_question)  
        
        st.write('### Resposta:')
        st.write(resposta)