import chromadb
import requests
import numpy as np
from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings

# Coletar texto 
def extract_text_from_url(url):
    '''Extrai o texto da página da Hotmart'''
    response = requests.get(url)   
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')
    text = ' '.join([p.get_text() for p in paragraphs])
    return text

# Separar textos
def split_text(text, chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    text_chunks = text_splitter.split_text(text)

    return text_chunks

# Separar textos pela semântica
def semantic_split(text, embeddings, threshold=0.7):
    '''Divide o texto semânticamente em chunks baseados na similaridade entre frases'''
    sentences = text.split('. ')
    sentence_embeddings = embeddings.embed_documents(sentences)  
    
    chunks = []
    current_chunk = []
    
    for i in range(len(sentences) - 1):
        current_chunk.append(sentences[i])
        similarity = np.dot(sentence_embeddings[i], sentence_embeddings[i + 1]) / (np.linalg.norm(sentence_embeddings[i]) * np.linalg.norm(sentence_embeddings[i + 1]))
        
        if similarity < threshold:
            chunks.append('. '.join(current_chunk))
            current_chunk = []
    
    if current_chunk:
        chunks.append('. '.join(current_chunk))
    
    return chunks



url = 'https://hotmart.com/pt-br/blog/como-funciona-hotmart'
hotmart_text = extract_text_from_url(url)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") 


hotmart_text_splitted = semantic_split(hotmart_text, embeddings)



# 🔹 Convertendo cada string em um objeto Document
documents = [Document(page_content=chunk) for chunk in hotmart_text_splitted]


db = Chroma.from_documents(documents, embeddings)


query = "o que é a hotmart?"
matching_docs = db.similarity_search(query,k=3)


