from sentence_transformers import SentenceTransformer
import numpy as np
import requests
from bs4 import BeautifulSoup
from chromadb import Client

# Modelo para geração de embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_url(url):
    '''Extrai o texto da página da Hotmart'''
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')
    text = ' '.join([p.get_text() for p in paragraphs])
    return text

def semantic_split(text, threshold=0.7):
    '''Divide o texto semânticamente em chunks baseados na similaridade entre frases'''
    sentences = text.split('. ')
    sentence_embeddings = embedding_model.encode(sentences)
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

def generate_embeddings(text_chunks):
    '''Gera embeddings para os chunks de texto'''
    return embedding_model.encode(text_chunks)

def store_in_vectordb(chunks, embeddings, collection_name='documents'):
    '''Armazena os chunks e embeddings no VectorDB'''
    client = Client()
    collection = client.get_collection(collection_name)
    
    for i, chunk in enumerate(chunks):
        collection.add(
            ids=[f'doc_{i}'],
            embeddings=[embeddings[i]],
            metadatas=[{'text': chunk}]
        )
