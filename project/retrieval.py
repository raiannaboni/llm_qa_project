import chromadb
from openai import OpenAI

# # Sua chave de API do DeepSeek
# API_KEY = 'sk-7c7103f69a834039a6051f74e6d0e7f5'

# # Inicializa o client do DeepSeek com a chave da API
# deepseek_client = OpenAI(api_key=API_KEY,
#                          base_url='https://api.deepseek.com/')

# in disk client
chroma_client = chromadb.PersistentClient(path='./vectorstore')

# Função para armazenar documentos no VectorDB
def store_documents_in_vectordb(documents):
    '''Armazena documentos vetorizados no VectorDB utilizando o Chromadb'''
    collection = chroma_client.create_collection('documents')
    
    for doc in documents:
        # Suponha que 'doc' seja um dicionário com 'text' e 'metadata' como chaves
        vector = deepseek_client.embed(doc['text'])  # Usando o DeepSeek para gerar o vetor do documento
        collection.add(
            documents=[doc['text']],
            metadatas=[doc['metadata']],
            embeddings=[vector]
        )

# Função para buscar documentos no VectorDB e gerar a resposta com o modelo DeepSeek
def search_vectordb(query):
    '''Busca no VectorDB baseado na consulta e gera a resposta com o modelo DeepSeek'''
    # Consulta ao VectorDB (Chromadb) para encontrar os documentos mais relevantes
    collection = chroma_client.get_collection('documents')
    results = collection.query(query=query, n_results=3)  # Retorna 3 documentos mais relevantes
    documents = results['documents']
    
    # Agora, usamos o modelo DeepSeek para gerar a resposta com os documentos recuperados
    response = deepseek_client.query(query, documents)  # Gera a resposta com o DeepSeek
    
    return response
