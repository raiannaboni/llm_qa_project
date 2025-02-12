from fastapi import FastAPI
from utils import extract_text_from_url, semantic_split, generate_embeddings, store_in_vectordb
from retrieval import search_vectordb

# Iniciar o FastAPI
app = FastAPI()

# URL do documento que será processado
URL = 'https://hotmart.com/pt-br/blog/como-funciona-hotmart'

# Função para processar o documento
def process_document():
    '''Processa o documento completo'''
    print('Extraindo texto da URL...')
    text = extract_text_from_url(URL)  
    
    print('Dividindo o texto semânticamente...')
    chunks = semantic_split(text)  
    
    print('Gerando embeddings para os chunks...')
    embeddings = generate_embeddings(chunks)  

    print('Armazenando os dados no VectorDB...')
    store_in_vectordb(chunks, embeddings)  
    
    print('Documento processado e armazenado com sucesso!')

# Função para realizar uma busca no VectorDB
def perform_search(query):
    '''Realiza uma busca no VectorDB baseado na consulta e usa DeepSeek para gerar resposta'''
    print(f'Buscando resultados para: {query}')
    response = search_vectordb(query)  # Função de busca que agora usa DeepSeek
    print(f'Resposta gerada: {response}')

# Endpoint para processar e armazenar o documento
@app.post('/process_document')
def process_and_store():
    '''Extrai texto, gera embeddings e armazena no VectorDB'''
    process_document()
    return {'message': 'Documento processado e armazenado!'}

# Endpoint para realizar a busca no VectorDB
@app.get('/search')
def search(query: str):
    '''Busca no VectorDB baseado na consulta e gera uma resposta com DeepSeek'''
    perform_search(query)
    return {'message': 'Busca realizada com sucesso!'}

# Chamada de função para processar o documento e armazenar no banco
if __name__ == '__main__':
    print('Iniciando o processo de extração e armazenamento de documento...')
    process_document()  # Processa e armazena o documento
    
    # Teste
    example_query = 'Como funciona o Hotmart?'
    perform_search(example_query)
