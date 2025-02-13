import chromadb
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.schema import Document
from flask import Flask, request, jsonify
import numpy as np
from langchain.prompts import PromptTemplate

# Flask app
app = Flask(__name__)

# Definindo o LLM
llm = HuggingFacePipeline.from_model_id(
    model_id="deepseek/deepseek-xl",  # Caso queira usar DeepSeek, ou use outro modelo HuggingFace
)

# Função de split semântico já foi definida anteriormente
def semantic_split(text, embeddings, threshold=0.7):
    '''Divide o texto semânticamente em chunks baseados na similaridade entre frases'''
    sentences = text.split('. ')
    sentence_embeddings = embeddings.embed_documents(sentences)  
    
    chunks = []
    current_chunk = []
    
    for i in range(len(sentences) - 1):
        current_chunk.append(sentences[i])
        similarity = np.dot(sentence_embeddings[i], sentence_embeddings[i + 1]) / (
            np.linalg.norm(sentence_embeddings[i]) * np.linalg.norm(sentence_embeddings[i + 1])
        )
        
        if similarity < threshold:
            chunks.append('. '.join(current_chunk))
            current_chunk = []
    
    if current_chunk:
        chunks.append('. '.join(current_chunk))
    
    return chunks


# Função para buscar os documentos relevantes e gerar a resposta
def get_answer_from_db(query, db, llm):
    # Buscar os documentos mais relevantes no banco de dados vetorial
    matching_docs = db.similarity_search(query, k=3)
    
    # Inicializando o Prompt para a LLM
    prompt_template = "Com base nos seguintes documentos, responda à pergunta: \n\n{documents}\n\nPergunta: {query}\nResposta:"
    prompt = PromptTemplate(input_variables=["documents", "query"], template=prompt_template)
    
    # Criando a cadeia de perguntas e respostas com os documentos encontrados
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever())
    
    # Gerando a resposta da LLM
    result = qa_chain.run(input={"documents": "\n".join([doc.page_content for doc in matching_docs]), "query": query})
    return result


# API endpoint para receber a pergunta e retornar a resposta
@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    query = data['query']
    
    # Resposta gerada a partir da base de conhecimento
    answer = get_answer_from_db(query, db, llm)
    
    # Retorna a resposta como JSON
    return jsonify({"response": answer})

# Rodar a aplicação
if __name__ == '__main__':
    app.run(debug=True)
