from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

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