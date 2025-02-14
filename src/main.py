from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import HuggingFaceHub
import warnings

from repository.vectorDB import create_vector_store, get_hotmart_data, text_splitter
from utils.prompt import prompt_rag
from interface.st_interface import streamlit_interface
warnings.filterwarnings('ignore')

HF_API_TOKEN = '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'

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

def format_docs(docs):
  return '\n\n'.join(doc.page_content for doc in docs)

chain_rag = ({'contexto': retriever | format_docs, 'pergunta': RunnablePassthrough()}
             | prompt_rag()
             | llm
             | StrOutputParser())

streamlit_interface(chain_rag)