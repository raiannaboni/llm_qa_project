import getpass
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from dotenv import load_dotenv
import streamlit as st
import warnings
warnings.filterwarnings('ignore')
from langchain import hub
from langchain_community.llms import HuggingFaceHub


load_dotenv()
HF_API_TOKEN = os.getenv('HF_TOKEN')

# Load and chunk contents of the blog
loader = WebBaseLoader(web_paths = ('https://hotmart.com/pt-br/blog/como-funciona-hotmart',),)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, 
                                               chunk_overlap=200, 
                                               add_start_index=True)
splits = text_splitter.split_documents(docs)

hf_embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')

vectorstore = Chroma.from_documents(documents=splits, embedding=hf_embeddings)

# Define prompt for question-answering
prompt = hub.pull("rlm/rag-prompt")

model_id = 'meta-llama/Meta-Llama-3-8B-Instruct'
llm = HuggingFaceHub(
      huggingfacehub_api_token=HF_API_TOKEN,
      repo_id=model_id,
      model_kwargs={
          'temperature': 0.1,
          'return_full_text': False,
          'max_new_tokens': 500,
          # 'stop': ['<|eot_id|>']
      }
  )

# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Define application steps
def retrieve(state: State):
    retrieved_docs = vectorstore.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

response = graph.invoke({"question": "o que e hotmart?"})
print(response["answer"])

