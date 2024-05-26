# memory_prompt_template is a string that contains the template for the memory prompt
from prompt_templates import memory_prompt_template

# LLmChain is a class that creates the language model chain
from langchain.chains import LLMChain

# RetrievalQA is a class that creates the retrieval question and answer
from langchain.chains.retrieval_qa.base import RetrievalQA

# HuggingFaceInstructEmbeddings is a class that creates the embeddings
from langchain_community.embeddings import HuggingFaceInstructEmbeddings

# ConversationBufferWindowMemory is a class that creates the conversation buffer window memory
from langchain.memory import ConversationBufferWindowMemory

# PromptTemplate is a class that creates the prompt template
from langchain.prompts import PromptTemplate

# CTransformers is a class that creates the language model chain with the specified model path, model type, and model config
from langchain_community.llms import CTransformers

# Chroma is a class that creates the vector stores
from langchain_community.vectorstores import Chroma

# Ollama is a class that creates the language model chain with the specified model path
from langchain_community.llms import Ollama

# PersistentClient is a class that creates the persistent client
from operator import itemgetter

# load_config is a function that loads the configuration from the config.yaml file
from utils import load_config

# chromadb is a module that contains the chromadb
import chromadb

config = load_config()

# load_ollama_model is a function that loads the language model chain with the specified model path
def load_ollama_model():
    llm = Ollama(model=config["ollama_model"])
    return llm

# create_llm is a function that creates the language model chain with the specified model path, model type, and model config
def create_llm(model_path = config["ctransformers"]["model_path"]["large"], model_type = config["ctransformers"]["model_type"], model_config = config["ctransformers"]["model_config"]):
    llm = CTransformers(model=model_path, model_type=model_type, config=model_config)
    return llm

# create_embeddings is a function that creates the embeddings with the specified embeddings path
def create_embeddings(embeddings_path = config["embeddings_path"]):
    return HuggingFaceInstructEmbeddings(model_name=embeddings_path)

# create_chat_memory is a function that creates the chat memory with the specified chat history
def create_chat_memory(chat_history):
    return ConversationBufferWindowMemory(memory_key="history", chat_memory=chat_history, k=3)

# create_prompt_from_template is a function that creates the prompt from the specified template
def create_prompt_from_template(template):
    return PromptTemplate.from_template(template)

# create_llm_chain is a function that creates the language model chain with the specified language model and chat prompt
def create_llm_chain(llm, chat_prompt):
    return LLMChain(llm=llm, prompt=chat_prompt)
    
# load_normal_chain is a function that loads the normal language model chain
def load_normal_chain():
    return chatChain()

# load_vectordb is a function that loads the vector database with the specified embeddings
def load_vectordb(embeddings):
    persistent_client = chromadb.PersistentClient(config["chromadb"]["chromadb_path"])

    langchain_chroma = Chroma(
        client=persistent_client,
        collection_name=config["chromadb"]["collection_name"],
        embedding_function=embeddings,
    )

    return langchain_chroma

# load_retrieval_chain is a function that loads the retrieval chain with the specified language model and vector database
def load_retrieval_chain(llm, vector_db):
    return RetrievalQA.from_llm(llm=llm, retriever=vector_db.as_retriever(search_kwargs={"k": config["chat_config"]["number_of_retrieved_documents"]}), verbose=True)


# chatChain is a class that creates the chat chain
class chatChain:

    def __init__(self):
        llm = create_llm()
        #llm = load_ollama_model()
        chat_prompt = create_prompt_from_template(memory_prompt_template)
        self.llm_chain = create_llm_chain(llm, chat_prompt)

    def run(self, user_input, chat_history):
        return self.llm_chain.invoke(input={"human_input" : user_input, "history" : chat_history} ,stop=["Human:"])["text"]