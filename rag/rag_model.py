
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import os
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader



os.environ["OPENAI_API_KEY"]  = 'sk-Your OPEN AI KEY'

embedding=OpenAIEmbeddings(model="text-embedding-3-large")

# Make sure you have download the MedPub using download.py

from langchain.document_loaders import JSONLoader

def metadata_func(record: dict, metadata: dict) -> dict:
    # Define the metadata extraction function.
    metadata["year"] = record.get("pub_date").get('year')
    metadata["month"] = record.get("pub_date").get('month')
    metadata["day"] = record.get("pub_date").get('day')
    metadata["title"] = record.get("article_title")
    
    return metadata

loader = JSONLoader(
    file_path='/data/pubmed_article_april-2024.json',
    jq_schema='.[]',
    content_key='article_abstract',
    metadata_func=metadata_func)
data = loader.load()
print(f"{len(data)} pubmed articles are loaded!")
data[1]


from langchain.text_splitter import TokenTextSplitter,CharacterTextSplitter
text_splitter = TokenTextSplitter(chunk_size=128, chunk_overlap=64)
chunks = text_splitter.split_documents(data)
print(f"{len(data)} pubmed articles are converted to {len(chunks)} text fragments!")
chunks[0]

def create_vectordb(file_dir:str="./rag/raw_data/", embedding=embedding):
    persist_directory = './rag/vectordb'

    # Here we test with OpenAIEmbeddings first
    vectordb = Chroma.from_documents(documents=chunks, 
                                    embedding=embedding,
                                    persist_directory=persist_directory)
    vectordb.persist()
    return vectordb


def load_vectordb(persist_directory:str="./data/vectordb/", embedding=embedding):
    vectordb = Chroma(persist_directory=persist_directory,
                    embedding_function=embedding)
    return vectordb

def create_prompt():

    prompt_template = """
    Your are a medical assistant for question-answering tasks. Answer the Question using the provided Contex only. Your answer should be in your own words and be no longer than 128 words. \n\n Context: {context} \n\n Question: {question} \n\n Answer:
    """

    prompt = PromptTemplate(template=prompt_template, 
                            input_variables=["context", "question"])
    return prompt

def rag_chat(query:str):

    llm = ChatOpenAI(model = "gpt-3.5-turbo", temperature=0, max_tokens=50)

    if not os.path.exists("./rag/vectordb"):
        print("Vector DB not found. Creating new one...")
        vectordb = create_vectordb()
        
    else:
        vectordb = load_vectordb()
        
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    llama_prompt = create_prompt()
    chain_type_kwargs = {"prompt": llama_prompt}

    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type="stuff",
                                           retriever=retriever,
                                           chain_type_kwargs=chain_type_kwargs, 
                                           return_source_documents=True)
    
    llm_response = qa_chain(query)

    return llm_response