from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
import os

os.environ['HUGGINGFACEHUB_API_TOKEN']="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"

loader=PyPDFDirectoryLoader("./documents")
documents=loader.load()
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
final_documents=text_splitter.split_documents(documents)

huggingface_embeddings=HuggingFaceBgeEmbeddings(
    model_name="sentence-transformers/all-MiniLM-l6-v2",      
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings':True}

)

vectorstore=FAISS.from_documents(final_documents[:100],huggingface_embeddings)
retriever=vectorstore.as_retriever(search_type="similarity",search_kwargs={"k":3})

llm=Ollama(model="gemma:2b")

prompt_template="""You are a helpful book writing assistant.
You may use the following piece of context to answer the question asked.

{context}
Question:{question}

Helpful Answers:
 """

prompt=PromptTemplate(template=prompt_template,input_variables=["context","question"])

retrievalQA=RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt":prompt}
)

query="""Explain in detail in about 200 words, what are base isolation systems?"""

result = retrievalQA.invoke({"query": query})
print(result['result'])


