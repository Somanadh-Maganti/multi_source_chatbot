
from langchain_community.llms import HuggingFaceHub
import faiss , os
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

from uuid import uuid4
from langchain_core.documents import Document


import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "XXX"

import numpy as np
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter

prompt_template = """
Role: You are a Question answering agent who will answer the user query in a single line from available context.

Please provide the answer in single line without including any other source details

query : {my_query}

context : {my_context}

"""



def vector_store_creation(uploaded_file):

    reader = PyPDF2.PdfReader(uploaded_file)
    text = "".join(page.extract_text() for page in reader.pages)

    # Step 3: Chunk the text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=20)
    chunks = text_splitter.split_text(text)
    documents = [Document(page_content=chunk) for chunk in chunks]

    # Step 4: Initialize FAISS index
    index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
    
    # Step 5: Create vector store
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    
    # Step 6: Create UUIDs for each chunk
    uuids = [str(uuid4()) for _ in range(len(documents))]

    # Step 7: Add documents (chunks) to the vector store
    vector_store.add_documents(documents=documents, ids=uuids)
    return vector_store

def document_qa(query, uploaded_file):

    vector_store = vector_store_creation(uploaded_file)
    hf = HuggingFaceHub(
        repo_id="mistralai/Mistral-Small-Instruct-2409",
        model_kwargs={"temperature": 0.1, "max_length": 500}
    )

    retriever=vector_store.as_retriever(search_type="similarity",search_kwargs={"k":2})
    retrieved_docs = retriever.invoke(query)

    page_contents = "\n\n".join([doc.page_content for doc in retrieved_docs])
    prompt = prompt_template.format(my_query=query, my_context=str(page_contents))
    res = hf.invoke(prompt)
    answer = res.split(prompt)[-1].strip()
    return answer

