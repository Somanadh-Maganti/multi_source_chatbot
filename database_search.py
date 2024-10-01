from langchain_community.llms import HuggingFaceHub
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

from langchain.vectorstores import FAISS

directory = r"XXXXXXXXXX"



vector_store = FAISS.load_local(
    directory, embeddings, allow_dangerous_deserialization=True
)

import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "XXX"

def database_search(query):
    retriever1=vector_store.as_retriever(search_type="similarity",search_kwargs={"k":2})
    contents = retriever1.invoke(query)

    hf = HuggingFaceHub(
        repo_id="mistralai/Mistral-Small-Instruct-2409",
        model_kwargs={"temperature": 0.1, "max_length": 5000}
    )

    context = query + "\n" + "answer from below content "+ "\n"  + "\n\n".join([doc.page_content for doc in contents])
    res = hf.invoke(context)
    answer = res.split(context)[-1].strip()
    return answer

