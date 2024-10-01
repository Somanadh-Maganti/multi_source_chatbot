from langchain_community.llms import HuggingFaceHub

import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "XXXXX"

def content_search(query, content):
    hf = HuggingFaceHub(
        repo_id="mistralai/Mistral-Small-Instruct-2409",
        model_kwargs={"temperature": 0.1, "max_length": 500}
    )

    context = query + "\n" + content
    res = hf.invoke(context)
    answer = res.split(context)[-1].strip()
    return answer