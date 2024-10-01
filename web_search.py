from langchain.utilities import GoogleSearchAPIWrapper
import os


os.environ["HUGGINGFACEHUB_API_TOKEN"] = "XXXX"
os.environ["GOOGLE_API_KEY"] = "XXXX"
os.environ["GOOGLE_CSE_ID"] = "XXXX"

from langchain_community.llms import HuggingFaceHub

search = GoogleSearchAPIWrapper()

prompt_template = """
Role: You are a Question answering agent who will answer the user query in a single line from available context.

query : {my_query}

context : {my_context}

"""

def safe_search(query):
    result = search.run(query)
    if not result:
        return "No results found"
    return result

def web_search(query):
    hf = HuggingFaceHub(
        repo_id="mistralai/Mistral-Small-Instruct-2409",
        model_kwargs={"temperature": 0.1, "max_length": 500}
    )


    results = safe_search(query)
    prompt = prompt_template.format(my_query=query, my_context=str(results))
    res = hf.invoke(prompt)
    answer = res.split(prompt)[-1].strip()
    return answer
