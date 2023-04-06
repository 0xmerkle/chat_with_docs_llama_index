from langchain.document_loaders import UnstructuredURLLoader
from scrape_utils import scrape
import tiktoken
import streamlit as st

from llama_index import (
    GPTSimpleVectorIndex,
    ServiceContext,
    LLMPredictor,
    PromptHelper,
)
from langchain import OpenAI
from llama_index.node_parser import SimpleNodeParser

import tiktoken

tokenizer = tiktoken.get_encoding("cl100k_base")


OPENAI_API_KEY = "..."  # your openai api key. either export in your environment or set as environment variable and load before running this script

documents = []


def load_documents_to_gpt_vectorstore(url):
    from llama_index import download_loader

    urls = scrape(url)
    BeautifulSoupWebReader = download_loader("BeautifulSoupWebReader")
    loader = BeautifulSoupWebReader()
    documents = loader.load_data(urls)
    parser = SimpleNodeParser()

    nodes = parser.get_nodes_from_documents(documents)
    llm_predictor = LLMPredictor(
        llm=OpenAI(
            temperature=0, model_name="text-davinci-003", openai_api_key=OPENAI_API_KEY
        )
    )

    # define prompt helper
    # set maximum input size
    max_input_size = 4096
    # set number of output tokens
    num_output = 256
    # set maximum chunk overlap
    max_chunk_overlap = 20
    prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor, prompt_helper=prompt_helper
    )
    index = GPTSimpleVectorIndex(nodes, service_context=service_context)
    # index.save_to_disk("./gpt_index_docs_api_remotion_v2.json")
    index.save_to_disk("./[your_index_name].json")

    return index


def chat(query):
    index = GPTSimpleVectorIndex.load_from_disk("[your_index_name].json")
    response = index.query(query)
    print(response)
    return response


st.header("Docs")


doc_input = st.text_input("paste documentation url")


if st.button("load documents"):
    st.markdown(load_documents_to_gpt_vectorstore(doc_input))

user_input = st.text_input("ask about the docs")
if st.button("Ask"):
    st.markdown(chat(user_input))


# https://shotstack.io/docs/api/#tocs_edit
# https://shotstack.io/docs/api/#shotstack-edit,https://python.langchain.com/en/latest/modules/agents.html,https://python.langchain.com/en/latest/modules/agents/getting_started.html,https://python.langchain.com/en/latest/modules/agents/tools.html,
