from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from giskard.llm.client.openai import OpenAIClient
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from giskard.rag import generate_testset, KnowledgeBase
import giskard
from langchain_chroma import Chroma
import openai
import pandas as pd
from giskard.rag import evaluate
from test_set import chain
import sys
import os
from dotenv import load_dotenv
load_dotenv()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, add_start_index=True)
loader = PyPDFLoader(sys.argv[1])
chunks = loader.load_and_split(text_splitter)
db = Chroma.from_documents(loader.load_and_split(text_splitter), HuggingFaceEmbeddings())

groq_client = openai.OpenAI(
    api_key = os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"         
    )
giskard.llm.set_llm_api("openai")
oc = OpenAIClient(model="llama3-70b-8192", client=groq_client)
giskard.llm.set_default_client(oc)

knolwdge_base = []
for chunk in chunks:
    sample = {}
    meta_data = chunk.metadata
    contet_ = chunk.page_content
    sample["meta_data"] = meta_data
    sample["content"] = contet_
    knolwdge_base.append(sample)
kb = pd.DataFrame(knolwdge_base)
giskard.llm.embeddings.set_default_embedding(HuggingFaceEmbeddings())
knowledge_base = KnowledgeBase.from_pandas(kb)
testset = generate_testset(
    knowledge_base, 
    num_questions=20,
    language='en',  # optional, we'll auto detect if not provided
    agent_description="A chatbot from github to help beginers to learn", # helps generating better questions
)
testset.save(sys.argv[2])

def get_answer_fn(question: str, history=None) -> str:
    """A function representing your RAG agent."""
    # Format appropriately the history for your RAG agent
    messages = history if history else []
    messages.append({"role": "user", "content": question})

    # Get the answer
    answer = chain(messages[-1].get("content"),db)  # could be langchain, llama_index, etc.

    return answer.get("result")

report = evaluate(get_answer_fn, testset=testset, knowledge_base=knowledge_base, llm_client=oc)
report.to_html(sys.argv[3])

# arg1: pdf = rag_source
# arg2: jsonl = test_set_filename
# arg3: html = report_file_name