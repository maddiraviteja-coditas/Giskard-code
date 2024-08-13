from giskard.llm.client.openai import OpenAIClient
from langchain import FAISS, PromptTemplate
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import openai
import giskard
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

groq_client = openai.OpenAI(
    api_key ="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",
    base_url="https://api.groq.com/openai/v1"         
    )
giskard.llm.set_llm_api("openai")
oc = OpenAIClient(model="llama3-70b-8192", client=groq_client)
giskard.llm.set_default_client(oc)
# giskard.llm.embeddings.set_default_embedding(HuggingFaceEmbeddings())

# Prepare vector store (FAISS) with IPPC report
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, add_start_index=True)
loader = PyPDFLoader("https://www.ipcc.ch/report/ar6/syr/downloads/report/IPCC_AR6_SYR_LongerReport.pdf")
db = FAISS.from_documents(loader.load_and_split(text_splitter),HuggingFaceEmbeddings())

# Prepare QA chain
PROMPT_TEMPLATE = """You are the Climate Assistant, a helpful AI assistant made by Giskard.
Your task is to answer common questions on climate change.
You will be given a question and relevant excerpts from the IPCC Climate Change Synthesis Report (2023).
Please provide short and clear answers based on the provided context. Be polite and helpful.

Context:
{context}

Question:
{question}

Your answer:
"""

llm = ChatGroq(model="gpt-3.5-turbo-instruct", temperature=0)
prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["question", "context"])
climate_qa_chain = RetrievalQA.from_llm(llm=llm, retriever=db.as_retriever(), prompt=prompt)

def model_predict(df: pd.DataFrame):
    """Wraps the LLM call in a simple Python function.

    The function takes a pandas.DataFrame containing the input variables needed
    by your model, and must return a list of the outputs (one for each row).
    """
    return [climate_qa_chain.invoke({"query": question}) for question in df["question"]]


# Don’t forget to fill the `name` and `description`: they are used by Giskard
# to generate domain-specific tests.
giskard_model = giskard.Model(
    model=model_predict,
    model_type="text_generation",
    name="Climate Change Question Answering",
    description="This model answers any question about climate change based on IPCC reports",
    feature_names=["question"],
)
examples = [
    "According to the IPCC report, what are key risks in the Europe?",
    "Is sea level rise avoidable? When will it stop?",
]
giskard_dataset = giskard.Dataset(pd.DataFrame({"question": examples}), target=None)
                                  
report = giskard.scan(giskard_model, giskard_dataset, only="hallucination")
report.to_html("ipcc_scan_report.html")
report.to_json("ipcc_scan_report.json")