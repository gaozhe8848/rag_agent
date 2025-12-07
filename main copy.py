from langchain.agents import create_agent
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import JSONLoader
from langchain.tools import tool

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
model = ChatOpenAI(model="gpt-5.1")


vector_store = InMemoryVectorStore(embeddings)

# # Load and chunk contents of the blog
# loader = WebBaseLoader(
#     web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
#     bs_kwargs=dict(
#         parse_only=bs4.SoupStrainer(
#             class_=("post-content", "post-title", "post-header")
#         )
#     ),
# )
# docs = loader.load()

dataset_loader = JSONLoader(
    file_path='./data/remedy_data.json',
    # Drill down into the list of entries
    jq_schema='.entries[].values',  
    # Use the detailed text as the main content
    content_key='Detailed Description',
    # (Optional) Add metadata extraction logic here if needed
    text_content=False
)

# --- Configuration 2: The API Guide Loader ---
# This loads the ENTIRE guide as a single context document
def metadata_func_guide(record: dict, metadata: dict) -> dict:
    metadata["document_type"] = "guide"
    return metadata

guide_loader = JSONLoader(
    file_path='./data/remedy_guide.json',
    # Select the root (.) to pass the entire structure
    jq_schema='.', 
    # We do NOT specify content_key because we want the whole JSON object stringified
    # Setting text_content=False keeps the structure as a valid dict/string
    text_content=False,
    metadata_func=metadata_func_guide
)

# --- Execution ---

print("Loading documents...")

# 1. Load the Ticket Data (Multiple Docs)
data_docs = dataset_loader.load()
print(f"✅ Loaded {len(data_docs)} ticket log entries.")

# 2. Load the Guide (Single Doc)
guide_docs = guide_loader.load()
print(f"✅ Loaded {len(guide_docs)} guide document.")

# 3. Combine for RAG
docs = guide_docs + data_docs

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# Index chunks
_ = vector_store.add_documents(documents=all_splits)

# Construct a tool for retrieving context
@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

tools = [retrieve_context]
# If desired, specify custom instructions
prompt = (
    "System: You are an IT Support Assistant.\n\n"
    "Context: I am providing you with two things:\n\n"
    "A Data Guide explaining the BMC Remedy API structure.\n\n"
    "A JSON Dataset containing work logs for multiple tickets.\n\n"
    "Use this tool to answer user's query."
)
agent = create_agent(model, tools, system_prompt=prompt)

def main():
    query = (
    # "What is the standard method for Task Decomposition?\n\n"
    # "Once you get the answer, look up common extensions of that method."
    "Provide a chronological summary of what happened with ticket INC000000987655. Distinguish between automated system checks and human actions."
    )
    for event in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",
    ):
        event["messages"][-1].pretty_print()

if __name__ == "__main__":
    main()
