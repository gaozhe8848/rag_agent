from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.tools import tool
from langchain.agents import create_agent

# 1. Setup Models
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
model = ChatOpenAI(model="gpt-5.1", temperature=0)

vector_store = InMemoryVectorStore(embeddings)

# --- Loaders (Same as before) ---
def metadata_func_dataset(record: dict, metadata: dict) -> dict:
    metadata["incident_number"] = record.get("Incident Number")
    metadata["submitter"] = record.get("Submitter")
    metadata["log_type"] = record.get("Work Log Type")
    metadata["document_type"] = "ticket_log"
    return metadata

def metadata_func_guide(record: dict, metadata: dict) -> dict:
    metadata["document_type"] = "guide"
    return metadata

dataset_loader = JSONLoader(
    file_path='./data/remedy_data.json',
    jq_schema='.entries[].values',  
    text_content=False, 
    metadata_func=metadata_func_dataset
)

guide_loader = JSONLoader(
    file_path='./data/remedy_guide.json',
    jq_schema='.', 
    text_content=False,
    metadata_func=metadata_func_guide
)

# --- Loading & Indexing ---
print("Loading documents...")
data_docs = dataset_loader.load()
guide_docs = guide_loader.load()
docs = guide_docs + data_docs

print(f" Loaded {len(data_docs)} ticket log entries.")
print(f" Loaded {len(guide_docs)} guide document.")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

print("Indexing documents...")
vector_store.add_documents(documents=all_splits)

# --- Tool Definition ---
@tool
def retrieve_context(query: str):
    """
    Retrieve ticket logs and API guide context to help answer a query.
    Useful for finding chronologies, submitters, and resolutions.
    """
    retrieved_docs = vector_store.similarity_search(query, k=10)
    
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized

tools = [retrieve_context]

# --- Agent Construction (LangGraph) ---

# The system prompt is passed as a state modifier
system_prompt = (
    "You are an IT Support Assistant. "
    "Use the 'retrieve_context' tool to find work logs for specific tickets. "
    "Always check the 'Source' metadata to distinguish between the 'Guide' and actual 'Ticket Logs'. "
    "When summarizing, use the Guide's logic to distinguish between System Events and Human Actions."
)
agent = create_agent(model, tools, system_prompt=system_prompt)

def main():
    print("Running Agent query...\n")
    
    # query = "Provide a chronological summary of what happened with ticket INC000000987654. Distinguish between automated system checks and human actions."
    query = "Conduct a root cause analysis on the ticket INC000000987654. Distinguish between automated system checks and human actions."
    
    # Run the agent using .stream()
    # stream_mode="values" returns the full list of messages at each step
    for event in agent.stream({"messages": [("user", query)]}, stream_mode="values"):
        # Print the last message received
        event["messages"][-1].pretty_print()

if __name__ == "__main__":
    main()