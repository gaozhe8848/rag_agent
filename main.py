import time
import config  # Import the configuration file
from datetime import datetime

from langchain.agents import create_agent
from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.tools import tool

# --- 1. Setup Models ---
embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL)
model = ChatOpenAI(model=config.LLM_MODEL, temperature=config.TEMPERATURE)

vector_store = InMemoryVectorStore(embeddings)

# --- 2. Loaders ---


def metadata_func_dataset(record: dict, metadata: dict) -> dict:
    metadata["incident_number"] = record.get("Incident Number")
    metadata["submitter"] = record.get("Submitter")
    metadata["log_type"] = record.get("Work Log Type")
    metadata["date"] = record.get("Submit Date")
    metadata["document_type"] = "ticket_log"
    return metadata


def metadata_func_guide(record: dict, metadata: dict) -> dict:
    metadata["document_type"] = "guide"
    return metadata


dataset_loader = JSONLoader(
    file_path=config.DATA_FILE_PATH,
    jq_schema='.entries[].values',
    text_content=False,
    metadata_func=metadata_func_dataset
)

guide_loader = JSONLoader(
    file_path=config.GUIDE_FILE_PATH,
    jq_schema='.',
    text_content=False,
    metadata_func=metadata_func_guide
)

# --- 3. Tool Definition ---


@tool(response_format="content_and_artifact")
def retrieve_ticket_history(incident_number: str):
    """
    Retrieves the full work log history for a specific Incident Number.
    Also retrieves the 'Analysis Guide' which explains how to interpret the logs.
    """
    # Use config.RETRIEVER_K to control depth
    guide_results = vector_store.similarity_search(
        "guide analysis rules",
        k=1,
        filter=lambda doc: doc.metadata.get("document_type") == "guide"
    )

    ticket_results = vector_store.similarity_search(
        f"Incident Number {incident_number}",
        k=config.RETRIEVER_K,
        filter=lambda doc: doc.metadata.get(
            "incident_number") == incident_number
    )

    combined_results = guide_results + ticket_results
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in combined_results
    )
    return serialized, combined_results


tools = [retrieve_ticket_history]

# --- 4. Agent Construction ---
agent = create_agent(model, tools, system_prompt=config.SYSTEM_PROMPT)

# --- 5. Main Execution ---


def main():
    global_start = time.time()

    # Generate timestamped filename using config prefix
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{config.LOG_FILE_PREFIX}_{timestamp_str}.log"

    print(f"--- Starting Process [{timestamp_str}] ---")

    # --- MEASURE LOADING & INDEXING ---
    t0 = time.time()
    print("Loading documents...")
    data_docs = dataset_loader.load()
    guide_docs = guide_loader.load()
    docs = guide_docs + data_docs

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    all_splits = text_splitter.split_documents(docs)

    vector_store.add_documents(documents=all_splits)
    indexing_time = time.time() - t0
    print(f"✅ Indexing complete in {indexing_time:.2f} seconds.")

    # Open the log file
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"=== AGENT WORKFLOW TEST RESULTS ===\n")
        f.write(f"Run Timestamp: {timestamp_str}\n")
        f.write(f"Model: {config.LLM_MODEL}\n")
        f.write(f"Indexing Time: {indexing_time:.4f}s\n\n")

        # Iterate through test cases from config
        for case in config.TEST_CASES:
            print(f"Processing: {case['workflow']}...", end=" ", flush=True)

            f.write(f"🔹 TESTING WORKFLOW: {case['workflow']}\n")
            f.write(f"❓ User Query: {case['query']}\n")
            f.write("-" * 40 + "\n")

            # --- MEASURE QUERY TIME ---
            query_start = time.time()

            inputs = {"messages": [("user", case['query'])]}
            final_response = None

            # Run Agent
            for event in agent.stream(inputs, stream_mode="values"):
                message = event["messages"][-1]
                if message.type == "ai" and not message.tool_calls:
                    final_response = message.content

            query_duration = time.time() - query_start
            print(f"Done ({query_duration:.2f}s)")

            f.write(f"💡 Agent Response:\n{final_response}\n")
            f.write(f"\n⏱️ Query Duration: {query_duration:.4f}s\n")
            f.write("=" * 60 + "\n\n")

        total_duration = time.time() - global_start
        f.write(f"🏁 Total Program Runtime: {total_duration:.4f}s\n")

    print(f"\n✅ All tasks finished in {total_duration:.2f}s.")
    print(f"📄 Results written to '{filename}'")


if __name__ == "__main__":
    main()
