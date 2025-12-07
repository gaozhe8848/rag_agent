from langchain.agents import create_agent
from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.tools import tool
import os
from datetime import datetime

# 1. Setup Models
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
model = ChatOpenAI(model="gpt-5-mini", temperature=0)

vector_store = InMemoryVectorStore(embeddings)

# --- Loaders ---


def metadata_func_dataset(record: dict, metadata: dict) -> dict:
    # We capture Date and Submitter specifically for the "Status & Contacts" workflow
    metadata["incident_number"] = record.get("Incident Number")
    metadata["submitter"] = record.get("Submitter")
    metadata["log_type"] = record.get("Work Log Type")
    metadata["date"] = record.get("Submit Date")
    metadata["document_type"] = "ticket_log"
    return metadata


def metadata_func_guide(record: dict, metadata: dict) -> dict:
    metadata["document_type"] = "guide"
    return metadata


# Loader 1: The Ticket Data
dataset_loader = JSONLoader(
    file_path='./data/remedy_data.json',
    jq_schema='.entries[].values',
    text_content=False,
    metadata_func=metadata_func_dataset
)

# Loader 2: The Context Guide
# (Ensure your remedy_guide.json has the "analysis_rules" section we added previously)
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

print(f"✅ Loaded {len(data_docs)} ticket log entries.")
print(f"✅ Loaded {len(guide_docs)} guide document.")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

vector_store.add_documents(documents=all_splits)

# --- Tool Definition ---


@tool(response_format="content_and_artifact")
def retrieve_ticket_history(incident_number: str):
    """
    Retrieves the full work log history for a specific Incident Number.
    Also retrieves the 'Analysis Guide' which explains how to interpret the logs.
    """
    # 1. Get the Guide (Always useful context)
    guide_results = vector_store.similarity_search(
        "guide analysis rules",
        k=1,
        filter=lambda doc: doc.metadata.get("document_type") == "guide"
    )

    # 2. Get the Ticket Logs
    # We use a filter (lambda) to ensure we ONLY get logs for the requested ticket.
    # This prevents hallucination from mixing up ticket INC...54 and INC...55
    ticket_results = vector_store.similarity_search(
        f"Incident Number {incident_number}",
        k=15,
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

# --- Agent Construction ---

system_prompt = (
    "You are an IT Service Management Expert Agent. "
    "You use the 'retrieve_ticket_history' tool to access raw data. "
    "You must adapt your answer based on the user's specific workflow request:\n"
    "1. IF SUMMARY: Tell the chronological story.\n"
    "2. IF ROOT CAUSE: Filter strictl for 'Resolution' or 'Vendor' logs to find the technical defect.\n"
    "3. IF STATUS/CONTACTS: List the unique 'Submitters' and check the timestamp of the LAST log entry.\n\n"
    "Always rely on the 'remedy_guide.json' rules found in the context."
)

agent = create_agent(model, tools, system_prompt=system_prompt)

# --- Simulation of User Workflows ---


def main():
    # We will simulate 3 different users asking 3 different types of questions
    # to prove the single agent handles them all.

    test_cases = [
        {
            "workflow": "ROOT CAUSE ANALYSIS",
            "query": "What was the technical root cause of incident INC000000987654? Don't give me a summary, just the cause."
        },
        {
            "workflow": "STATUS & CONTACTS",
            "query": "Who worked on ticket INC000000987654 and is it closed yet?"
        },
        {
            "workflow": "FULL SUMMARY",
            "query": "Give me a walkthrough of what happened in ticket INC000000987655."
        }
    ]

    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"./runtime/results/agent_results_{timestamp}.log"

    # Open the log file
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"=== AGENT WORKFLOW TEST RESULTS ===\n")
        f.write(f"Timestamp: {timestamp}\n\n")

        for case in test_cases:
            # Console feedback
            print(f"Processing: {case['workflow']}...")

            # Write Header to File
            f.write(f"🔹 TESTING WORKFLOW: {case['workflow']}\n")
            f.write(f"❓ User Query: {case['query']}\n")
            f.write("-" * 40 + "\n")

            inputs = {"messages": [("user", case['query'])]}

            # Execute Agent
            final_response = None
            for event in agent.stream(inputs, stream_mode="values"):
                message = event["messages"][-1]
                if message.type == "ai" and not message.tool_calls:
                    final_response = message.content

            # Write Response to File
            f.write(f"💡 Agent Response:\n{final_response}\n")
            f.write("=" * 60 + "\n\n")

    print(f"\n✅ Done! Results have been written to '{filename}'")


if __name__ == "__main__":
    main()
