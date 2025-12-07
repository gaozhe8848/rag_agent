# config.py

# --- Model Settings ---
LLM_MODEL = "gpt-5-mini"
EMBEDDING_MODEL = "text-embedding-3-large"
TEMPERATURE = 0

# --- File Paths ---
# Ensure the 'data' folder exists in the same directory as this script
DATA_FILE_PATH = "./data/remedy_data.json"
GUIDE_FILE_PATH = "./data/remedy_guide.json"
LOG_FILE_PREFIX = "./runtime/results/agent_results"

# --- Text Splitting & Indexing ---
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200

# --- Retrieval Settings ---
RETRIEVER_K = 15  # High K ensures we catch all logs for a ticket

# --- Agent System Prompt ---
SYSTEM_PROMPT = (
    "You are an IT Service Management Expert Agent. "
    "You use the 'retrieve_ticket_history' tool to access raw data. "
    "You must adapt your answer based on the user's specific workflow request:\n"
    "1. IF SUMMARY: Tell the chronological story.\n"
    "2. IF ROOT CAUSE: Filter strictly for 'Resolution' or 'Vendor' logs.\n"
    "3. IF STATUS/CONTACTS: List unique 'Submitters' and check the last log timestamp.\n\n"
    "Always rely on the 'remedy_guide.json' rules found in the context."
)

# --- Test Workflows ---
TEST_CASES = [
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
