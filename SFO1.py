def handle_failure(user_query, attempt):
    if attempt == 1:
        return "I apologize, I did not fully understand that query. Could you please rephrase it, or tell me if this relates to (A) Safety/Emergency, (B) Operations/Status, or (C) Facilities/Construction?" # Structured clarification
    elif attempt == 2 and 'safety' in user_query:
        return "SAFETY PRIORITY ALERT: For immediate safety issues, please hang up the bot chat and call [Local Emergency Number]. I am trained for information only." # Hard escalation
    else:
        return f"I apologize. I could not resolve this issue with my current data set. Please speak to a human supervisor at extension XXXX, or visit the Information Desk." # Graceful exit

# Key Concept: Attempt Counter & Forced Escalation

CONSTRUCTION_UPDATE = {
    "date": "2024-10-25",
    "zone_id": "T2_GATE_C34",
    "status": ["CLOSED", "WARNING"],
    "reason": "Maintenance scheduled.",
    "effective_time": "Until 2024-10-27 06:00 AM EST",
    "public_impact": "Use Gate C35. Shuttle buses will be rerouted."
}


class StatusDatabase:
    def __init__(self):
        self.records = [CONSTRUCTION_UPDATE]

    def query_status(self, zone, search_term):
        zone_upper = zone.upper()
        search = search_term.lower()
        return [
            record for record in self.records
            if zone_upper in record["zone_id"].upper()
            and search in " ".join(str(value) for value in record.values()).lower()
        ]


database = StatusDatabase()

def get_construction_update(zone, search_term):
    # 1. Check the live data feed (simulated API call)
    live_data = database.query_status(zone, search_term) 

    if not live_data:
        return "No current construction updates found for this zone."
    
    update = live_data[0] # Get the most recent status
    
    # 2. Build a precise, actionable summary
    response = (
        f"ATTENTION: {zone} is currently {update['status'][0]}. "
        f"Reason: {update['reason']} It will reopen/be accessible by {update['effective_time']}. "
        f"ACTION REQUIRED: Please follow the alternate route specified: {update['public_impact']}."
    )
    return response

# Key Concept: Always state the source and validity date.
# Assume you have loaded all SOPs into a directory named 'manuals/'

from pathlib import Path


def create_embeddings(chunks):
    """Create embeddings with the OpenAI Python SDK."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("Embeddings require the 'openai' package.") from exc

    response = OpenAI().embeddings.create(
        model="text-embedding-3-small",
        input=chunks,
    )
    return [item.embedding for item in response.data]


def load_text(file_path):
    """Load text from a plain-text, PDF, or DOCX file."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".txt":
        return path.read_text(encoding="utf-8")

    if suffix == ".pdf":
        try:
            from pypdf import PdfReader  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError("PDF support requires the 'pypdf' package. Install it with 'pip install pypdf'.") from exc

        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)

    if suffix == ".docx":
        try:
            from docx import Document  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError("DOCX support requires the 'python-docx' package.") from exc

        document = Document(path)
        return "\n".join(paragraph.text for paragraph in document.paragraphs)

    raise ValueError(f"Unsupported document format: {suffix}")

def ingest_documents(file_paths):
    return [load_text(file) for file in file_paths]

# CALL THIS FIRST: ingest_documents(["path/to/sop1.pdf", "path/to/sop2.docx"])


def greet():
    message = "Hello"
    print(message)  # OK

# This function MUST use standard Python code, NOT an LLM call, 
# because safety protocols are too important to let AI "interpret."

def run_safety_protocol(query):
    query = query.lower()
    if 'medical' in query:
        return "EMERGENCY PROTOCOL ACTIVE: For immediate medical emergencies, dial [LOCAL NUMBER] and follow the instructions of local uniformed staff immediately."
    elif 'fire' in query or 'smoke' in query:
        return "LIFE SAFETY ALERT: Evacuate IMMEDIATELY. Move away from the source, follow designated exit signs, and meet at Assembly Point Gamma."
    else:
        return "Safety Check: If you are unsure about a safety procedure, always consult local law enforcement or fire marshals on site."
def route_query(user_input: str):
    lower_query = user_input.lower()
    if any(k in lower_query for k in ["emergency", "safety", "medical", "fire", "smoke"]):
        return "SAFETY"
    if any(k in lower_query for k in ["construction", "gate", "zone"]):
        return "CONSTRUCTION"
    if any(k in lower_query for k in ["operations", "status"]):
        return "OPERATIONS"
    return "GENERAL"


def chat_with_aioc(user_query):
    intent = route_query(user_query)
    if intent == "SAFETY":
        return run_safety_protocol(user_query)
    if intent == "CONSTRUCTION":
        return get_construction_update("T2_GATE_C34", "gate")
    if intent == "OPERATIONS":
        return "Current operations information is available through the airport operations desk."
    return "I apologize, I am unsure how to assist with that topic. Please try rephrasing your question."


def main():
    while True:
        user_input = input("\nYOU: ").strip()
        if user_input.lower() in ["quit", "exit"]:
            break
        if user_input:
            print("BOT:", chat_with_aioc(user_input))


if __name__ == "__main__":
    main()

