def handle_failure(user_query, attempt):
    if attempt == 1:
        return "I apologize, I did not fully understand that query. Could you please rephrase it, or tell me if this relates to (A) Safety/Emergency, (B) Operations/Status, or (C) Facilities/Construction?" # Structured clarification
    elif attempt == 2 and 'safety' in user_query:
        return "🚨 SAFETY PRIORITY ALERT: For immediate safety issues, please hang up the bot chat and call [Local Emergency Number]. I am trained for information only." # Hard escalation
    else:
        return f"I apologize. I could not resolve this issue with my current data set. Please speak to a human supervisor at extension XXXX, or visit the Information Desk." # Graceful exit

# Key Concept: Attempt Counter & Forced Escalation

{
    "date": "2024-10-25",
    "zone_id": "T2_GATE_C34",
    "status": ["CLOSED", "WARNING"],
    "reason": "Maintenance scheduled.",
    "effective_time": "Until 2024-10-27 06:00 AM EST",
    "public_impact": "Use Gate C35. Shuttle buses will be rerouted."
}

def get_construction_update(zone, search_term):
    # 1. Check the live data feed (simulated API call)
    live_data = database.query_status(zone, search_term) 

    if not live_data:
        return "No current construction updates found for this zone."
    
    update = live_data[0] # Get the most recent status
    
    # 2. Build a precise, actionable summary
    response = (
        f"⚠️ ATTENTION: {zone} is currently {update['status'][0]}. "
        f"Reason: {update['reason']}. It will reopen/be accessible by {update['effective_time']}. "
        f"ACTION REQUIRED: Please follow the alternate route specified: {update['public_impact']}."
    )
    return response

# Key Concept: Always state the source and validity date.
username = "Alice"
print(username)  # OK

def greet():
    message = "Hello"
    print(message)  # OK
    username = "Alice"
print(username)  # OK
# Assume you have loaded all SOPs into a directory named 'manuals/'

from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb import Client
from openai import OpenAI # Used here to create embeddings

# 1. Initialize ChromaDB client (this is your vector store)
client = Client()
collection = client.get_or_create_collection("aioc_knowledge_base")

def ingest_documents(file_paths):
    print("--- Starting Document Indexing ---")
    all_texts = []
    for file in file_paths:
        # (Simplified function to read text from PDF/DOCX)
        text = load_text(file) 
        all_texts.append(text)

    # 2. Split the documents into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_texts(all_texts)

    # 3. Create Embeddings (The Magic Step: turning words to math)
    embeddings = openai.EmbeddingClient().create_embeddings(input=chunks) # Use OpenAI embeddings API

    # 4. Store in ChromaDB
    ids = [f"doc_{i}" for i in range(len(chunks))]
    collection.add(
        embeddings=embeddings,  # The math representation
        metadatas=[{"source": "SOP Manual"}], # Source context
        documents=chunks
    )
    print(f"✅ Successfully indexed {len(chunks)} chunks of data.")

# CALL THIS FIRST: ingest_documents(["path/to/sop1.pdf", "path/to/sop2.docx"])


def greet():
    message = "Hello"
    print(message)  # OK
