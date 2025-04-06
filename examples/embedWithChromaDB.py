# from https://ollama.com/blog/embedding-models, with fixes.
# (example in blogpost doesn't work)

import ollama
import chromadb

# Step 1: generate embeddings
# Generate embeddings and store the resulting vectors into the chromadb.
documents = [
  "Llamas are members of the camelid family meaning they're pretty closely related to vicuñas and camels",
  "Llamas were first domesticated and used as pack animals 4,000 to 5,000 years ago in the Peruvian highlands",
  "Llamas can grow as much as 6 feet tall though the average llama between 5 feet 6 inches and 5 feet 9 inches tall",
  "Llamas weigh between 280 and 450 pounds and can carry 25 to 30 percent of their body weight",
  "Llamas are vegetarians and have very efficient digestive systems",
  "Llamas live to be about 20 years old, though some only live for 15 years and others live to be 30 years old",
]

client = chromadb.Client()
collection = client.create_collection(name="docs")

# store each document in a vector embedding database
for i, d in enumerate(documents):
  response = ollama.embed(model="mxbai-embed-large", input=d)
  embeddings = response["embeddings"]
  collection.add(
    ids=[str(i)],
    embeddings=embeddings,
    documents=[d]
  )

# Step 2: Retrieve
# retrieve the most relevant document given an example inputPrompt
inputPrompt = "What animals are llamas related to?"

# generate an embedding for the inputPrompt
response = ollama.embed(
  model="mxbai-embed-large",
  input=inputPrompt
)

# retrieve the most relevant doc
results = collection.query(
  query_embeddings=response["embeddings"],
  n_results=1
)
data = results['documents'][0][0]

# Step 3: Generate
# Use the prompt and the document retrieved in step 2 to generate an answer!
output = ollama.generate(
  model="llama3.2",
  prompt=f"Using this data: {data}. Respond to this prompt: {inputPrompt}"
)

print(output['response'])
