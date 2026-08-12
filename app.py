import gradio as gr
from huggingface_hub import InferenceClient
import glob

# This is the same pattern from the Generative AI lesson! It uses the
# Inference Provider API to send your messages to an AI model and get
# a response back. Swap out the model below for a different one if
# you want to experiment!
#
# Note: if this Space doesn't already have one, you'll need to add an
# HF_TOKEN secret in the Space's Settings tab for this to work
# (Settings -> Variables and secrets -> New secret).


all_text = ""

text_files = glob.glob("*.txt")

for filename in text_files:
    with open(filename, "r", encoding="utf-8") as file:
        file_text = file.read()
        all_text += "\n" + file_text

print(all_text)


client = InferenceClient("Qwen/Qwen2.5-7B-Instruct", bill_to="kode-with-klossy")
def preprocess_text(text):
    cleaned_text = text.strip()
    chunks = cleaned_text.split("\n")
    cleaned_chunks = []
    for chunk in chunks:
        stripped_chunk = chunk.strip()
        if len(stripped_chunk) > 0:
            cleaned_chunks.append(stripped_chunk)

    print(cleaned_chunks)
    print(len(cleaned_chunks))
    return cleaned_chunks

def create_embeddings(text_chunks):
  # Convert each text chunk into a vector embedding and store as a tensor
  chunk_embeddings = model.encode(text_chunks, convert_to_tensor=True) # Replace ... with the cleaned_chunks list

  # Print the chunk embeddings
  print(chunk_embeddings)

  # Print the shape of chunk_embeddings
  print(chunk_embeddings.shape)

  # Return the chunk_embeddings
  return chunk_embeddings


def respond(message, history):
    # ===== APPLY THE COMPLETE WORKFLOW =====
    all_cleaned_chunks = preprocess_text(all_text)
    all_chunk_embeddings = create_embeddings(all_cleaned_chunks)
    
    messages = [{"role": "system", "content": "You are a budget-friendly travel agent chatbot who specializes in helping users create their dream vacation at Puerto Rico, Bahamas, Jamaica, Trinidad and Tobago, Turks and Caicos, Cuba."}]

    if history:
        messages.extend(history)

    messages.append({"role": "user", "content": message})

    response = client.chat_completion(
        messages,
        max_tokens=200, temperature = 0.5
    )

    return response.choices[0].message.content.strip()

chatbot = gr.ChatInterface(respond)

chatbot.launch()


# TODO: This is just a starting point! Customize the system prompt,
# the model, and the interface to make this project your own!