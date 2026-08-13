import gradio as gr
from huggingface_hub import InferenceClient
import glob
from sentence_transformers import SentenceTransformer
import torch

# This is the same pattern from the Generative AI lesson! It uses the
# Inference Provider API to send your messages to an AI model and get
# a response back. Swap out the model below for a different one if
# you want to experiment!
#
# Note: if this Space doesn't already have one, you'll need to add an
# HF_TOKEN secret in the Space's Settings tab for this to work
# (Settings -> Variables and secrets -> New secret).


#all_text = ""

#text_files = glob.glob("*.txt")

#for filename in text_files:
    #with open(filename, "r", encoding="utf-8") as file:
     #   file_text = file.read()
      #  all_text += "\n" + file_text

#print(all_text)


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

model = SentenceTransformer('all-MiniLM-L6-v2')

def create_embeddings(text_chunks):
  # Convert each text chunk into a vector embedding and store as a tensor
  chunk_embeddings = model.encode(text_chunks, convert_to_tensor=True) # Replace ... with the cleaned_chunks list

  # Print the chunk embeddings
  print(chunk_embeddings)

  # Print the shape of chunk_embeddings
  print(chunk_embeddings.shape)

  # Return the chunk_embeddings
  return chunk_embeddings

def get_top_chunks(query, chunk_embeddings, text_chunks):
  # Convert the query text into a vector embedding
  query_embedding = model.encode(query, convert_to_tensor=True) # Complete this line

  # Normalize the query embedding to unit length for accurate similarity comparison
  query_embedding_normalized = query_embedding / query_embedding.norm()

  # Normalize all chunk embeddings to unit length for consistent comparison
  chunk_embeddings_normalized = chunk_embeddings / chunk_embeddings.norm(dim=1, keepdim=True)

  # Calculate cosine similarity between all chunks and the query using matrix multiplication
  similarities = torch.matmul(chunk_embeddings_normalized, query_embedding_normalized) # Complete this line

  # Print the similarities
  print(similarities)

  # Find the indices of the 3 chunks with highest similarity scores
  top_indices = torch.topk(similarities, k=3).indices

  # Print the top indices
  print(top_indices)

  # Create an empty list to store the most relevant chunks
  top_chunks = []

  # Loop through the top indices and retrieve the corresponding text chunks
  # This is only one way scholars may write this, but there are other ways!
  for i in top_indices:
    chunk = text_chunks[i]
    top_chunks.append(chunk)

  # ===== SPICY CHALLENGE: LIST COMPREHENSION =====
  # top_chunks = [chunks[i] for i in top_indices]

  # Return the list of most relevant chunks
  return top_chunks


def respond(message, history):

    if user_location=="Bahamas":
        Bahamas_text = ""
        Bahamas_files = glob.glob("*Bahamas.txt")
        for filename in Bahamas_files:
            with open(filename, "r", encoding="utf-8") as file:
                file_text = file.read()
                Bahamas_text += "\n" + file_text

        
        Bahamas_cleaned_chunks = preprocess_text(Bahamas_text)
        Bahamas_chunk_embeddings = create_embeddings(Bahamas_cleaned_chunks)
        top_chunks=get_top_chunks(message,all_chunk_embeddings,all_cleaned_chunks)
        
    elif user_location=="Cuba":
        Cuba_text=""
        Cuba_files = glob.glob("*Cuba.txt")
        for filename in Cuba_files:
            with open(filename, "r", encoding="utf-8") as file:
                file_text = file.read()
                Cuba_text += "\n" + file_text

        
        Cuba_cleaned_chunks = preprocess_text(Cuba_text)
        Cuba_chunk_embeddings = create_embeddings(Cuba_cleaned_chunks)
        top_chunks=get_top_chunks(message,all_chunk_embeddings,all_cleaned_chunks)
    elif user_location=="Cuba":
        Jamaica_text=""
        Jamaica_files = glob.glob("*Jamaica.txt")
        for filename in Jamaica_files:
            with open(filename, "r", encoding="utf-8") as file:
                file_text = file.read()
                Jamaica_text += "\n" + file_text

        
        Jamaica_cleaned_chunks = preprocess_text(Jamaica_text)
        Jamaica_chunk_embeddings = create_embeddings(Jamaica_cleaned_chunks)
        top_chunks=get_top_chunks(message,all_chunk_embeddings,all_cleaned_chunks)
    elif user_location=="Cuba":
        PuertoRico_text=""
        PuertoRico_files = glob.glob("*PuertoRico.txt")
        for filename in PuertoRico_files:
            with open(filename, "r", encoding="utf-8") as file:
                file_text = file.read()
                PuertoRico_text += "\n" + file_text

        
        PuertoRico_cleaned_chunks = preprocess_text(PuertoRico_text)
        PuertoRico_chunk_embeddings = create_embeddings(PuertoRico_cleaned_chunks)
        top_chunks=get_top_chunks(message,all_chunk_embeddings,all_cleaned_chunks)
    elif user_location=="Trinidad and Tobago":
        TrinidadandTobago_text=""
        TrinidadandTobago_files = glob.glob("*TrinidadandTobago.txt")
        for filename in TrinidadandTobago_files:
            with open(filename, "r", encoding="utf-8") as file:
                file_text = file.read()
                TrinidadandTobago_text += "\n" + file_text

        
        TrinidadandTobago_cleaned_chunks = preprocess_text(TrinidadandTobago_text)
        TrinidadandTobago_chunk_embeddings = create_embeddings(TrinidadandTobago_cleaned_chunks)
        top_chunks=get_top_chunks(message,all_chunk_embeddings,all_cleaned_chunks)
    elif user_location=="TurksandCaicos":
        TurksandCaicos_text=""
        TurksandCaicos_files = glob.glob("*TurksandCaicos.txt")
        for filename in TurksandCaicos_files:
            with open(filename, "r", encoding="utf-8") as file:
                file_text = file.read()
                TurksandCaicos_text += "\n" + file_text

        
        TurksandCaicos_cleaned_chunks = preprocess_text(TurksandCaicos_text)
        TurksandCaicos_chunk_embeddings = create_embeddings(Turksand aicos_cleaned_chunks)
        top_chunks=get_top_chunks(message,all_chunk_embeddings,all_cleaned_chunks)

    
    # ===== APPLY THE COMPLETE WORKFLOW =====
    all_cleaned_chunks = preprocess_text(all_text)
    all_chunk_embeddings = create_embeddings(all_cleaned_chunks)
    top_chunks=get_top_chunks(message,all_chunk_embeddings,all_cleaned_chunks)
    
    messages = [{"role": "system", "content": "You are a budget-friendly travel agent chatbot who specializes in helping users create their dream vacation at Puerto Rico, Bahamas, Jamaica, Trinidad and Tobago, Turks and Caicos, Cuba."}]

    if history:
        messages.extend(history)

    messages.append({"role": "user", "content": message})

    response = client.chat_completion(
        messages,
        max_tokens=200, temperature = 0.5
    )

    return response.choices[0].message.content.strip()

custom_theme = gr.themes.Soft(
    primary_hue="pink",
    secondary_hue="fuchsia", 
    neutral_hue="purple",
    spacing_size="lg",
    radius_size="lg",
    text_size="lg",
    font=[gr.themes.GoogleFont("IBM Plex Sans"), "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "monospace"]
)
with gr.Blocks(theme=custom_theme) as demo:
    with gr.Column():
        with gr.Row():
            gr.Markdown("# Travel Tropical")
        with gr.Row():
            gr.ChatInterface(respond)

    with gr.Column():
      with gr.Row():
          name=gr.Textbox(placeholder="Type your name", label="Name") 
          user_location=gr.CheckboxGroup(["Puerto Rico", "Bahamas", "Jamaica", "Trinidad and Tobago", "Turks and Caicos", "Cuba"], label="Where do you want to go?")
     

demo.launch(ssr_mode=False)


# TODO: This is just a starting point! Customize the system prompt,
# the model, and the interface to make this project your own!