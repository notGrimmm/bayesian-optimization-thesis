from email import parser
from platform import processor
from peft import PeftModel
from transformers import pipeline
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoModelForCausalLM, AutoModel, AutoTokenizer
from datasets import load_dataset
import optuna
import argparse
import torch
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from langchain_community.embeddings.ollama import OllamaEmbeddings
from get_embedding_function import get_embedding_function

print(f"PyTorch path: {torch.__path__}")
print(f"PyTorch version: {torch.__version__}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
devNumber = torch.cuda.current_device() if torch.cuda.is_available() else "CPU"
print(f"Device name: {torch.cuda.get_device_name(devNumber) if torch.cuda.is_available() else 'CPU'}")
torch.cuda.empty_cache()
print(torch.cuda.memory_summary(device=None, abbreviated=False))

adapter_path = "./qwen-onboarding-final-v2"
model_name = "Qwen/Qwen2.5-1.5B"

class EmbeddingWrapper:
    def __init__(self, embedding_function):  # Add 'embedding_function' here
        self.embedding_function = embedding_function

    def embed_query(self, input):  # Ensure this argument is named 'input'
        return self.embedding_function.embed_query(input)
    
    def __call__(self, input):
        # ChromaDB internally calls the object itself with the 'input' argument
        # We route this to embed_documents because Chroma passes a list of texts
        return self.embedding_function.embed_query(input)

#ds = load_dataset("HuggingFaceFW/fineweb-edu", "default")
DefaultEmbeddingWrapped = EmbeddingWrapper(DefaultEmbeddingFunction)

pipe = pipeline("image-text-to-text", model="Qwen/Qwen3.5-4B")


#####################################################################################################################
CHROMA_PATH = "./db/chroma"

# prompt_template = """
# Answer the question based only on the following context:

# {context}

# ---

# Answer the question based on the above context: {question}
# """

prompt_template = """
You are a chatbot from an AI-orchestrated training module. The user wants to have a conversation with you. They are going through the training module. The context provided below is from the training module:

{context}

---

Respond to the user's query based on the conversation history between you and the user and above context. 
If the context does not contain relevant information to answer the user's query, respond with "Sorry, I don't have enough information to answer that question.". Do not attempt to answer the question if the context does not contain relevant information.
Response to ONLY the final prompt as an assistant based on the context and conversation history. 

Conversation history: 
{question}
"""


def main():
    # Create CLI.
    #parser = argparse.ArgumentParser()
    #parser.add_argument("query_text", type=str, help="The query text.")
    #args = parser.parse_args()
    query_text = """User: How many trees are there on this planet?
Assistant: Sorry, I don't have enough information to answer that question.
User: Why not?"""
    #args.query_text

    # Prepare the DB.
    db = Chroma(collection_name="courses", persist_directory=CHROMA_PATH, embedding_function=get_embedding_function)
    embedded_query = get_embedding_function().embed_query(query_text)

    # Search the DB.
    #results = db.similarity_search_with_relevance_scores(query_text, k=3)
    results = db.similarity_search_by_vector_with_relevance_scores(embedding=embedded_query, k=3)
    if len(results) == 0 or results[0][1] < 0.3:
        print(f"Unable to find matching results.")
        return

    #Get context from results, query text from API, then get response from model.
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt = prompt_template.format(context=context_text, question=query_text)

    print("Loading processor...")
    tokenizer = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    print("Loading model...")
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    print("Model loaded!")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    messages = [
    {
        "role": "user",
        "content":  [{"type": "text", "text": prompt}]
    }
    ]
    print("message format: \n", messages)
    pipe(messages)
    # inputs = tokenizer.apply_chat_template(
    #     messages,
    #     add_generation_prompt=True,
    #     tokenize=True,
    #     return_dict=True,
    #     return_tensors="pt",
    #     enable_thinking=False
    # ).to(model.device)

    inputs = tokenizer(prompt, enable_thinking=False, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    print(outputs.shape)
    print(inputs["input_ids"].shape)

    response_text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    )

    print(response_text)

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)


if __name__ == "__main__":
    main()