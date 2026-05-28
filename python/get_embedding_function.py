#import boto3
#from langchain_community.embeddings.ollama import OllamaEmbeddings
#from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings


def get_embedding_function():
    # embeddings = BedrockEmbeddings(
    #     model_id="amazon.titan-embed-text-v1", region_name="us-east-1"
    # )
    #embeddings = OllamaEmbeddings(model="nomic-embed-text")

    embeddings = HuggingFaceEmbeddings()
    return embeddings