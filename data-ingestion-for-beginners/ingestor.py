from langchain_community.document_loaders import S3DirectoryLoader
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, models
from dotenv import load_dotenv
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import boto3
import io
import os


load_dotenv()

aws_access_key_id = os.getenv("ACCESS_KEY")
aws_secret_access_key = os.getenv("SECRET_ACCESS_KEY")
qdrant_key = os.getenv("QDRANT_KEY")

s3 = boto3.client('s3', aws_access_key_id = aws_access_key_id, aws_secret_access_key = aws_secret_access_key)
qdrant_client = QdrantClient(
    url = "<QDRANT URL>",
    api_key = qdrant_key,
)


# Initialize CLIP model and processor for images
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
#Intialize text embedding model
text_embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")


def embed_image_with_clip(image):
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)
    return image_features.cpu().numpy()

# Define document processing function
def process_document(doc):
    source = doc.metadata['source']
    
    # Check file type
    if source.endswith('.txt'):
        # Text file processing
        text = doc.page_content
        print(f"Processing .txt file: {source}")
        return text, text_embedding_model.embed_documents([text])
    
    elif source.endswith('.pdf'):
        # PDF file processing using PDFPlumberLoader
        print(f"Processing .pdf file: {source}")
        content = doc.page_content
        return content, text_embedding_model.embed_documents([content])
    
    elif source.endswith('.png'):
        # Image file processing using OCR
        print(f"Processing .png file: {source}")
        bucket_name, object_key = parse_s3_url(source)
        response = s3.get_object(Bucket=bucket_name, Key=object_key)
        img_bytes = response['Body'].read()

        img = Image.open(io.BytesIO(img_bytes))

        return  source, embed_image_with_clip(img)

# Helper function to parse S3 URL
def parse_s3_url(s3_url):
    parts = s3_url.replace("s3://", "").split("/", 1)
    bucket_name = parts[0]
    object_key = parts[1]
    return bucket_name, object_key


def create_collection(collection_name):
    qdrant_client.create_collection(
    collection_name,
    vectors_config={
        "text_embedding": models.VectorParams(
            size=1536,
            distance=models.Distance.COSINE,
        ),
        "image_embedding": models.VectorParams(
            size=512,
            distance=models.Distance.COSINE,
        ),
    },
)

def ingest_data(points):


    operation_info = qdrant_client.upsert(
        collection_name="products-data",  # Collection where data is being inserted
        points=points
    )

    return operation_info

    

if __name__ == "__main__":


    collection_name = "products-data"
    create_collection(collection_name)
    folders_count = 6
    points = []
    for i in range(folders_count):
        folder = f"p_{i}" # Name of the folder
        loader = S3DirectoryLoader(
            "product-dataset",
            folder,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )

        docs = loader.load()
    
        text_embedding, image_embedding, text_review, product_image = [], [], "", ""

        for idx, doc in enumerate(docs):
            source = doc.metadata['source']
            
            if source.endswith(".txt"):
                text_review, text_embedding = process_document(doc)

            elif source.endswith(".png"):
                product_image, image_embedding = process_document(doc)
        
        if text_review:
            point = PointStruct(
                id=idx,  # Unique identifier for each point
                vector={
                    "text_embedding": text_embedding[0],  # The text embedding vector
                    "image_embedding": image_embedding[0].tolist(),  # The image embedding vector
                },
                payload={
                    "review": text_review,  # Storing the review in the payload
                    "product_image": product_image  # Storing the product image reference (URL/path) in the payload
                }
            )
            points.append(point)

    operation_info = ingest_data(points)
    print(operation_info)