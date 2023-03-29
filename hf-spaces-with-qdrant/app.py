import gradio as gr
from qdrant_client import QdrantClient

from config import article, collection_name, description, examples, qdrant_api_key, qdrant_url
from encoder import encode_text
from utils import get_images

client = QdrantClient(url=qdrant_url,
                      port=443,
                      api_key=qdrant_api_key, prefer_grpc=False)


def search_images(query, modality):
    query_vector = encode_text(query)
    vector_name = "image" if modality == "images" else "text"
    results = client.search(
        collection_name, (vector_name, query_vector), limit=20, with_payload=True)

    images = get_images(results)

    return images


text = gr.Textbox(
    label="Enter your query",
    show_label=False,
    max_lines=1,
    placeholder="Enter your query",
).style(
    container=False,
)

modality = gr.Radio(["images", "captions"], label="Search against",
                    info="Search against image or text embeddings", value="images")

gallery = gr.Gallery(
    label="Semantically similar images", show_label=False, elem_id="gallery"
).style(grid=[2], height="auto")

demo = gr.Interface(fn=search_images, inputs=[text, modality], outputs=gallery,
                    title="Semantic image search with Qdrant Cloud", description=description, article=article, allow_flagging="never", examples=examples, cache_examples=False)

if __name__ == "__main__":
    demo.launch()
