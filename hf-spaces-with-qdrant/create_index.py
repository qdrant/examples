"""Clip index is a tool to index clip embeddings using Qdrant"""
import json
import fire
import glob
import os
import time
import logging

import numpy as np
import pandas as pd
import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, Record, VectorParams, OptimizersConfigDiff, Payload


from config import collection_name, qdrant_api_key, qdrant_url

LOGGER = logging.getLogger(__name__)



def get_vector_size_and_number(img_emb_files):
    embeddings = np.load(img_emb_files[0])
    number_of_vectors, vector_size = embeddings.shape
    total_number_of_vectors = int(number_of_vectors * len(img_emb_files))
    vector_size = int(vector_size)

    LOGGER.info(f"Vector size = {vector_size}")
    LOGGER.info(f"Estimated number of vectors = {total_number_of_vectors}")

    return total_number_of_vectors, vector_size


def get_embeddings_and_records(img_emb_files, txt_emb_files, metadata_files):
    for i, (img_file, txt_file, metadata_file) in enumerate(zip(img_emb_files, txt_emb_files, metadata_files)):
        payload_data = pd.read_parquet(metadata_file)
        payload_data.drop(columns=["image_path", "hash", "key", "status",
                                   "error_message", "width", "height", "exif", "sha256", "original_width", "original_height"], errors="ignore", inplace=True)
        payload_data = payload_data.to_dict(orient='records')
        
        img_embeddings = np.load(img_file)
        txt_embeddings = np.load(txt_file)

        records = (Record(id=(i+1)*j, vector={"image": img_embeddings[j].tolist(
        ), "text": txt_embeddings[j].tolist()}, payload=payload_data[j] or {}) for j in range(img_embeddings.shape[0]))

        yield records


def clip_index(
    embeddings_folder,
    batch_size=64,
    parallel=2,
    max_retries=5,
    image_subfolder="img_emb",
    text_subfolder="text_emb",
):
    """indexes clip embeddings using Qdrant"""
    client = QdrantClient(
        url=qdrant_url, api_key=qdrant_api_key, prefer_grpc=True)

    image_folder = f"{embeddings_folder}/{image_subfolder}"
    text_folder = f"{embeddings_folder}/{text_subfolder}"
    metadata_folder = f"{embeddings_folder}/metadata"
    img_emb_files = glob.glob(f"{image_folder}/*.npy")
    txt_emb_files = glob.glob(f"{text_folder}/*.npy")
    metadata_files = glob.glob(f"{embeddings_folder}/metadata/*.parquet")
    assert len(img_emb_files) == len(txt_emb_files) == len(
        metadata_files), "Image, text or metadata directories have a different number of files"

    # Fix for glob returning unnormalized paths on Windows
    img_emb_files = [os.path.normpath(fname) for fname in img_emb_files]
    txt_emb_files = [os.path.normpath(fname) for fname in txt_emb_files]
    metadata_files = [os.path.normpath(fname) for fname in metadata_files]

    number_of_vectors, vector_size = get_vector_size_and_number(img_emb_files)
    vectors_config = VectorParams(
        size=vector_size, distance=Distance.COSINE)

    client.recreate_collection(collection_name, vectors_config={
                               "image": vectors_config, "text": vectors_config}, on_disk_payload=True, optimizers_config=OptimizersConfigDiff(indexing_threshold=number_of_vectors, memmap_threshold=20000))

    records_gen = get_embeddings_and_records(
        img_emb_files, txt_emb_files, metadata_files)
    for records in tqdm.tqdm(records_gen, desc="Uploading", total=len(img_emb_files)):
        client.upload_records(collection_name, records, batch_size=batch_size,
                              parallel=parallel, max_retries=max_retries)

    LOGGER.warn("Upload finished, re-enabling indexing...")
    client.update_collection(collection_name, optimizer_config=OptimizersConfigDiff(
        indexing_threshold=10000, memmap_threshold=100000))

    time.sleep(2)

    collection = client.get_collection(collection_name)
    print(collection)


if __name__ == "__main__":
    fire.Fire(clip_index)
