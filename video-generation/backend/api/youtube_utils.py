from googleapiclient.discovery import build
import os
import yt_dlp
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
import os
import uuid
from openai import OpenAI
from google.oauth2.credentials import Credentials
import logging
from django.conf import settings

logger = logging.getLogger(__name__)
# Set up basic logging configuration
logging.basicConfig(
    level=logging.INFO,  # or logging.DEBUG for more verbose output
    format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
)

def get_authenticated_channel_id(token_obj):
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build

    creds = Credentials(
        token=token_obj.access_token,
        refresh_token=token_obj.refresh_token,
        token_uri=token_obj.token_uri,
        client_id=token_obj.client_id,
        client_secret=token_obj.client_secret,
        scopes=token_obj.scopes.split(","),
    )

    youtube = build("youtube", "v3", credentials=creds)

    response = youtube.channels().list(
        part="id",
        mine=True
    ).execute()

    return response["items"][0]["id"]

    
def get_top_video_ids(channel_id, max_results=50):
    youtube = build('youtube', 'v3', developerKey=os.getenv("YOUTUBE_API_KEY"))
    res = youtube.search().list(
        part="id", channelId=channel_id, order="viewCount", maxResults=max_results
    ).execute()

    return [item["id"]["videoId"] for item in res["items"] if item["id"]["kind"] == "youtube#video"]

import os
import yt_dlp

def download_audio(video_id):
    url = f"https://www.youtube.com/watch?v={video_id}"
    output_dir = os.path.abspath("shorts")
    os.makedirs(output_dir, exist_ok=True)

    # We set output as a template, yt-dlp will append correct extension
    output_template = os.path.join(output_dir, f"{video_id}.%(ext)s")
    expected_output = os.path.join(output_dir, f"{video_id}.mp3")

    print(f"[🎧] Downloading audio to: {expected_output}")

    cookiefile_path = os.path.join(settings.BASE_DIR, 'cookies.txt')

    ydl_opts = {
        'cookiefile': cookiefile_path,
        'format': 'bestaudio/best',
        'outtmpl': output_template,
        'quiet': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '0',  # Highest quality
        }],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    if not os.path.exists(expected_output):
        raise FileNotFoundError(f"Audio file not found after download: {expected_output}")

    print(f"[✅] Audio downloaded: {expected_output}")
    return expected_output



qdrant = QdrantClient(url=os.getenv("QDRANT_HOST"),prefer_grpc=False )

def ensure_qdrant_collection():
    if not qdrant.collection_exists("video_transcripts"):
        qdrant.create_collection(
            collection_name="video_transcripts",
            vectors_config=VectorParams(
                size=1536,
                distance=Distance.COSINE
            )
        )
        

def embed_and_store(user, text, metadata):
    logger.info(f"[🔑] Starting embed_and_store for user {user.id} with metadata: {metadata}")

    try:
        client = OpenAI(api_key=user.openai_api_key_decrypted)
        logger.info("[🧠] Initialized OpenAI client.")
    except Exception as e:
        logger.exception("[❌] Failed to initialize OpenAI client.")
        raise e

    try:
        response = client.embeddings.create(
            input=[text],
            model="text-embedding-ada-002"
        )
        embedding = response.data[0].embedding
        logger.info("[✅] Embedding successfully created.")
    except Exception as e:
        logger.exception("[❌] Failed to generate embedding.")
        raise e

    try:
        point_id = str(uuid.uuid4())
        logger.info(f"[🆔] Generated UUID: {point_id}")

        point = PointStruct(id=point_id, vector=embedding, payload=metadata)
        logger.info("[📦] PointStruct created.")

        qdrant.upsert("video_transcripts", [point])
        logger.info(f"[📤] Upserted into Qdrant with point ID {point_id}")

        return point_id

    except Exception as e:
        logger.exception("[❌] Failed to upsert into Qdrant.")
        raise e
