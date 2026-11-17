from celery import shared_task
from api.youtube_utils import get_top_video_ids, download_audio
from api.youtube_utils import embed_and_store, ensure_qdrant_collection
from api.transcription import transcribe_audio  
import logging
from api.qdrant_utils import qdrant_client
from users.models import User
from users.models import Video
from qdrant_client.http import models as rest
import os, subprocess
import openai
from django.core.mail import EmailMultiAlternatives
from googleapiclient.http import MediaFileUpload
from googleapiclient.discovery import build
from .qdrant_utils import search_similar_transcripts, generate_video_idea
from api.redis_client import r
from elevenlabs import ElevenLabs
from replicate import Client as ReplicateClient
from google.oauth2.credentials import Credentials
import json

logger = logging.getLogger(__name__)

def send_video_upload_email(user_email, youtube_url):
    subject = "🎬 New Video Uploaded to Your Channel!"
    from_email = "Taledy Team <hello@taledy.com>"
    to = [user_email]

    text_content = f"New video uploaded: {youtube_url}"

    html_content = f"""
    <html>
      <body style="font-family: Arial, sans-serif; background-color: #f4f4f7; padding: 20px;">
        <table width="100%" cellpadding="0" cellspacing="0" style="max-width: 600px; margin: auto; background: #ffffff; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
          <tr>
            <td style="padding: 30px 40px;">
              <h2 style="color: #333333; margin-top: 0;">🚀 Your New Video is Live!</h2>
              <p style="font-size: 16px; color: #555555;">
                Hey there, your new video has just been uploaded and is ready for viewers.
              </p>
              <p style="text-align: center; margin: 30px 0;">
                <a href="{youtube_url}" style="background-color: #FF0000; color: white; padding: 14px 24px; text-decoration: none; border-radius: 6px; font-weight: bold;">
                  🎥 Watch on YouTube
                </a>
              </p>
              <p style="font-size: 14px; color: #999999;">
                If you didn't expect this email, you can safely ignore it.
              </p>
            </td>
          </tr>
        </table>
      </body>
    </html>
    """

    msg = EmailMultiAlternatives(subject, text_content, from_email, to)
    msg.attach_alternative(html_content, "text/html")
    msg.send()

@shared_task
def process_youtube_channel(channel_id, user_id):
    user = User.objects.get(id=user_id)
    ensure_qdrant_collection()
    video_ids = get_top_video_ids(channel_id)

    for vid in video_ids:
        try:
            path = download_audio(vid)
            text = transcribe_audio(user, path)
            embed_and_store(user, text, {
                "video_id": vid,
                "user_id": user_id,
                "channel_id": channel_id,
                "transcript": text,
            })
            logger.info(f"Processed and stored video {vid}")
        except Exception as e:
            logger.error(f"Failed to process {vid}: {e}")
        logger.info(f"Processed and stored all videos for user {user_id}")


@shared_task
def generate_and_upload_youtube_short_task(user_id,task_id):
    log_path = os.path.join("shorts", f"make_short_{task_id}.log")
    logger = logging.getLogger(f"make_short_{task_id}")
    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    try:
        logger.info("Received user request for short video generation")
        logger.info(f"Task ID: {task_id}")
        user = User.objects.get(id=user_id)
        print(f"[👤] User: {user.id}, [🆔] Task ID: {task_id} landed in celery")
        os.makedirs("shorts", exist_ok=True)
        r.hset(f"task:{task_id}", mapping={
                "status": "starting",
                "type": "transcription"
            })

        openai.api_key = user.openai_api_key_decrypted
        replicate_key = user.replicate_api_key_decrypted
        elevenlabs_key = user.elevenlabs_api_key_decrypted
        voice_id = user.elevenlabs_voice_id
        model = getattr(user, 'openai_model', None) or "gpt-4o"

        seed_prompt = f"new YouTube video idea for {user.audience}"
        similar_transcripts = search_similar_transcripts(seed_prompt, user=user)
        logger.info("Obtained similar transcripts")

        # 🧠 Generate unique idea based on what's already covered
        idea = generate_video_idea(user,seed_prompt, similar_transcripts)
        logger.info(f"Generated video idea: {idea}")

        # 1️⃣ Research with OpenAI
        research_prompt = f"""
        Research the following topic and provide a summary of key points, insights, and examples: 
        {idea}
        Summarize the key points and insights in a concise format.
        """
        research_response = openai.chat.completions.create(
            model="gpt-4o-search-preview",
            web_search_options={
                "search_context_size": "medium",
                "user_location": {
                    "type": "approximate",
                    "approximate": {"country": "US"}
                }
            },
            messages=[{"role": "user", "content": research_prompt}]
        )
        research_output = research_response.choices[0].message.content.strip()
        logger.info("Obtained research output")

        # 2️⃣ Script Generation
        script_prompt = f"""
        You are a viral short-form content writer for TikTok and YouTube Shorts.

        Your job is to create a fast-paced, high-retention, 1-minute video script about the following topic:
        {idea}

        Based on the research below:
        {research_output}

        Target audience: {user.audience}
        Tone: energetic, concise, and curiosity-driven.  
        Format: Use simple language, analogies if needed, and build toward an "aha!" moment.
        Make sure the script includes:
        1. A strong hook in the first 3 seconds
        2. Fast transitions between key points (no fluff)
        3. A payoff (surprising insight, use case, or why it matters)
        4. A call to action (e.g., Subscribe for more)
        Keep it under 300 words.
        Output the script only, no scene directions, no emojis or headings, just the script. 
        """
        script_response = openai.chat.completions.create(
            model=model,
            max_tokens=600,
            messages=[{"role": "user", "content": script_prompt}]
        )
        logger.info("Obtained script")
        transcript = script_response.choices[0].message.content.strip()

        # 3️⃣ Generate Voiceover
        audio_path = os.path.abspath(f"shorts/{task_id}_voiceover.mp3")
        logger.info("Generating voiceover")
        client = ElevenLabs(api_key=elevenlabs_key, timeout=1000)
        temp_audio = client.text_to_speech.convert(
            text=transcript,
            voice_settings={"speed": 1.2},
            voice_id=voice_id,
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
        )
        logger.info("Obtained voiceover")
        with open(audio_path, "wb") as f:
            for chunk_data in temp_audio:
                f.write(chunk_data)

        # 4️⃣ Generate Images with Replicate (ASYNC)
        import asyncio, aiohttp
        import math
        # Duration
        logger.info("Obtaining audio duration")
        result = subprocess.run([
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration", "-of",
            "default=noprint_wrappers=1:nokey=1", audio_path
        ], capture_output=True, text=True)
        try:
            duration = float(result.stdout.strip())
        except:
            duration = 30.0
        if duration < 5:
            duration = 30
        logger.info(f"Audio duration: {duration:.2f}s")

        num_images = int(duration // 5) + 1
        logger.info(f"Using {num_images} image(s)")
        words = transcript.split()
        words_per_chunk = math.ceil(len(words) / num_images)
        logger.info(f"Approx {words_per_chunk} words per chunk.")
        transcript_chunks = [
            ' '.join(words[i:i + words_per_chunk])
            for i in range(0, len(words), words_per_chunk)
        ]
        prompts = [f"Generate a descriptive image prompt for this transcript chunk:\n\n{chunk}" for chunk in transcript_chunks]
        logger.info(f"Starting image generation for {len(prompts)} prompts...")
        async def generate_images(prompts, replicate_key, task_id):
            replicate_api = ReplicateClient(api_token=replicate_key)
            image_paths = []
            for idx, image_prompt in enumerate(prompts):
                logger.info(f"Processing image {idx + 1}/{len(prompts)}")
                logger.info(f"Prompt: {image_prompt}")
                image_path = os.path.abspath(f"shorts/{task_id}_{idx}.jpg")
                for attempt in range(3):
                    try:
                        if not os.path.exists(image_path):
                            logger.info(f"Attempt {attempt+1}: Generating image {idx + 1} using FLUX")
                            prediction = replicate_api.predictions.create(
                                model=user.flux_model,
                                input={
                                    "prompt": image_prompt,
                                    "prompt_upsampling": True,
                                    "aspect_ratio": "9:16",
                                    "width": 1440,
                                    "height": 1440,
                                    "output_format": "jpg"
                                }
                            )
                            logger.info(f"Waiting for prediction {prediction.id}...")
                            for attempt in range(600):
                                prediction = replicate_api.predictions.get(prediction.id)
                                if prediction.status == "succeeded" and prediction.output:
                                    image_url = prediction.output[0] if isinstance(prediction.output, list) else prediction.output
                                    async with aiohttp.ClientSession() as session:
                                        async with session.get(image_url) as resp:
                                            if resp.status == 200:
                                                with open(image_path, "wb") as f:
                                                    f.write(await resp.read())
                                            else:
                                                raise Exception(f"Failed to download image: {resp.status}")
                                    break
                                elif prediction.status in ["failed", "canceled"]:
                                    logger.error(f"Prediction failed: {prediction}")
                                    raise RuntimeError(f"Prediction failed: {prediction}")
                                await asyncio.sleep(1)
                        else:
                            logger.info(f"Using cached image for chunk {idx + 1}")
                        break
                    except Exception as e:
                        logger.error(f"Image generation failed at chunk {idx + 1}, attempt {attempt+1}: {e}")
                        await asyncio.sleep(2)
                image_paths.append(image_path)
            return image_paths

        image_paths = asyncio.run(generate_images(prompts, replicate_key, task_id))
        logger.info(f"Generated {len(image_paths)} images")

        import re
        # --- SRT and chunking helpers ---
        def srt_time_to_seconds(t):
            h, m, s_ms = t.split(':')
            s, ms = s_ms.split(',')
            return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0
        def seconds_to_srt_time(sec):
            hrs = int(sec // 3600)
            sec -= hrs * 3600
            mins = int(sec // 60)
            sec -= mins * 60
            secs = int(sec)
            ms = int(round((sec - secs) * 1000))
            return f"{hrs:02}:{mins:02}:{secs:02},{ms:03d}"
        def parse_srt_blocks(raw_srt):
            blocks = []
            raw_blocks = re.split(r'\n\s*\n', raw_srt.strip(), flags=re.MULTILINE)
            index_pattern = re.compile(r'^(\d+)$')
            time_pattern = re.compile(r'^(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})$')
            for rb in raw_blocks:
                lines = rb.strip().split('\n')
                if len(lines) < 2:
                    continue
                idx_match = index_pattern.match(lines[0].strip())
                time_match = time_pattern.match(lines[1].strip())
                if not (idx_match and time_match):
                    continue
                block_index = int(idx_match.group(1))
                start_sec = srt_time_to_seconds(time_match.group(1))
                end_sec = srt_time_to_seconds(time_match.group(2))
                text_lines = lines[2:]
                blocks.append({
                    'index': block_index,
                    'start': start_sec,
                    'end': end_sec,
                    'lines': text_lines
                })
            return blocks
        def build_srt_block_str(block_index, start_sec, end_sec, text_lines):
            start_str = seconds_to_srt_time(start_sec)
            end_str = seconds_to_srt_time(end_sec)
            text_part = "\n".join(text_lines)
            return f"{block_index}\n{start_str} --> {end_str}\n{text_part}\n"
        def chunk_into_few_words(line, chunk_size=2):
            words = line.split()
            lines = []
            for i in range(0, len(words), chunk_size):
                lines.append(" ".join(words[i:i + chunk_size]))
            return lines
        def transform_srt_for_few_words_timed(raw_srt, chunk_size=2):
            blocks = parse_srt_blocks(raw_srt)
            new_blocks = []
            new_index = 1
            for b in blocks:
                start = b['start']
                end = b['end']
                duration = end - start if end > start else 1
                original_text = " ".join(b['lines'])
                chunked = chunk_into_few_words(original_text, chunk_size)
                if len(chunked) <= 1:
                    new_blocks.append({
                        'index': new_index,
                        'start': start,
                        'end': end,
                        'lines': [original_text],
                    })
                    new_index += 1
                else:
                    block_duration = duration / len(chunked)
                    cur_start = start
                    for chunk_text in chunked:
                        cur_end = cur_start + block_duration
                        new_blocks.append({
                            'index': new_index,
                            'start': cur_start,
                            'end': cur_end,
                            'lines': [chunk_text],
                        })
                        new_index += 1
                        cur_start = cur_end
            new_srt = []
            for nb in new_blocks:
                new_srt.append(build_srt_block_str(nb['index'], nb['start'], nb['end'], nb['lines']))
            return "\n".join(new_srt).strip() + "\n"

        # --- Animate each image with zoompan, then concatenate ---
        logger.info("\nStarting video clip generation...")
        video_clips = []
        for i, path in enumerate(image_paths):
            out_path =  os.path.abspath(f"shorts/{task_id}_clip_{i}.mp4")
            logger.info(f"Processing clip {i+1}/{len(image_paths)}")
            if not os.path.exists(path):
                print(f"Image file does not exist: {path}")
                continue
            cmd = [
                "ffmpeg", "-y", "-loop", "1", "-i", path,
                "-vf", "zoompan=z='min(zoom+0.0015,1.15)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=125:s=576x1024,format=yuv420p",
                "-frames:v", "125", "-r", "25",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                out_path
            ]
            logger.info(f"Running ffmpeg command with zoompan effect...")
            subprocess.run(cmd, check=True)
            logger.info(f"Generated clip {i+1}")
            video_clips.append(out_path)
        logger.info(f"\nCreated {len(video_clips)} video clips")

        # Concatenate all video clips
        concat_list = os.path.abspath(f"shorts/{task_id}_concat.txt")
        with open(concat_list, "w") as f:
            for clip in video_clips:
                f.write(f"file '{os.path.abspath(clip)}'\n")
        concat_path =  os.path.abspath(f"shorts/{task_id}_final_video.mp4")
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", concat_list,
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            concat_path
        ], check=True)
        if not os.path.exists(concat_path):
            raise Exception("Concatenated video not created")

        # Merge audio with concatenated video
        logger.info("\nMerging audio with video...")
        final_audio_path =  os.path.abspath(f"shorts/{task_id}_with_audio.mp4")
        subprocess.run([
            "ffmpeg", "-y",
            "-i", concat_path, "-i", audio_path,
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            final_audio_path
        ], check=True)
        if not os.path.exists(final_audio_path):
            logger.error("Audio merge failed")
            raise Exception("Audio merge failed")

        # Generate SRT subtitles with OpenAI Whisper
        logger.info("\nGenerating SRT subtitles...")
        srt_path = f"shorts/{task_id}.srt"
        with open(audio_path, "rb") as audio_file:
            transcript_res = openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="srt"
            )
        few_words_srt = transform_srt_for_few_words_timed(transcript_res, chunk_size=2)
        logger.info("Generated SRT subtitles")
        with open(srt_path, "w", encoding="utf-8") as srt_out:
            srt_out.write(few_words_srt)

        # Burn captions onto video
        logger.info("\nBurning captions onto video...")
        final_captioned_path =  os.path.abspath(f"shorts/{task_id}_animated_video.mp4")
        srt_abs = os.path.abspath(srt_path)
        style_str = (
            "Fontname=Arial,"
            "Bold=1,"
            "Fontsize=15,"
            "BorderStyle=1,"
            "Outline=2,"
            "Shadow=0,"
            "PrimaryColour=&H00FFFFFF,"
            "OutlineColour=&H00000000,"
            "Alignment=6,"
            "MarginV=120,"
            "MarginL=60,"
            "MarginR=60,"
            "WrapStyle=2"
        )
        logger.info("Burning captions onto video...")
        subtitles_filter = f"subtitles={srt_abs}:force_style='{style_str}'"
        full_filter = f"{subtitles_filter},format=yuv420p"
        subprocess.run([
            "ffmpeg", "-y",
            "-i", final_audio_path,
            "-vf", full_filter,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-c:a", "copy",
            final_captioned_path
        ], check=True)
        logger.info("Burned captions onto video")
        video_path = final_captioned_path

        # 6️⃣ Generate YouTube Metadata
        logger.info("\nGenerating YouTube metadata...")
        title_prompt = f"Generate a viral YouTube Shorts title based on this script:\n\nScript:\n{transcript}\nRespond only with the title text."
        title_response = openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": title_prompt}]
        ).choices[0].message.content.strip()
        title = title_response if title_response else "AI Video"
        logger.info(f"Generated title: {title}")

        description_prompt = f"Generate a compelling YouTube Shorts description for this script:\n\nScript:\n{transcript}\nKeep it concise and engaging. Respond only with the description."
        description_response = openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": description_prompt}]
        ).choices[0].message.content.strip()
        description = description_response if description_response else ""
        description = description_response if description_response else ""

        tags_prompt = f"Suggest 10 trending and relevant tags (comma-separated) for a YouTube Shorts video based on this script:\n\nScript:\n{transcript}\nRespond as: tag1, tag2, tag3, ..., tag10. Respond with the tags only"
        tags_response = openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": tags_prompt}]
        ).choices[0].message.content.strip()
        tags = [tag.strip() for tag in tags_response.split(",") if tag.strip()]
        logger.info(f"Generated tags: {tags}")

        # 7️⃣ Upload to YouTube
        logger.info("\nUploading to YouTube...")
        creds = Credentials(
            token=user.youtube_token.access_token,
            refresh_token=user.youtube_token.refresh_token,
            token_uri=user.youtube_token.token_uri,
            client_id=user.youtube_token.client_id,
            client_secret=user.youtube_token.client_secret,
            scopes=user.youtube_token.scopes.split(",")
        )
        logger.info("YouTube credentials loaded")
        youtube = build("youtube", "v3", credentials=creds)
        body = {
            "snippet": {
                "title": title.replace('"', ''),
                "description": description,
                "tags": tags,
                "categoryId": "28"
            },
            "status": {
                "privacyStatus": "public",
                "madeForKids": False
            }
        }
        logger.info("YouTube body created")
        media = MediaFileUpload(video_path, mimetype="video/mp4")
        logger.info("YouTube media created")
        request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
        logger.info("YouTube upload request created")
        response = request.execute()
        youtube_url = f"https://youtube.com/watch?v={response['id']}"
        logger.info(f"Uploaded to YouTube: {youtube_url}")
        # save to db
        logger.info("Saving to database...")
        Video.objects.create(
            user=user,
            video_url=youtube_url,
            title=title,
            description=description
        )
        # send email 
        logger.info("Sending email...")
        send_video_upload_email(user.email, youtube_url)
        logger.info("Email sent")
        result_json = {
            "task_id": task_id,
            "youtube_url": youtube_url,
            "title": title,
            "description": description,
            "tags": tags,
            "video_path": video_path,
            "audio_path": audio_path,
            "image_paths": image_paths,
            "transcript": transcript,
            "research_output": research_output,
        }
        logger.info("Task completed, saving to Redis")
        r.hset(f"task:{task_id}", mapping={
                    "status": "completed",
                    "result": json.dumps(result_json)
                })
        logger.info(f"Task {task_id} completed")
        logger.info("Removing logger handlers")
        logger.removeHandler(handler)
        handler.close()
        return result_json

    except Exception as e:
        logger.error(f"Failed to process: {e}")
        r.hset(f"task:{task_id}", mapping={
            "status": "failed",
            "type": "generate_and_upload_youtube_short",
            "error": str(e)
        })
        return {"error": str(e)}

@shared_task
def update_vectordb_from_youtube(user_id):
    user = User.objects.get(id=user_id)

    ensure_qdrant_collection()

    channel_id = user.youtube_token.channel_id  # Make sure you're storing this!
    top_video_ids = get_top_video_ids(channel_id)

    # Fetch existing video IDs from Qdrant
    scroll_result = qdrant_client.scroll(
        collection_name="video_transcripts",
        scroll_filter=rest.Filter(
            must=[
                rest.FieldCondition(
                    key="user_id",
                    match=rest.MatchValue(value=str(user.id))
                )
            ]
        ),
        with_payload=True,
        limit=1000
    )

    stored_video_ids = set(point.payload.get("video_id") for point in scroll_result[0])
    new_video_ids = [vid for vid in top_video_ids if vid not in stored_video_ids]

    logger.info(f"Found {len(new_video_ids)} new videos for user {user.email}")

    for vid in new_video_ids:
        try:
            path = download_audio(vid)
            text = transcribe_audio(user, path)
            embed_and_store(user, text, {
                "user_id": str(user.id),
                "video_id": vid,
                "channel_id": channel_id,
                "transcript": text,
            })
            logger.info(f"✅ Processed and stored video {vid} for user {user.email}")
        except Exception as e:
            logger.error(f"❌ Failed to process video {vid} for user {user.email}: {e}")
