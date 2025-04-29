## 🧠 Qdrant-Powered AI Video Idea Generation

<div style="background: linear-gradient(90deg, #ff7e5f, #feb47b); padding: 20px; border-radius: 10px; color: white; text-align: center; font-size: 1.5rem; font-weight: bold;"> Qdrant + OpenAI: Making Your YouTube Channel Smarter </div>

## ✨ How Qdrant is Used
This application intelligently generates video ideas and finds related content by combining:

- Qdrant for vector search (semantic memory of past transcripts)

- OpenAI for embedding generation and creative idea generation

## 🔎 Step-by-Step Process
### 1. Create Embeddings
When a user uploads or processes a video transcript, we:

- Use OpenAI’s `text-embedding-ada-002 model` to convert the text into a vector.

- Store that vector + metadata (like `user_id`, `video_id`, etc.) inside Qdrant under the collection `video_transcripts`.

```
🚀 Function: embed_and_store(user, text, metadata)
```
### 2. Semantic Search
When a user searches for a new video idea:

- We embed the search query using OpenAI again.

- Then search Qdrant to find the most similar past transcripts, filtering only for the specific user's own content.
```
🔎 Function: search_similar_transcripts(query_text, user, top_k=10)

```
Flow:
```
[Transcript Text] 
    ⬇ (embedding)
[Vector Embedding + Payload (including transcript)] 
    ➡ (store in Qdrant)

Later...

[Query Text]
    ⬇ (embedding)
[Query Embedding]
    ➡ (search Qdrant for similar vectors)
    ➡ (retrieve payloads including transcripts)
    ➡ (use transcripts + query to ask OpenAI to propose a new idea)

```
## 🗂 Project Structure

```markdwown 
root/
├── backend/    # Django REST Framework (DRF) project
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── (other backend code)
│
├── frontend/   # Vue.js frontend project
│   └── (Vue.js code)
└── README.md   # (this file)

```
### 3. Creative Generation
With the top similar transcripts found:

- We prompt OpenAI to brainstorm a fresh video idea, blending the user’s past content with the current query trend.

- OpenAI returns a ready-to-use idea + title.
```
✍️ Function: generate_video_idea(user, query_text, similar_payloads)
```
## 🗄️ Qdrant Collection
We ensure the collection exists before use:
```
Collection name: video_transcripts
Vector size: 1536
Distance metric: COSINE
```


```
🛠️ Function: ensure_qdrant_collection()

```
<div style="background-color: #e0f7fa; border: 1px solid #00acc1; border-radius: 10px; padding: 20px; font-size: 1.2rem;"> 🎯 Thanks to Qdrant, users don't just get random content ideas. They get highly relevant, fresh, and personalized ideas based on their real content history — boosting channel growth smartly! </div>

## ⚙️ Prerequisites
Before starting, ensure you have installed:

- Docker + Docker Compose

- Node.js 

- npm 

## 🛠️ Backend Setup (`backend/`)
- Step 1: Environment Variables
- Create a `.env` file inside the `backend/` folder with your environment variables.

- Example `.env`:
```
DEBUG=True
SECRET_KEY=your_secret_key_here
ALLOWED_HOSTS=*
DATABASE_URL=postgres://postgres:postgres@db:5432/yourdb
```
## Step 2: Run Backend Services
From inside the backend/ folder:
```bash
docker-compose up --build

```
This will:

- Build the Django app and Postgres database

- Start the backend server with Gunicorn at http://localhost:8000

- Automatically create a volume for persistent Postgres data

## Step 3: Collect Static Files (if needed)
If you need static files immediately:
```bash
docker-compose exec web python manage.py collectstatic --noinput

```
## 🎨 Frontend Setup (`frontend/`)
Step 1: Install Dependencies

```bash 
npm install
```

## Step 2: Run the Frontend
```bash 
npm run dev

```
This will start the Vue.js frontend, typically at http://localhost:5173/ (Vite default).

## 📡 Connect Frontend to Backend
Update the API base URL in your Vue.js app to point to the backend:

For example in your `frontend/src/config.js` (or wherever you configure API):
```js
export const API_BASE_URL = "http://localhost:8000";
```
Or use `.env` variables if your Vue project is set up for them.

## 🌟 Production Notes
- In production, you should properly set up `CORS`, `HTTPS`, secure Gunicorn settings, and serve the frontend with Nginx or another web server.

- Backend static files should be collected into the /staticfiles volume and served separately.

- Consider setting `DEBUG=False` and securely managing your `.env` variables.
## 🛟 Useful Commands
| Action | Command |
|:------|:---------|
| Run Backend | `docker-compose up --build` |
| Stop Backend | `docker-compose down` |
| Enter Backend Container | `docker-compose exec web bash` |
| Run Migrations | `docker-compose exec web python manage.py migrate` |
| Create Superuser | `docker-compose exec web python manage.py createsuperuser` |

## 🎯 Quick URLs
- Backend API: http://localhost:8000

- Frontend App: http://localhost:5173

## 📺 YouTube API Integration

<div style="background: linear-gradient(90deg, #ff6a00, #ee0979); padding: 20px; border-radius: 10px; color: white; text-align: center; font-size: 1.5rem; font-weight: bold;"> Upload Videos Directly to YouTube — Without Leaving the App! </div>

## ✨ How YouTube Upload Works
This application uses the YouTube Data API v3 to:

- Authenticate users securely via OAuth2

- Upload generated videos directly to their YouTube channels

- Set titles, descriptions, tags, privacy settings, and more

## 🔑 Authentication Flow
When a user connects their YouTube account:

- We save their OAuth2 tokens: access token, refresh token, client ID, client secret, and scopes.

- We use these credentials to build a YouTube client dynamically whenever uploads are needed.

🔐 Code Example: Building authenticated YouTube client
```python
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

```
## ⬆️ Uploading a Video
Once authenticated:
- We prepare a metadata body (title, description, tags, etc.)

- We upload the video file using YouTube's videos.insert endpoint.

- After successful upload, the video URL is returned, and sent to the user via email.

🚀 Code Example: Uploading to YouTube
```python
body = {
    "snippet": {
        "title": "Your Video Title",
        "description": "Video description here",
        "tags": ["tag1", "tag2"],
        "categoryId": "28"  # e.g., Science & Technology
    },
    "status": {
        "privacyStatus": "public",
        "madeForKids": False
    }
}
media = MediaFileUpload(video_path, mimetype="video/mp4")
request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
response = request.execute()
youtube_url = f"https://youtube.com/watch?v={response['id']}"

```
## ⚙️ Setting up YouTube API Credentials
Before users can connect their YouTube accounts, you must create your own YouTube OAuth2 credentials.

📋 Steps to create credentials:
1. Go to the Google Cloud Console.

2. Create a new project or select an existing one.

3. Enable the YouTube Data API v3 for your project.

4. Create OAuth2 Client ID credentials:

- Application type: Web Application

- Add your app's authorized redirect URIs.

5. Save the Client ID and Client Secret into your backend.
## 🔥 Required OAuth2 Scopes:
```
https://www.googleapis.com/auth/youtube.upload
https://www.googleapis.com/auth/youtube.readonly

```
<div style="background-color: #fbe9e7; border: 1px solid #ff7043; border-radius: 10px; padding: 20px; font-size: 1.2rem;"> ⚡ <b> Important:</b> You must set up OAuth consent screen, branding, and request necessary scopes for production approval if you plan to allow external users (outside your Google Cloud organization). </div>

## 🎯 Summary
- Users authenticate once.

- Uploads happen securely in the background.

- Uploaded videos are public, tagged, and ready to grow your channel!
## 🔥 Flow Overview
<div style="background: #ffffff; border: 2px dashed #90caf9; border-radius: 10px; padding: 15px; font-size: 1.1rem;"> User logs in → Grants YouTube access → App uploads videos to their YouTube channel → App returns published video URL. </div>
<br>
<div style="background-color: #f0f9ff; border: 1px solid #bae6fd; border-radius: 10px; padding: 20px; font-size: 1.2rem;"> ✨ Thank you for using this application! 
Built with Qdrant. ✨ </div>