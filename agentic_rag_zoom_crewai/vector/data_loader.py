import json
import os
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import uuid
import base64
from openai import OpenAI

# Load environment variables
env_path = Path(__file__).parent.parent / '.env.local'
load_dotenv(env_path)

# Set OpenAI key explicitly in environment with correct name
os.environ['OPENAI_API_KEY'] = os.getenv('openai_api_key')

class MeetingData:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._initialize()
            self._initialized = True

    def _initialize(self):
        """Initialize the instance only once"""
        self.data_dir = Path(__file__).parent.parent / 'data'
        self.meetings = self._load_meetings()
        
        # Initialize clients
        self.qdrant_client = QdrantClient(
            url=os.getenv('qdrantUrl'),
            api_key=os.getenv('qdrantApiKey')
        )
        self.openai_client = OpenAI()
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Ensure collection exists and is populated
        self._ensure_collection_exists()
        self._populate_collection()
        
        # Check Qdrant status after loading
        self._check_qdrant_status()
        print(f"LOG: Initialized MeetingData with directory: {self.data_dir}")

    def _base64_to_uuid(self, base64_string: str) -> str:
        """Convert base64 string to UUID."""
        try:
            base64_string = base64_string.rstrip('=')
            byte_string = base64.urlsafe_b64decode(base64_string + '=='*(-len(base64_string) % 4))
            return str(uuid.UUID(bytes=byte_string[:16]))
        except:
            return str(uuid.uuid4())

    def _ensure_collection_exists(self):
        """Create the Qdrant collection if it doesn't exist."""
        try:
            self.qdrant_client.get_collection('zoom_recordings')
            print("LOG: Collection 'zoom_recordings' already exists")
        except Exception:
            print("LOG: Creating collection 'zoom_recordings'...")
            self.qdrant_client.recreate_collection(
                collection_name='zoom_recordings',
                vectors_config=models.VectorParams(
                    size=384,  # SentenceTransformer dimension
                    distance=models.Distance.COSINE
                )
            )
            print("LOG: Collection created successfully")

    def _populate_collection(self):
        """Populate the Qdrant collection with meeting data."""
        print("LOG: Starting collection population...")
        
        try:
            # Get existing points count
            collection_info = self.qdrant_client.get_collection('zoom_recordings')
            if collection_info.points_count >= len(self.meetings):
                print("LOG: Collection already populated")
                return
        except Exception as e:
            print(f"LOG: Error checking collection: {e}")
        
        # Prepare points for insertion
        points = []
        for i, meeting in enumerate(self.meetings):
            try:
                # Create text for embedding
                text_to_embed = f"""
                Topic: {meeting.get('topic', '')}
                Content: {meeting.get('vtt_content', '')}
                Summary: {json.dumps(meeting.get('summary', {}))}
                """
                
                # Get embedding from SentenceTransformer instead of OpenAI
                vector = self.embedding_model.encode(text_to_embed).tolist()
                
                # Create point ID from meeting UUID if available
                point_id = self._base64_to_uuid(meeting.get('uuid', str(uuid.uuid4())))
                
                # Create point
                points.append(models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        'topic': meeting.get('topic'),
                        'start_time': meeting.get('start_time'),
                        'duration': meeting.get('duration'),
                        'summary': meeting.get('summary'),
                        'vtt_content': meeting.get('vtt_content'),
                        'user': meeting.get('user')
                    }
                ))
                
                if (i + 1) % 100 == 0:
                    print(f"LOG: Processed {i + 1} meetings...")
                    
            except Exception as e:
                print(f"LOG: Error processing meeting {i}: {e}")
        
        # Insert points in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            try:
                self.qdrant_client.upsert(
                    collection_name='zoom_recordings',
                    points=batch
                )
                print(f"LOG: Inserted batch {i//batch_size + 1} of {len(points)//batch_size + 1}")
            except Exception as e:
                print(f"LOG: Error inserting batch: {e}")
        
        print("LOG: Collection population complete")

    def _load_meetings(self) -> List[Dict[str, Any]]:
        """Load all meeting data from JSON files in the data directory."""
        all_meetings = []
        
        # Walk through all files in the data directory
        for file_path in self.data_dir.glob('*.txt'):
            try:
                print(f"LOG: Loading data from {file_path}")
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    # Extract meetings from the recordings array
                    if 'recordings' in data:
                        for recording in data['recordings']:
                            # Add user info to each recording
                            recording['user'] = {
                                'firstname': data.get('firstname'),
                                'lastname': data.get('lastname'),
                                'email': data.get('email')
                            }
                            all_meetings.append(recording)
            except Exception as e:
                print(f"LOG: Error loading file {file_path}: {e}")
        
        print(f"LOG: Loaded {len(all_meetings)} meetings total")
        return all_meetings

    def _check_qdrant_status(self):
        """Check if meetings are properly indexed in Qdrant."""
        try:
            # Get collection info
            print(f"LOG: Connecting to Qdrant at: {os.getenv('qdrantUrl')}")
            collection_info = self.qdrant_client.get_collection('zoom_recordings')
            points_count = collection_info.points_count
            
            print(f"LOG: Qdrant collection status:")
            print(f"LOG: - Total points in collection: {points_count}")
            print(f"LOG: - Total meetings loaded: {len(self.meetings)}")
            
            if points_count < len(self.meetings):
                print("LOG: WARNING - Some meetings may not be indexed in Qdrant!")
                print("LOG: Run the indexing script to ensure all meetings are searchable.")
            elif points_count > len(self.meetings):
                print("LOG: WARNING - More points in Qdrant than loaded meetings!")
                print("LOG: Collection may contain outdated or duplicate entries.")
            else:
                print("LOG: ✓ Qdrant collection is in sync with loaded meetings")
                
        except Exception as e:
            print(f"LOG: Error checking Qdrant status: {e}")
            print("LOG: WARNING - Qdrant collection may not be properly configured!")

    def search_meetings(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search through meetings using vector search"""
        print(f"LOG: Searching meetings with query: {query}")
        
        # For statistical queries, return all meetings
        if any(word in query.lower() for word in ['average', 'mean', 'total', 'count', 'statistics']):
            print("LOG: Statistical query detected - returning all meetings")
            return self.meetings

        try:
            # Get embedding from OpenAI
            print("LOG: Getting OpenAI embedding for query")
            response = self.openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=query
            )
            query_vector = response.data[0].embedding
            
            # Search Qdrant with limit of 10
            print("LOG: Searching Qdrant")
            vector_results = self.qdrant_client.search(
                collection_name='zoom_recordings',
                query_vector=query_vector,
                limit=10,  # Changed from default to 10
                score_threshold=0.7  # Only return good matches
            )
            
            if vector_results:
                print(f"LOG: Found {len(vector_results)} matches in Qdrant")
                return [
                    {
                        'score': hit.score,
                        'topic': hit.payload.get('topic', 'N/A'),
                        'start_time': hit.payload.get('start_time', 'N/A'),
                        'duration': hit.payload.get('duration', 'N/A'),
                        'summary': hit.payload.get('summary', {}),
                        'user': hit.payload.get('user', {}),
                        'content': hit.payload.get('vtt_content', '')
                    }
                    for hit in vector_results
                ]
            else:
                print("LOG: No vector matches found, falling back to content matching")
                
        except Exception as e:
            print(f"LOG: Vector search failed: {e}")
            print("LOG: Falling back to content matching")

        # Fallback to content matching
        matches = []
        for meeting in self.meetings:
            score = 0
            if query.lower() in meeting['topic'].lower():
                score += 0.5
            if 'vtt_content' in meeting and query.lower() in meeting['vtt_content'].lower():
                score += 0.3
            if 'summary' in meeting and query.lower() in str(meeting['summary']).lower():
                score += 0.2
            
            if score > 0:
                matches.append({
                    'score': score,
                    'topic': meeting['topic'],
                    'start_time': meeting['start_time'],
                    'duration': meeting['duration'],
                    'summary': meeting.get('summary', {}),
                    'user': meeting.get('user', {}),
                    'content': meeting.get('vtt_content', '')
                })
        
        matches.sort(key=lambda x: x['score'], reverse=True)
        print(f"LOG: Found {len(matches)} matches using content matching")
        return matches[:limit]

    def get_average_duration(self) -> float:
        """Calculate and return the average meeting duration."""
        if not self.meetings:
            return 0
        total_duration = sum(meeting.get('duration', 0) for meeting in self.meetings)
        avg_duration = total_duration / len(self.meetings)
        print(f"LOG: Average meeting duration across {len(self.meetings)} meetings: {avg_duration:.2f} minutes")
        return avg_duration

if __name__ == "__main__":
    print("LOG: Running MeetingData loader test...")
    
    # Initialize MeetingData
    meeting_data = MeetingData()
    
    # Print some basic stats
    print("\nLOG: Basic meeting stats:")
    print(f"LOG: - Number of meetings loaded: {len(meeting_data.meetings)}")
    
    # Test search functionality
    test_query = "marketing strategy"
    print(f"\nLOG: Testing search with query: '{test_query}'")
    results = meeting_data.search_meetings(test_query)
    
    print("\nLOG: Search results:")
    for i, result in enumerate(results, 1):
        print(f"\nLOG: Result {i}:")
        print(f"LOG: - Topic: {result['topic']}")
        print(f"LOG: - Score: {result['score']}")
        print(f"LOG: - User: {result['user'].get('firstname', 'N/A')} {result['user'].get('lastname', 'N/A')}")
        print(f"LOG: - Duration: {result['duration']} minutes")
