import sys
import os
from crewai import Agent, Task, Crew
from crewai.tools import BaseTool
from typing import Type, List, Dict
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import anthropic
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env.local
env_path = Path(__file__).parent.parent / '.env.local'
load_dotenv(env_path)

# Set API keys from environment
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

# Initialize clients
qdrant_client = QdrantClient(
    url=os.getenv('QDRANT_URL'),
    api_key=os.getenv('QDRANT_API_KEY')
)
openai_client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Define tool input schemas
class CalculatorInput(BaseModel):
    """Input schema for calculator tools."""
    a: int = Field(..., description="First number")
    b: int = Field(..., description="Second number")

class SearchInput(BaseModel):
    """Input schema for search tool."""
    query: str = Field(..., description="The search query")

class AnalysisInput(BaseModel):
    """Input schema for meeting analysis tool."""
    meeting_data: dict = Field(..., description="Meeting data to analyze")

# Define custom tools
class CalculatorTool(BaseTool):
    name: str = "calculator"
    description: str = "Perform basic mathematical calculations"
    args_schema: Type[BaseModel] = CalculatorInput

    def _run(self, a: int, b: int) -> dict:
        return {
            "addition": a + b,
            "multiplication": a * b
        }

class SearchMeetingsTool(BaseTool):
    name: str = "search_meetings"
    description: str = "Search through meeting recordings using vector similarity"
    args_schema: Type[BaseModel] = SearchInput

    def _run(self, query: str) -> List[Dict]:
        # Use OpenAI embeddings to match data_loader.py
        response = openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=query
        )
        query_vector = response.data[0].embedding
        
        search_results = qdrant_client.search(
            collection_name='zoom_recordings',
            query_vector=query_vector,
            limit=10,
            score_threshold=0.7
        )
        
        return [
            {
                "score": hit.score,
                "topic": hit.payload.get('topic', 'N/A'),
                "start_time": hit.payload.get('start_time', 'N/A'),
                "duration": hit.payload.get('duration', 'N/A'),
                "summary": hit.payload.get('summary', {}).get('summary_overview', 'N/A')
            }
            for hit in search_results
        ]

class MeetingAnalysisTool(BaseTool):
    name: str = "analyze_meeting"
    description: str = "Analyze meeting content using Claude"
    args_schema: Type[BaseModel] = AnalysisInput

    def _run(self, meeting_data: dict) -> Dict:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        
        # Check if we received a list of meetings in the meetings key
        meetings = meeting_data.get('meetings', [])
        if not isinstance(meetings, list):
            meetings = [meeting_data]  # Convert single meeting to list
            
        # Format all meetings for analysis
        meetings_text = "\n\n".join([
            f"""Meeting {i+1}:
            Topic: {m.get('topic')}
            Start Time: {m.get('start_time')}
            Duration: {m.get('duration')} minutes
            Summary: {m.get('summary')}"""
            for i, m in enumerate(meetings)
        ])
        
        prompt = f"""
        Please analyze these meetings:
        
        {meetings_text}
        
        Provide:
        1. Key discussion points across all meetings
        2. Main decisions or action items
        3. Overall patterns and insights
        4. Notable participants and their contributions
        5. Recommendations for follow-up
        """
        
        message = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return {
            "meetings_analyzed": len(meetings),
            "analysis": message.content,
            "timestamp": datetime.now().isoformat()
        }

def get_crew_response(query: str) -> str:
    # Create tool instances
    calculator = CalculatorTool()
    searcher = SearchMeetingsTool()
    analyzer = MeetingAnalysisTool()
    
    # Create agents
    researcher = Agent(
        role='Research Assistant',
        goal='Find and analyze relevant information',
        backstory="""You are an expert at finding and analyzing information.
                  You know when to use calculations, when to search meetings,
                  and when to perform detailed analysis.""",
        tools=[calculator, searcher, analyzer],
        verbose=True
    )
    
    synthesizer = Agent(
        role='Information Synthesizer',
        goal='Create comprehensive and clear responses',
        backstory="""You excel at taking raw information and analysis
                  and creating clear, actionable insights.""",
        verbose=True
    )
    
    # Create tasks with expected_output
    research_task = Task(
        description=f"""Process this query: '{query}'
                    1. If it involves calculations, use the calculator tool
                    2. If it needs meeting information, use the search tool
                    3. For detailed analysis, use both search and analysis tools
                    Explain your tool selection and process.""",
        expected_output="""A dictionary containing:
                       - The tools used
                       - The raw results from each tool
                       - Any calculations or analysis performed""",
        agent=researcher
    )
    
    synthesis_task = Task(
        description="""Take the research results and create a clear response.
                    Explain the process used and why it was appropriate.
                    Make sure the response directly addresses the original query.""",
        expected_output="""A clear, structured response that includes:
                       - Direct answer to the query
                       - Supporting evidence from the research
                       - Explanation of the process used""",
        agent=synthesizer
    )
    
    # Create and run crew
    crew = Crew(
        agents=[researcher, synthesizer],
        tasks=[research_task, synthesis_task],
        verbose=True
    )
    
    result = crew.kickoff()
    return result

if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = ' '.join(sys.argv[1:])
        try:
            result = get_crew_response(query)
            print(f"\nResult: {result}")
        except Exception as e:
            print(f"Error processing query: {str(e)}")
    else:
        print("Please provide a query as a command line argument.") 
