# Agentic Rag with Qdrant and CrewAI

## Overview

This project demonstrates how to build a Vector Search powered Agentic workflow to extract insights from meeting recordings. By combining Qdrant's vector search capabilities with CrewAI agents, users can search through and analyze their own meeting content.

The application first converts the meeting transcript into vector embeddings and stores them in a Qdrant vector database. It then uses CrewAI agents to query the vector database and extract insights from the meeting content. Finally, it uses Anthropic Claude to generate natural language responses to user queries based on the extracted insights from the vector database.

The system is built on three main components:

- **Qdrant Vector Database**: Stores meeting transcripts and summaries as vector embeddings, enabling semantic search
- **CrewAI Framework**: Coordinates AI agents that handle different aspects of meeting analysis
- **Anthropic Claude**: Provides natural language understanding and response generation

The example data includes meeting transcripts with speaker identification, timestamps, and AI-generated summaries.

Data Structure:

- `userid` - unique identifier for the user
- `firstname` - user's first name
- `lastname` - user's last name
- `email` - user's email address
- `recordings` - array of meeting recordings, each containing:
  - `uuid` - unique identifier for the recording
  - `topic` - meeting topic/title
  - `start_time` - meeting start time in ISO format
  - `duration` - meeting duration in minutes
  - `vtt_content` - timestamped transcript of the meeting
  - `summary` - AI-generated meeting summary containing:
    - `summary_title` - brief title of the meeting
    - `summary_overview` - high-level overview of what was discussed
    - `summary_details` - array of key points discussed, each with a label and summary
    - `next_steps` - array of action items and next steps from the meeting

In this repository in the `data` folder, we've included a sample dataset of 60 meetings from 3 users so you can get started quickly.

## Setup

Before running any code, complete these prerequisite steps:

1. Get API credentials and cluster url for Qdrant:
   - Create an account at [Qdrant Cloud](https://cloud.qdrant.io/)
   - Create a new cluster
   - Get the cluster URL from the cluster details page
   - Go to "Data Access Control" tab and create an API key
   Get API credentials for Anthropic and OpenAI
   - Get an API key from [Anthropic](https://www.anthropic.com/)
   - Get an API key from [OpenAI](https://platform.openai.com/)

2. Clone the repository:

```bash
git clone https://github.com/qdrant/examples
```

3. Navigate to the project directory:

```bash
cd examples/agentic_rag_zoom_crewai/
```

4. Create and activate a Python virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. Configure environment variables in `.env.local`:

```text
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
QDRANT_URL=your_qdrant_url_here
QDRANT_API_KEY=your_qdrant_api_key_here
```

## Usage

When you've set up the environment, follow these steps to process your meeting data and launch the interface:

1. Process meeting data:

```bash
python vector/data_loader.py
```

This script will:

- Load meeting data from JSON files in the `data/` directory
- Process each meeting's topic, content, and summary
- Create vector embeddings using SentenceTransformer
- Create or verify the 'zoom_recordings' collection in Qdrant
- Upload the embeddings to Qdrant in batches of 100
- Verify that all meetings were properly indexed

You'll see logs in your console showing:

- Number of meetings loaded
- Collection creation/verification status
- Batch upload progress
- Final verification of indexed meetings

If you see warnings about missing or duplicate entries, you may need to run the script again.

2. Test the search functionality:

```bash
# The script will automatically run a test search for "marketing strategy"
# You should see results showing:
# - Meeting topics
# - Search relevance scores
# - User information
# - Meeting durations
```

3. Launch the Streamlit interface:

```bash
streamlit run vector/streamlit_app.py
```

When the interface loads:

- Enter natural language queries in the search box
- View matching meetings ranked by relevance
- See meeting summaries and key details
- Get AI-generated insights about the meetings

Example queries:

- "Find meetings about product launches"
- "Show me marketing discussions from last month"
- "What was discussed in the longest meeting?"
- "Find meetings where [person's name] presented"

The system uses both vector search and content matching as a fallback, so you'll get relevant results even if exact matches aren't found.

## Components

### Data Processing

The `data_loader.py` script handles:

- Loading meeting transcripts and summaries
- Converting text to vector embeddings using FastEmbed
- Storing data in Qdrant for search

### AI Agents

The `crew.py` module implements:

- CrewAI agents for meeting analysis
- Custom tools for Qdrant Vector search integration
- Custom tools for Anthropic Claude response generation

### Web Interface

The Streamlit app provides:

- Interactive query interface
- Structured response generation

## Project Structure

```
.
├── .env.local - Environment variables
├── requirements.txt - Python dependencies
├── vector/
│   ├── crew.py - AI agent logic with CrewAI
│   ├── data_loader.py - Data processing with Qdrant
│   └── streamlit_app.py - Web interface
└── data/ - Meeting data files
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

```
