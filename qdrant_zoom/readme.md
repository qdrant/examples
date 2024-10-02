# Qdrant Zoom RAG Application
You can view the deployed application here: https://zoom-qdrant.vercel.app/

This repository demonstrates how to build a Retrieval-Augmented Generation (RAG) application using Qdrant, focusing on Zoom meeting data. The project is divided into three main parts:

1. Data Retrieval, Storage and Vector Search with Qdrant and Zoom
2. Building and Evaluating RAG application Retrieval Methods with Qdrant and Relari
3. Deploying the Application to the internet with Qdrant and Nextjs

## Part 1: Data Retrieval and Storage

In this section, we use a Node.js application to:
- Fetch data from the Zoom API
- Process and store the data in a Qdrant collection
- Implement basic query functionality

Key technologies:
- Qdrant for vector storage and search
- Zoom API
- Node.js

## Part 2: Building and Evaluating Retrieval Methods

This part uses a Jupyter notebook to:
- Implement and compare different retrieval methods (dense, sparse, and hybrid)
- Evaluate the performance of each method
- Fine-tune prompts for optimal results

Key technologies:
- Qdrant
- Relari
- Langchain
- Instructor

## Part 3: Deploying the Search Application

The final part involves building and deploying a web application to search through Zoom meeting data:
- Develop a user friendlyinterface using Next.js
- Integrate the Qdrant backend for search functionality
- Deploy the application online

Key technologies:
- Qdrant
- TypeScript
- Next.js
- Vercel for deployment

## Getting Started

1. Clone the repository
2. Navigate to each part's directory and follow the specific README instructions

## Final Application

You can view the deployed application here: https://zoom-qdrant.vercel.app/

To use your own Zoom data, update the values in the `.env.local` file.

## Project Structure

- `qdrant_zoom/get_and_store_data/`: Node.js app for data retrieval and storage
- `qdrant_zoom/building_and_evaluating/`: Jupyter notebook for method evaluation
- `qdrant_zoom/deploying/`: Next.js app for the search interface

Each directory contains its own README with specific instructions and details.


This project showcases the entire process of building a RAG application, from data acquisition to deployment, using Qdrant as the vector database.