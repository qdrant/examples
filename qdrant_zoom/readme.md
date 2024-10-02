
This repo is a simple example of how to use Qdrant to build a RAG app, from getting your data from Zoom to building a search app and deploying it on the internet.

In this repo you can find code for getting your from Zoom and storing it in a Qdrant collection.

In this repo we will build a website so that search thought your zoom meetings.

You'll learn how to get data from an API, how to store it in a vector database and how to search through it using Qdrant.

Here is the final site if you want to check it out: https://zoom-qdrant.vercel.app/

If you want to build the site with your own Zoom data, change the values in the `.env.local` file.

In the first repo you'll be working in a full stack project, but mosty with python.

In the seond repo you'll be working in a jupyter notebook on Collab or locall. 

In the third repo you'll be working with typescript and Next js to deploy your application.

## Setting up the environment

1. Clone the repo
2. Run `npm install`
3. Run `npm run dev`

It uses the following libraries:
- `qdrant-client` for interacting with Qdrant
- `python-zoomus` for interacting with Zoom's API

First we get our zoom data

then we store it in a Qdrant collection.

then we build a simple rag app with dense, sparse and hybrid retrieval

then we evaluate the performance of the different retrieval methods

Then we tune our prompt.

Then we build a site and deploy with Next js.