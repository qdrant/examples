---
title: Qdrant Cloud Demo
emoji: 🦀
colorFrom: red
colorTo: blue
sdk: gradio
sdk_version: 3.17.0
app_file: app.py
pinned: true
license: apache-2.0
---

# hf-spaces-demo
A semantic image search demo on HuggingFace Spaces backed by Qdrant Cloud

This example demonstrates how to leverage Qdrant Cloud and Huggingface Spaces to present your achievement  in vector similarity publically with ease.
We will use Huggingface Spaces to host the application.
Huggingface Spaces is a free platform for hosting machine learning models and web applications.
We will encode the [MSCOCO dataset](https://cocodataset.org/)
to embeddings with [CLIP](https://github.com/openai/CLIP)
and store them in a free tier cluster at Qdrant Cloud.
In the HF Spaces app, we will accept a textual query and make a request to Qdrant Cloud for vector search against either image or text embeddings based on the user's choice.

## Let's get started
I provide the CLIP embeddings of the MSCOCO dataset as a downloadable archive,
and we will use it to index in Qdrant Cloud.
As a side note, I will release the Qdrant snapshots of larger datasets that are ready for importing to your Qdrant instance on the following days,
and I will demonstrate how those snapshots can be used
for solving varius problems in different use cases. Stay tuned for the upcoming posts and join [Discord](https://qdrant.to/discord)
if you haven't already.

In the remainder of this post, I will provide
step-by-step instructions on how to host a [Gradio](https://gradio.app/)
app on HF Spaces for semantic image search,
backed by Qdrant Cloud. If you would like to prefer
the source code directly instead, go to the [project repository](https://github.com/qdrant/hf-spaces-demo).

## Step 1: Setting up

Before starting, make sure that you signed up at Qdrant Cloud,
created a cluster and obtained the URL and API key.
We will use them for accessing our instance, so let's set them as environment variables:

```shell
export QDRANT_API_KEY=<YOUR_API_KEY>

export QDRANT_URL=<YOUR_CLUSTER_URL>
```


Clone the repository to your development machine and install the dependencies.
Please note that indexing embeddings and web app have different sets of dependencies,
so I suggest holding dependencies in two different requirements files.

```shell
git clone https://github.com/qdrant/examples.git

cd examples/hf-spaces-with-qdrant

pip install -r requirements-indexing.txt
```

Sign up at Huggingface and create a [new space](https://huggingface.co/new-space).
Take the URL of the space repository, and set it as a new remote:

```shell
git init

git remote add hf <YOUR_SPACE_URL>
```

## Step 2: Indexing embeddings

We are ready for indexing embeddings in our instance at Qdrant Cloud. It's a single command after downloading the embeddings file:

```shell
wget https://storage.googleapis.com/qdrant-datasets-index/mscoco_embeddings.zip

unzip mscoco_embeddings.zip

python create_index.py --embeddings_folder ./mscoco_embeddings
```

We are almost there! Let's create our HF Spaces app.

## Step 3: Creating app

One important point that you need to remember when using Qdrant Cloud on Huggingface Spaces
is that HF Spaces allows requests to external sources only on ports 80, 443 and 8080,
but Qdrant listens on ports 6333 and 6334 by default.
Therefore, you need to connect Qdrant Cloud on port 443,
which is configured as an additional port to the defaults in the cloud offering
to overcome such limitations of various services.
You can see the example in [`app.py`](https://github.com/qdrant/hf-spaces-demo/blob/master/app.py#L9).

Before pushing our code to Huggingface Spaces repository, we need to set credentials as secrets in the space settings.
Think of secrets like environment variables for the space app,
and in fact, they are accessible inside the app exactly as environment variables without exposing them publically.

Go to the repository you created in step 1, click `Settings`, and click `New Secret`.
Enter `QDRANT_API_KEY` and `QDRANT_URL` as secret names and respected values in the form.

Now we are ready for deploying the app.
HF Spaces deploys from the branch named `main` so we will first checkout that branch. Then, we will push to the remote named `hf`,
which we added in step 1, instead of `origin`.

```shell
git checkout main

git push hf main
```

Go to your HF Spaces repository,
and you'll see that your app is building.
Once it's finished in a few seconds,
you can enjoy your semantic image search app and share it with everyone on the internet.

## Conclusion

In this post, I demonstrated how to use Qdrant Cloud in a HF Spaces app
to build a demo for your vector search solution quickly.
When combined together, it considerably reduces the burden in converting a vector search solution
into something concrete that users may try easily.

Fun fact: I found the preview image for this post by running a search query on the app described in this post,
and you can also [give it a try](https://huggingface.co/spaces/mys/qdrant-cloud-demo).
