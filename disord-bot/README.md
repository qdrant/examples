# Discord RAG Bot

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/qdrant/examples/blob/master/disord-bot/discord_rag_bot.ipynb)

Build a Discord bot that answers questions from Qdrant documentation with Qdrant,
CAMEL-AI, and OpenAI. The notebook contains the full application code and pinned
dependencies from the [tutorial](https://qdrant.tech/documentation/tutorials-build-essentials/agentic-rag-camelai-discord/).

Use a fresh Python 3 CPU runtime with Python 3.12 or newer; no GPU is required.
The dependency set is tested with Python 3.12. Allow about 45 minutes, excluding
account setup. You need an OpenAI API key with access to
`text-embedding-3-large` and `gpt-4o-mini`, plus a Qdrant Cloud endpoint and key.
Ingestion and questions make billable OpenAI API calls.

1. Run notebook Sections 1 through 4 to install dependencies, enter credentials
   through hidden prompts when environment values are missing, create a dedicated
   collection, and ingest six documentation pages. Skip ingestion when reusing a
   completed collection.
2. Run Sections 5 through 7 to inspect retrieved evidence, ask a supported
   question, and check an unsupported question.
3. Optionally configure a Discord application and test server, enable Message
   Content Intent, and set `START_DISCORD_BOT = True` in Section 8.
4. Send the documented test messages yourself in the test channel.
5. Run Section 9 to stop the bot. The notebook leaves the Qdrant collection
   intact for reuse.

The notebook uses the existing event loop for the Discord client. The bot stays
online only while its background task and notebook runtime remain active. Use the
tutorial's local script on a persistent host for continuous availability.

Do not put credentials into code cells or commit them in notebook outputs.
Before sharing an executed notebook, clear its outputs.
