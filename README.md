# Pinecone Query Plugin 🌲

This plugin allows you to query your Pinecone vector database using natural language, providing powerful and flexible search capabilities.

**Key Features:**

*   **Natural Language Queries:** Search your Pinecone database using natural language questions or keywords.
*   **OpenAI Embeddings:** Utilizes the OpenAI embeddings API to vectorize your queries, enabling semantic search.
*   **Customizable Metadata Retrieval:** Specify which metadata fields to retrieve (e.g., text, filename), or leave blank to retrieve all available metadata.
*   **Cohere Rerank (Optional):** Enhance search result relevance by enabling Cohere Rerank, which reorders results based on their relevance to the query.
*   **Configurable Parameters:** Fine-tune your search with options for top K results, Pinecone API version, embedding dimensions, and namespace.
*   **Relevance Ranking:** Even if Cohere Rerank is disabled, the plugin sorts the results based on relevance calculated from similarity metric scores.

**How to Use:**

1.  **Plugin Settings:**
    *   Enter your Pinecone API key and index host URL.
    *   Provide your OpenAI API key and choose your desired OpenAI embedding model.
    *   Optionally, configure the top K results, Pinecone API version, embedding dimensions, and namespace.
    *   Specify the metadata fields you want to retrieve (comma-separated list, e.g., `text,filename`). Leave blank to retrieve all metadata. Defaults to `text`.
    *   **To enable Cohere Rerank:**
        *   Set "Enable Rerank" to `true`.
        *   Provide your Cohere API key.
        *   Optionally, configure the Cohere Rerank model, top N results, and max tokens per document.
    *   Choose the similarity metric used in Pinecone if rerank is disabled.
2.  **Usage:** In your chat, simply ask the AI to *query Pinecone* about a topic, question, or keyword. The plugin will search your Pinecone database and return the most relevant results. You can also use a system prompt to instruct the AI to always use this plugin with every response or when encountering specific topics or keywords.

**Example:**

> User: "Query Pinecone for documents related to the dangers of AI"

**Important Notes:**

*   Ensure your Pinecone index is properly set up and contains the data you want to search.
*   The plugin uses OpenAI embedding models API to embed your queries. Other providers are not currently supported.
*   Your OpenAI embedding model and dimension should match those used to embed your Pinecone database.
*   If using Cohere Rerank, ensure you have a valid Cohere API key.
*   The plugin provides a relevance score for each result, which is normalized based on the selected similarity metric or Cohere Rerank.
