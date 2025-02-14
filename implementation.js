async function vectorizeQuery(query, userSettings) {
  const { openaiAPIKey, openaiEmbeddingModel, embeddingDimensions } = userSettings;
  const headers = { "Content-Type": "application/json", "Authorization": `Bearer ${openaiAPIKey}` };
  const requestBody = { "input": query, "model": openaiEmbeddingModel, "truncate": "END" };
  if (embeddingDimensions) { requestBody.dimensions = parseInt(embeddingDimensions); }
  try {
    const response = await fetch("https://api.openai.com/v1/embeddings", { method: "POST", headers: headers, body: JSON.stringify(requestBody) });
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(`OpenAI API Error: ${response.status} - ${JSON.stringify(errorData)}`);
    }
    const data = await response.json();
    if (data.data && data.data.length > 0 && data.data[0].embedding) { return data.data[0].embedding; } else { throw new Error("Invalid response from OpenAI API: Embedding not found"); }
  } catch (error) {
    console.error("Error vectorizing query:", error);
    throw new Error(`Failed to vectorize query: ${error.message}`);
  }
}

async function rerankWithCohere(query, pineconeResults, userSettings) {
  const { cohereAPIKey, cohereRerankModel, cohereTopN, cohereMaxTokensPerDoc, metadataFields } = userSettings;
  const topN = parseInt(cohereTopN || 5);
  const maxTokens = parseInt(cohereMaxTokensPerDoc || 4096);
  const model = cohereRerankModel || "rerank-v3.5";
  const fields = metadataFields ? metadataFields.split(',').map(field => field.trim()) : null;
  const documents = pineconeResults.map(match => match.metadata.text);
  const headers = { "Content-Type": "application/json", "Authorization": `Bearer ${cohereAPIKey}`, "X-Client-Name": "TypingMindPlugin" };
  const requestBody = { "model": model, "query": query, "documents": documents, "top_n": topN, "max_tokens_per_doc": maxTokens };
  try {
    const response = await fetch("https://api.cohere.com/v2/rerank", { method: "POST", headers: headers, body: JSON.stringify(requestBody) });
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(`Cohere API Error: ${response.status} - ${JSON.stringify(errorData)}`);
    }
    const data = await response.json();
    if (data && data.results) {
      return data.results.map(result => {
        const originalIndex = result.index;
        const match = pineconeResults[originalIndex];
        const metadata = fields ? Object.fromEntries(fields.map(field => [field, match.metadata[field] || null])) : match.metadata;
        return { relevance: result.relevance_score.toFixed(2), metadata: metadata };
      });
    } else { throw new Error("Invalid response from Cohere API: Rerank results not found"); }
  } catch (error) {
    console.error("Error reranking with Cohere:", error);
    throw new Error(`Failed to rerank with Cohere: ${error.message}`);
  }
}

function normalizeRelevanceScore(score, metric, matches) {
  const calculateStats = (scores) => {
    const mean = scores.reduce((a, b) => a + b, 0) / scores.length;
    const variance = scores.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / scores.length;
    const stdDev = Math.sqrt(variance);
    return { mean, stdDev };
  };
  const minMaxNormalize = (score, min, max) => {
    if (max === min) return 1.0;
    return (score - min) / (max - min);
  };
  switch (metric) {
    case 'cosine': {
      const normalizedBase = (1 + score) / 2;
      return Math.pow(normalizedBase, 2).toFixed(4);
    }
    case 'euclidean': {
      const scores = matches.map(m => m.score);
      const { mean, stdDev } = calculateStats(scores);
      const gaussian = Math.exp(-Math.pow(score - mean, 2) / (2 * Math.pow(stdDev || 1, 2)));
      return (1 - gaussian).toFixed(4);
    }
    case 'dotproduct': {
      const scores = matches.map(m => m.score);
      const { mean, stdDev } = calculateStats(scores);
      const temperature = stdDev || 1;
      const shifted = (score - mean) / temperature;
      return (1 / (1 + Math.exp(-2 * shifted))).toFixed(4);
    }
    case 'other': {
      const scores = matches.map(m => m.score);
      const sortedScores = [...scores].sort((a, b) => a - b);
      const q1 = sortedScores[Math.floor(sortedScores.length * 0.25)];
      const q3 = sortedScores[Math.floor(sortedScores.length * 0.75)];
      const iqr = q3 - q1;
      const validMin = q1 - 1.5 * iqr;
      const validMax = q3 + 1.5 * iqr;
      return minMaxNormalize(Math.max(validMin, Math.min(validMax, score)), validMin, validMax).toFixed(4);
    }
    default:
      return score.toFixed(4);
  }
}

async function query_pinecone_database(params, userSettings) {
  const { query } = params;
  const { pineconeAPIKey, pineconeIndexHostURL, topK, pineconeAPIVersion, namespace, enableRerank, similarityMetric, metadataFields } = userSettings;
  const topKValue = parseInt(topK || 5);
  const apiVersion = pineconeAPIVersion || "2024-10";
  const rerankEnabled = (enableRerank || "false") === "true";
  const metric = similarityMetric || 'cosine';
  const fields = metadataFields ? metadataFields.split(',').map(field => field.trim()) : null;
  try {
    const vector = await vectorizeQuery(query, userSettings);
    const headers = { "Api-Key": pineconeAPIKey, "Content-Type": "application/json", "X-Pinecone-API-Version": apiVersion };
    const requestBody = { "vector": vector, "topK": topKValue, "includeValues": false, "includeMetadata": true };
    if (namespace) { requestBody.namespace = namespace; }
    const response = await fetch(`${pineconeIndexHostURL}/query`, { method: "POST", headers: headers, body: JSON.stringify(requestBody) });
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(`Pinecone API Error: ${response.status} - ${JSON.stringify(errorData)}`);
    }
    const data = await response.json();
    if (data && data.matches) {
      if (rerankEnabled) { return await rerankWithCohere(query, data.matches, userSettings); } else {
        return data.matches.map(match => {
          const metadata = fields ? Object.fromEntries(fields.map(field => [field, match.metadata[field] || null])) : match.metadata;
          return { relevance: normalizeRelevanceScore(match.score, metric, data.matches), metadata: metadata };
        });
      }
    } else { return "No matching results found in Pinecone."; }
  } catch (error) {
    console.error("Error querying Pinecone:", error);
    throw new Error(`Failed to query Pinecone: ${error.message}`);
  }
}
