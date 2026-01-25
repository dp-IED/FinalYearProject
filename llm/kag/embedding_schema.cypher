// ============================================================================
// Neo4j Schema for Window Embeddings and Similarity Relationships
// ============================================================================
// 
// This schema extends the existing Window and Sensor nodes with embedding
// properties and adds ClassCenter nodes and similarity relationships.
//
// Run these statements in Neo4j Browser or via Cypher shell to set up the schema.
// ============================================================================

// ----------------------------------------------------------------------------
// Window Node Properties
// ----------------------------------------------------------------------------
// Add embedding properties to existing Window nodes:
//   - embedding: list[float] (32 elements) - 32-dimensional embedding vector
//   - dist_normal: float - Euclidean distance to normal center
//   - dist_anomalous: float - Euclidean distance to anomalous center
//   - confidence: float - Confidence score (sigmoid of distance difference)
//   - predicted_class: string - "normal" or "anomalous" based on distances

// Note: Window nodes are created by Neo4jLoader.load_window_sensors()
// These properties will be added when syncing embeddings (see sync_embeddings_to_neo4j)


// ----------------------------------------------------------------------------
// ClassCenter Nodes
// ----------------------------------------------------------------------------
// Create ClassCenter nodes for normal and anomalous class centers

// Create unique constraint on ClassCenter.class
CREATE CONSTRAINT classcenter_class_unique IF NOT EXISTS
FOR (c:ClassCenter)
REQUIRE c.class IS UNIQUE;

// Create index on ClassCenter.class for efficient lookups
CREATE INDEX classcenter_class_idx IF NOT EXISTS
FOR (c:ClassCenter)
ON (c.class);


// ----------------------------------------------------------------------------
// ClassCenter Node Structure
// ----------------------------------------------------------------------------
// Properties:
//   - class: string - "normal" or "anomalous"
//   - embedding: list[float] (32 elements) - 32-dimensional center embedding
//   - mean_radius: float - Average distance to class members
//
// Example creation (will be done programmatically):
// CREATE (c:ClassCenter {
//   class: "normal",
//   embedding: [0.1, 0.2, ...],  // 32 floats
//   mean_radius: 0.085
// })


// ----------------------------------------------------------------------------
// SIMILAR_TO Relationship
// ----------------------------------------------------------------------------
// Connects Window nodes that are similar in embedding space
// Properties:
//   - similarity: float - Cosine similarity (0-1)
//   - distance: float - Euclidean distance
//   - same_class: boolean - Whether both windows have same predicted_class

// Example:
// MATCH (w1:Window {idx: 0})
// MATCH (w2:Window {idx: 1})
// CREATE (w1)-[s:SIMILAR_TO {
//   similarity: 0.95,
//   distance: 0.12,
//   same_class: true
// }]->(w2)


// ----------------------------------------------------------------------------
// DISTANCE_TO_CENTER Relationship
// ----------------------------------------------------------------------------
// Connects Window nodes to ClassCenter nodes
// Properties:
//   - distance: float - Euclidean distance from window to center
//   - z_score: float - Z-score: (distance - mean_radius) / std_radius

// Example:
// MATCH (w:Window {idx: 0})
// MATCH (c:ClassCenter {class: "normal"})
// CREATE (w)-[d:DISTANCE_TO_CENTER {
//   distance: 0.095,
//   z_score: 0.33
// }]->(c)


// ----------------------------------------------------------------------------
// Indexes for Efficient Queries
// ----------------------------------------------------------------------------

// Index on Window.idx (should already exist, but ensure it's unique)
CREATE CONSTRAINT window_idx_unique IF NOT EXISTS
FOR (w:Window)
REQUIRE w.idx IS UNIQUE;

// Index on Window.predicted_class for filtering
CREATE INDEX window_predicted_class_idx IF NOT EXISTS
FOR (w:Window)
ON (w.predicted_class);

// Index on SIMILAR_TO similarity for sorting
CREATE INDEX similar_to_similarity_idx IF NOT EXISTS
FOR ()-[s:SIMILAR_TO]-()
ON (s.similarity);

// Index on DISTANCE_TO_CENTER distance for filtering
CREATE INDEX distance_to_center_distance_idx IF NOT EXISTS
FOR ()-[d:DISTANCE_TO_CENTER]-()
ON (d.distance);


// ----------------------------------------------------------------------------
// Example Queries
// ----------------------------------------------------------------------------

// Find k most similar windows to a given window:
// MATCH (w1:Window {idx: 0})-[s:SIMILAR_TO]->(w2:Window)
// RETURN w2.idx, s.similarity, s.distance
// ORDER BY s.similarity DESC
// LIMIT 5

// Find windows within distance threshold of anomalous center:
// MATCH (w:Window)-[d:DISTANCE_TO_CENTER]->(c:ClassCenter {class: "anomalous"})
// WHERE d.distance < 0.2 AND w.predicted_class = "anomalous"
// RETURN w.idx, d.distance
// ORDER BY d.distance ASC

// Get all windows with their distances to both centers:
// MATCH (w:Window)
// OPTIONAL MATCH (w)-[d1:DISTANCE_TO_CENTER]->(c1:ClassCenter {class: "normal"})
// OPTIONAL MATCH (w)-[d2:DISTANCE_TO_CENTER]->(c2:ClassCenter {class: "anomalous"})
// RETURN w.idx, d1.distance AS dist_normal, d2.distance AS dist_anomalous, w.confidence
// ORDER BY w.idx
