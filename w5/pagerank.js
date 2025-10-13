// pagerank.js
/**
 * Computes PageRank scores for nodes in a graph using TensorFlow.js
 * @param {Object} graph - Graph object with nodes and edges
 * @param {number} damping - Damping factor (default: 0.85)
 * @param {number} iterations - Number of iterations (default: 50)
 * @returns {Promise<Array>} - Array of PageRank scores indexed by node id
 */
async function computePageRank(graph, damping = 0.85, iterations = 50) {
    const numNodes = graph.nodes.length;
    if (numNodes === 0) return [];

    // Create adjacency matrix and out-degree array
    const adjacency = Array(numNodes).fill().map(() => Array(numNodes).fill(0));
    const outDegree = Array(numNodes).fill(0);

    // Build adjacency matrix and calculate out-degrees
    graph.edges.forEach(edge => {
        const sourceIndex = graph.nodes.findIndex(n => n.id === edge.source);
        const targetIndex = graph.nodes.findIndex(n => n.id === edge.target);
        
        if (sourceIndex !== -1 && targetIndex !== -1) {
            adjacency[sourceIndex][targetIndex] = 1;
            adjacency[targetIndex][sourceIndex] = 1; // Undirected graph
            outDegree[sourceIndex]++;
            outDegree[targetIndex]++;
        }
    });

    // Convert to TensorFlow.js tensors
    const tf = window.tf;
    
    // Create transition probability matrix
    const transitionData = [];
    for (let i = 0; i < numNodes; i++) {
        const row = [];
        for (let j = 0; j < numNodes; j++) {
            if (outDegree[i] > 0) {
                row.push(adjacency[i][j] / outDegree[i]);
            } else {
                row.push(1 / numNodes); // Teleport for dangling nodes
            }
        }
        transitionData.push(row);
    }

    const transitionMatrix = tf.tensor2d(transitionData);
    
    // Initialize PageRank vector (uniform distribution)
    let pagerank = tf.ones([numNodes, 1]).div(tf.scalar(numNodes));
    
    // Damping factor components
    const dampingMatrix = tf.ones([numNodes, numNodes]).div(tf.scalar(numNodes));
    const dampingFactor = tf.scalar(damping);
    const teleportFactor = tf.scalar(1 - damping);
    
    // Power iteration
    for (let iter = 0; iter < iterations; iter++) {
        const term1 = transitionMatrix.matMul(pagerank).mul(dampingFactor);
        const term2 = dampingMatrix.matMul(pagerank).mul(teleportFactor);
        pagerank = term1.add(term2);
    }
    
    // Normalize the final PageRank vector
    const sum = pagerank.sum();
    pagerank = pagerank.div(sum);
    
    // Convert to JavaScript array and map back to node ids
    const pagerankArray = await pagerank.data();
    
    // Create a mapping from node id to PageRank score
    const scoreMap = {};
    graph.nodes.forEach((node, index) => {
        scoreMap[node.id] = pagerankArray[index];
    });
    
    // Clean up tensors
    transitionMatrix.dispose();
    pagerank.dispose();
    dampingMatrix.dispose();
    dampingFactor.dispose();
    teleportFactor.dispose();
    sum.dispose();
    
    return scoreMap;
}
