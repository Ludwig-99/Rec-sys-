// graph.js
// Note: The graph rendering functionality has been integrated into app.js
// This file is kept for modularity but the main graph code is in app.js

/**
 * Graph utility functions
 */
class GraphUtils {
    /**
     * Finds common neighbors between two nodes
     */
    static getCommonNeighbors(graph, node1, node2) {
        const neighbors1 = new Set(
            graph.edges
                .filter(edge => edge.source === node1 || edge.target === node1)
                .map(edge => edge.source === node1 ? edge.target : edge.source)
        );
        
        const neighbors2 = new Set(
            graph.edges
                .filter(edge => edge.source === node2 || edge.target === node2)
                .map(edge => edge.source === node2 ? edge.target : edge.source)
        );
        
        return [...neighbors1].filter(node => neighbors2.has(node));
    }

    /**
     * Calculates Jaccard similarity between two nodes
     */
    static jaccardSimilarity(graph, node1, node2) {
        const neighbors1 = new Set(
            graph.edges
                .filter(edge => edge.source === node1 || edge.target === node1)
                .map(edge => edge.source === node1 ? edge.target : edge.source)
        );
        
        const neighbors2 = new Set(
            graph.edges
                .filter(edge => edge.source === node2 || edge.target === node2)
                .map(edge => edge.source === node2 ? edge.target : edge.source)
        );
        
        const intersection = [...neighbors1].filter(node => neighbors2.has(node)).length;
        const union = new Set([...neighbors1, ...neighbors2]).size;
        
        return union === 0 ? 0 : intersection / union;
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { GraphUtils };
}
