// app.js
class PageRankApp {
    constructor() {
        this.graph = null;
        this.pageRankScores = null;
        this.selectedNode = null;
        this.isComputing = false;
        
        this.initializeEventListeners();
        this.loadDefaultData();
    }

    initializeEventListeners() {
        document.getElementById('computeBtn').addEventListener('click', () => this.computePageRank());
        document.getElementById('resetBtn').addEventListener('click', () => this.resetGraph());
    }

    async loadDefaultData() {
        try {
            // Load the karate club dataset
            const response = await fetch('data/karate.csv');
            if (!response.ok) throw new Error('Failed to load data file');
            
            const csvText = await response.text();
            this.graph = this.parseCSVToGraph(csvText);
            
            // Initialize graph renderer if available
            if (window.graphRenderer) {
                window.graphRenderer.renderGraph(this.graph);
            }
            
            this.updateTable();
        } catch (error) {
            console.error('Error loading data:', error);
            this.showError('Error loading graph data. Please check if data/karate.csv exists.');
        }
    }

    parseCSVToGraph(csvText) {
        const edges = [];
        const nodes = new Set();
        
        const lines = csvText.trim().split('\n');
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i].trim();
            if (!line) continue;
            
            const parts = line.split(',');
            if (parts.length < 2) continue;
            
            const source = parseInt(parts[0].trim());
            const target = parseInt(parts[1].trim());
            
            if (isNaN(source) || isNaN(target)) continue;
            
            edges.push({ source, target });
            nodes.add(source);
            nodes.add(target);
        }

        // Create adjacency list
        const adjacencyList = {};
        Array.from(nodes).sort((a, b) => a - b).forEach(node => {
            adjacencyList[node] = [];
        });

        edges.forEach(edge => {
            if (!adjacencyList[edge.source].includes(edge.target)) {
                adjacencyList[edge.source].push(edge.target);
            }
            if (!adjacencyList[edge.target].includes(edge.source)) {
                adjacencyList[edge.target].push(edge.source);
            }
        });

        return {
            nodes: Array.from(nodes).sort((a, b) => a - b).map(id => ({ id })),
            edges: edges,
            adjacencyList: adjacencyList
        };
    }

    async computePageRank() {
        if (this.isComputing || !this.graph) return;
        
        this.isComputing = true;
        const computeBtn = document.getElementById('computeBtn');
        computeBtn.disabled = true;
        computeBtn.textContent = 'Computing...';

        try {
            this.pageRankScores = await computePageRank(this.graph.adjacencyList, 50, 0.85);
            this.updateTable();
            
            // Update graph visualization with new scores
            if (window.graphRenderer) {
                window.graphRenderer.updatePageRankScores(this.pageRankScores);
            }
            
            // Refresh node details if a node is selected
            if (this.selectedNode !== null) {
                this.showNodeDetails(this.selectedNode);
            }
        } catch (error) {
            console.error('Error computing PageRank:', error);
            this.showError('Error computing PageRank scores. Check console for details.');
        } finally {
            this.isComputing = false;
            computeBtn.disabled = false;
            computeBtn.textContent = 'Compute PageRank';
        }
    }

    resetGraph() {
        this.pageRankScores = null;
        this.selectedNode = null;
        
        // Clear selections
        document.querySelectorAll('#tableBody tr').forEach(row => {
            row.classList.remove('selected');
        });
        
        document.getElementById('nodeDetails').innerHTML = `
            <div class="node-info">
                <p>Select a node from the graph or table to view details</p>
            </div>
        `;
        
        this.loadDefaultData();
    }

    updateTable() {
        const tableBody = document.getElementById('tableBody');
        tableBody.innerHTML = '';

        if (!this.graph) return;

        // Create array of nodes with their scores and sort by PageRank (descending)
        const nodesWithScores = this.graph.nodes.map(node => ({
            ...node,
            score: this.pageRankScores ? this.pageRankScores[node.id] || 0 : 0,
            friends: this.graph.adjacencyList[node.id] || []
        }));

        // Sort by PageRank score (descending)
        nodesWithScores.sort((a, b) => b.score - a.score);

        nodesWithScores.forEach(nodeData => {
            const row = document.createElement('tr');
            row.dataset.nodeId = nodeData.id;
            
            const pageRank = this.pageRankScores ? 
                nodeData.score.toFixed(4) : 'N/A';
            
            row.innerHTML = `
                <td>${nodeData.id}</td>
                <td>${pageRank}</td>
                <td>${nodeData.friends.join(', ')}</td>
            `;
            
            row.addEventListener('click', () => this.selectNode(nodeData.id));
            
            if (this.selectedNode === nodeData.id) {
                row.classList.add('selected');
            }
            
            tableBody.appendChild(row);
        });
    }

    selectNode(nodeId) {
        this.selectedNode = nodeId;
        
        // Update table selection
        document.querySelectorAll('#tableBody tr').forEach(row => {
            const rowNodeId = parseInt(row.dataset.nodeId);
            row.classList.toggle('selected', rowNodeId === nodeId);
        });
        
        // Update graph selection
        if (window.graphRenderer) {
            window.graphRenderer.highlightNode(nodeId);
        }
        
        this.showNodeDetails(nodeId);
    }

    showNodeDetails(nodeId) {
        const nodeDetails = document.getElementById('nodeDetails');
        const friends = this.graph.adjacencyList[nodeId] || [];
        
        let recommendationsHtml = '';
        if (this.pageRankScores) {
            const recommendations = this.getRecommendations(nodeId);
            if (recommendations.length > 0) {
                recommendationsHtml = `
                    <div class="recommendations">
                        <h4>🌟 Top 3 Friend Recommendations:</h4>
                        ${recommendations.map(rec => `
                            <div class="recommendation-item">
                                <div>
                                    <strong>User ${rec.node}</strong><br>
                                    PageRank: ${rec.score.toFixed(4)}<br>
                                    Current Friends: ${rec.currentFriends}
                                </div>
                                <button onclick="app.connectNodes(${nodeId}, ${rec.node})">
                                    Connect 🤝
                                </button>
                            </div>
                        `).join('')}
                    </div>
                `;
            } else {
                recommendationsHtml = '<p>No new friend recommendations available.</p>';
            }
        } else {
            recommendationsHtml = '<p>Compute PageRank to see recommendations.</p>';
        }
        
        nodeDetails.innerHTML = `
            <div class="node-info">
                <h4>👤 User ${nodeId}</h4>
                <p><strong>PageRank Score:</strong> ${this.pageRankScores ? this.pageRankScores[nodeId].toFixed(4) : 'N/A'}</p>
                <p><strong>Current Friends (${friends.length}):</strong> ${friends.length > 0 ? friends.join(', ') : 'None'}</p>
            </div>
            ${recommendationsHtml}
        `;
    }

    getRecommendations(nodeId) {
        if (!this.pageRankScores || !this.graph) return [];
        
        const currentFriends = new Set(this.graph.adjacencyList[nodeId] || []);
        currentFriends.add(nodeId); // Exclude self
        
        const recommendations = [];
        
        this.graph.nodes.forEach(node => {
            if (!currentFriends.has(node.id)) {
                recommendations.push({
                    node: node.id,
                    score: this.pageRankScores[node.id],
                    currentFriends: (this.graph.adjacencyList[node.id] || []).length
                });
            }
        });
        
        // Sort by PageRank score descending and take top 3
        return recommendations.sort((a, b) => b.score - a.score).slice(0, 3);
    }

    connectNodes(sourceId, targetId) {
        if (!this.graph) return;
        
        // Add edge to graph
        this.graph.edges.push({ source: sourceId, target: targetId });
        
        // Update adjacency list
        if (!this.graph.adjacencyList[sourceId].includes(targetId)) {
            this.graph.adjacencyList[sourceId].push(targetId);
            this.graph.adjacencyList[sourceId].sort((a, b) => a - b);
        }
        if (!this.graph.adjacencyList[targetId].includes(sourceId)) {
            this.graph.adjacencyList[targetId].push(sourceId);
            this.graph.adjacencyList[targetId].sort((a, b) => a - b);
        }
        
        // Recompute PageRank
        this.computePageRank();
        
        // Show success message
        this.showSuccess(`Connected user ${sourceId} with user ${targetId}! PageRank recomputed.`);
    }

    showError(message) {
        alert(`Error: ${message}`);
    }

    showSuccess(message) {
        // You could replace this with a nicer notification system
        console.log(`Success: ${message}`);
    }
}

// Initialize app when page loads
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new PageRankApp();
    window.app = app;
});
