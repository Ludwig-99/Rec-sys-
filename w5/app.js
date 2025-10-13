// app.js
class FriendRecommenderApp {
    constructor() {
        this.graph = { nodes: [], edges: [] };
        this.pagerankScores = [];
        this.currentSelectedNode = null;
        this.svg = null;
        this.colorScale = null;
        this.radiusScale = null;
        
        this.init();
    }

    async init() {
        try {
            await this.loadData('data/karate.csv');
            this.pagerankScores = await computePageRank(this.graph);
            this.updateTable();
            this.renderGraph();
            this.setupEventListeners();
        } catch (error) {
            console.error('Error initializing app:', error);
        }
    }

    async loadData(filePath) {
        try {
            const data = await d3.csv(filePath);
            this.graph.edges = data.map(d => ({
                source: +d.source,
                target: +d.target
            }));

            // Extract unique nodes
            const nodeSet = new Set();
            this.graph.edges.forEach(edge => {
                nodeSet.add(edge.source);
                nodeSet.add(edge.target);
            });

            this.graph.nodes = Array.from(nodeSet).map(id => ({ id }));
            
            console.log(`Loaded ${this.graph.nodes.length} nodes and ${this.graph.edges.length} edges`);
        } catch (error) {
            console.error('Error loading data:', error);
            throw error;
        }
    }

    updateTable() {
        const tbody = d3.select('#nodes-table tbody');
        tbody.selectAll('*').remove();

        // Sort nodes by PageRank score (descending)
        const sortedNodes = this.graph.nodes.map(node => ({
            ...node,
            pagerank: this.pagerankScores[node.id] || 0
        })).sort((a, b) => b.pagerank - a.pagerank);

        const rows = tbody.selectAll('tr')
            .data(sortedNodes)
            .enter()
            .append('tr')
            .on('click', (event, d) => this.handleNodeClick(d.id));

        rows.append('td').text(d => d.id);
        rows.append('td').text(d => this.pagerankScores[d.id]?.toFixed(4) || '0.0000');
        rows.append('td').text(d => this.getCurrentFriends(d.id).join(', '));
    }

    getCurrentFriends(nodeId) {
        return this.graph.edges
            .filter(edge => edge.source === nodeId || edge.target === nodeId)
            .map(edge => edge.source === nodeId ? edge.target : edge.source)
            .sort((a, b) => a - b);
    }

    handleNodeClick(nodeId) {
        this.currentSelectedNode = nodeId;
        
        // Update table selection
        d3.selectAll('#nodes-table tr').classed('selected', false);
        d3.selectAll('#nodes-table tr').filter(d => d.id === nodeId).classed('selected', true);

        // Update graph highlighting
        this.updateGraphHighlighting(nodeId);

        // Show recommendations
        this.showRecommendations(nodeId);
    }

    updateGraphHighlighting(selectedNodeId) {
        if (!this.svg) return;

        this.svg.selectAll('.node')
            .classed('selected', d => d.id === selectedNodeId)
            .attr('r', d => d.id === selectedNodeId ? 
                this.radiusScale(this.pagerankScores[d.id]) * 1.5 : 
                this.radiusScale(this.pagerankScores[d.id]));

        this.svg.selectAll('.link')
            .classed('highlighted', d => 
                d.source.id === selectedNodeId || d.target.id === selectedNodeId)
            .attr('stroke-width', d => 
                (d.source.id === selectedNodeId || d.target.id === selectedNodeId) ? 3 : 1);
    }

    showRecommendations(nodeId) {
        const currentFriends = this.getCurrentFriends(nodeId);
        const allNodes = this.graph.nodes.map(node => node.id);
        
        // Get potential friends (not current friends and not self)
        const potentialFriends = allNodes.filter(id => 
            id !== nodeId && !currentFriends.includes(id)
        );

        // Sort by PageRank score (descending) and take top 3
        const recommendations = potentialFriends
            .map(id => ({
                id,
                score: this.pagerankScores[id] || 0,
                currentFriends: this.getCurrentFriends(id).length
            }))
            .sort((a, b) => b.score - a.score)
            .slice(0, 3);

        const recommendationDiv = d3.select('#recommendations');
        const listDiv = d3.select('#recommendation-list');

        if (recommendations.length > 0) {
            listDiv.html(`
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px;">
                    ${recommendations.map(rec => `
                        <div style="background: rgba(255,255,255,0.2); padding: 10px; border-radius: 8px;">
                            <strong>User ${rec.id}</strong><br>
                            Score: ${rec.score.toFixed(4)}<br>
                            Friends: ${rec.currentFriends}
                            <button onclick="app.simulateConnection(${nodeId}, ${rec.id})">
                                Connect 🤝
                            </button>
                        </div>
                    `).join('')}
                </div>
            `);
            recommendationDiv.style('display', 'block');
        } else {
            recommendationDiv.style('display', 'none');
        }

        // Update selected node info
        d3.select('#selected-node-info').html(`
            <p><strong>Selected User: ${nodeId}</strong></p>
            <p>PageRank Score: ${(this.pagerankScores[nodeId] || 0).toFixed(4)}</p>
            <p>Current Friends: ${currentFriends.join(', ') || 'None'}</p>
        `);
    }

    simulateConnection(sourceId, targetId) {
        // Add new edge
        this.graph.edges.push({ source: sourceId, target: targetId });
        
        // Recompute PageRank
        computePageRank(this.graph).then(newScores => {
            this.pagerankScores = newScores;
            this.updateTable();
            this.renderGraph();
            this.showRecommendations(this.currentSelectedNode);
            
            console.log(`Simulated connection between ${sourceId} and ${targetId}`);
        });
    }

    renderGraph() {
        const container = d3.select('#graph-container');
        container.select('svg').selectAll('*').remove();

        const width = container.node().getBoundingClientRect().width;
        const height = 500;

        this.svg = container.select('svg')
            .attr('width', width)
            .attr('height', height);

        // Create scales
        const scores = this.graph.nodes.map(node => this.pagerankScores[node.id] || 0);
        this.colorScale = d3.scaleSequential(d3.interpolatePlasma)
            .domain([0, d3.max(scores)]);
        this.radiusScale = d3.scaleSqrt()
            .domain([0, d3.max(scores)])
            .range([5, 20]);

        // Create force simulation
        const simulation = d3.forceSimulation(this.graph.nodes)
            .force('link', d3.forceLink(this.graph.edges).id(d => d.id).distance(100))
            .force('charge', d3.forceManyBody().strength(-300))
            .force('center', d3.forceCenter(width / 2, height / 2))
            .force('collision', d3.forceCollide().radius(d => this.radiusScale(this.pagerankScores[d.id]) + 5));

        // Draw links
        const link = this.svg.append('g')
            .selectAll('line')
            .data(this.graph.edges)
            .enter().append('line')
            .attr('class', 'link')
            .attr('stroke-width', 1);

        // Draw nodes
        const node = this.svg.append('g')
            .selectAll('circle')
            .data(this.graph.nodes)
            .enter().append('circle')
            .attr('class', 'node')
            .attr('r', d => this.radiusScale(this.pagerankScores[d.id] || 0))
            .attr('fill', d => this.colorScale(this.pagerankScores[d.id] || 0))
            .call(d3.drag()
                .on('start', (event, d) => {
                    if (!event.active) simulation.alphaTarget(0.3).restart();
                    d.fx = d.x;
                    d.fy = d.y;
                })
                .on('drag', (event, d) => {
                    d.fx = event.x;
                    d.fy = event.y;
                })
                .on('end', (event, d) => {
                    if (!event.active) simulation.alphaTarget(0);
                    d.fx = null;
                    d.fy = null;
                }))
            .on('click', (event, d) => this.handleNodeClick(d.id));

        // Add labels
        const label = this.svg.append('g')
            .selectAll('text')
            .data(this.graph.nodes)
            .enter().append('text')
            .text(d => d.id)
            .attr('font-size', 12)
            .attr('dx', 15)
            .attr('dy', 4)
            .attr('fill', '#2c3e50')
            .attr('font-weight', 'bold');

        // Update simulation
        simulation.on('tick', () => {
            link
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);

            node
                .attr('cx', d => d.x)
                .attr('cy', d => d.y);

            label
                .attr('x', d => d.x)
                .attr('y', d => d.y);
        });

        // Add tooltips
        node.append('title')
            .text(d => `User ${d.id}\nPageRank: ${(this.pagerankScores[d.id] || 0).toFixed(4)}`);
    }

    setupEventListeners() {
        // Additional event listeners can be added here if needed
    }
}

// Initialize the app when the page loads
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new FriendRecommenderApp();
});
