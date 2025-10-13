// graph.js
class GraphRenderer {
    constructor(containerId) {
        this.containerId = containerId;
        this.svg = null;
        this.simulation = null;
        this.nodes = [];
        this.links = [];
        this.width = 800;
        this.height = 500;
        this.selectedNode = null;
        
        this.initializeSVG();
    }

    initializeSVG() {
        const container = document.getElementById(this.containerId);
        if (!container) {
            console.error(`Container with id '${this.containerId}' not found`);
            return;
        }

        // Clear container
        container.innerHTML = '';
        
        this.width = container.clientWidth || 800;
        this.height = container.clientHeight || 500;

        this.svg = d3.select(`#${this.containerId}`)
            .append('svg')
            .attr('width', this.width)
            .attr('height', this.height)
            .attr('viewBox', [0, 0, this.width, this.height])
            .style('background', 'transparent');

        // Add zoom behavior
        const zoom = d3.zoom()
            .scaleExtent([0.1, 4])
            .on('zoom', (event) => {
                this.svg.select('g').attr('transform', event.transform);
            });

        this.svg.call(zoom)
            .append('g');
    }

    renderGraph(graph, pageRankScores = null) {
        if (!this.svg) {
            this.initializeSVG();
        }

        this.nodes = graph.nodes.map(node => ({
            ...node,
            pageRank: pageRankScores ? (pageRankScores[node.id] || 0.001) : 0.001
        }));

        this.links = graph.edges.map(edge => ({
            source: this.nodes.find(n => n.id === edge.source),
            target: this.nodes.find(n => n.id === edge.target)
        })).filter(link => link.source && link.target);

        this.updateGraph();
    }

    updateGraph() {
        const g = this.svg.select('g');
        
        // Clear existing elements
        g.selectAll('*').remove();

        if (this.nodes.length === 0) return;

        // Create force simulation
        this.simulation = d3.forceSimulation(this.nodes)
            .force('link', d3.forceLink(this.links).id(d => d.id).distance(80))
            .force('charge', d3.forceManyBody().strength(-200))
            .force('center', d3.forceCenter(this.width / 2, this.height / 2))
            .force('collision', d3.forceCollide().radius(d => this.getNodeRadius(d.pageRank) + 5));

        // Create links
        const link = g.append('g')
            .selectAll('line')
            .data(this.links)
            .enter()
            .append('line')
            .attr('class', 'link')
            .attr('stroke', '#7f8c8d')
            .attr('stroke-opacity', 0.6)
            .attr('stroke-width', 1);

        // Create nodes
        const node = g.append('g')
            .selectAll('circle')
            .data(this.nodes)
            .enter()
            .append('circle')
            .attr('class', 'node')
            .attr('r', d => this.getNodeRadius(d.pageRank))
            .attr('fill', d => this.getNodeColor(d.pageRank))
            .attr('stroke', '#fff')
            .attr('stroke-width', 2)
            .call(d3.drag()
                .on('start', (event, d) => this.dragStarted(event, d))
                .on('drag', (event, d) => this.dragged(event, d))
                .on('end', (event, d) => this.dragEnded(event, d)))
            .on('click', (event, d) => this.nodeClicked(event, d));

        // Add node labels
        const label = g.append('g')
            .selectAll('text')
            .data(this.nodes)
            .enter()
            .append('text')
            .text(d => d.id)
            .attr('font-size', '12px')
            .attr('font-weight', 'bold')
            .attr('dx', 15)
            .attr('dy', 4)
            .attr('pointer-events', 'none')
            .attr('fill', '#2c3e50');

        // Add tooltips
        node.append('title')
            .text(d => `User ${d.id}\nPageRank: ${d.pageRank.toFixed(4)}\nFriends: ${this.getFriendCount(d.id)}`);

        // Update positions on simulation tick
        this.simulation.on('tick', () => {
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

        // Highlight selected node if any
        if (this.selectedNode !== null) {
            this.highlightNode(this.selectedNode);
        }
    }

    getNodeRadius(pageRank) {
        // Scale radius based on PageRank (6 to 25 pixels)
        const baseRadius = 6;
        const maxRadius = 25;
        return baseRadius + (pageRank * 150);
    }

    getNodeColor(pageRank) {
        // Color scale from blue (low) to purple (high) with Frutiger Aero style
        const intensity = Math.min(pageRank * 8, 1);
        
        if (intensity < 0.33) {
            // Blue to teal
            const r = Math.floor(116 - intensity * 50);
            const g = Math.floor(185 + intensity * 70);
            const b = Math.floor(255 - intensity * 50);
            return `rgb(${r}, ${g}, ${b})`;
        } else if (intensity < 0.66) {
            // Teal to green
            const r = Math.floor(66 - (intensity - 0.33) * 100);
            const g = Math.floor(255 - (intensity - 0.33) * 50);
            const b = Math.floor(205 - (intensity - 0.33) * 100);
            return `rgb(${r}, ${g}, ${b})`;
        } else {
            // Green to purple
            const r = Math.floor(166 + (intensity - 0.66) * 89);
            const g = Math.floor(205 - (intensity - 0.66) * 155);
            const b = Math.floor(105 + (intensity - 0.66) * 150);
            return `rgb(${r}, ${g}, ${b})`;
        }
    }

    getFriendCount(nodeId) {
        const node = this.nodes.find(n => n.id === nodeId);
        return node ? this.links.filter(link => 
            link.source.id === nodeId || link.target.id === nodeId
        ).length : 0;
    }

    dragStarted(event, d) {
        if (!event.active) this.simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }

    dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }

    dragEnded(event, d) {
        if (!event.active) this.simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }

    nodeClicked(event, d) {
        event.stopPropagation();
        this.highlightNode(d.id);
        
        // Notify app about node selection
        if (window.app) {
            window.app.selectNode(d.id);
        }
    }

    highlightNode(nodeId) {
        this.selectedNode = nodeId;
        
        // Update node appearance
        this.svg.selectAll('.node')
            .classed('selected', d => d.id === nodeId)
            .attr('stroke-width', d => d.id === nodeId ? 4 : 2);

        // Highlight connected links
        this.svg.selectAll('.link')
            .classed('highlighted', d => 
                d.source.id === nodeId || d.target.id === nodeId)
            .attr('stroke-width', d => 
                (d.source.id === nodeId || d.target.id === nodeId) ? 3 : 1);
    }

    updatePageRankScores(pageRankScores) {
        if (!pageRankScores) return;
        
        this.nodes.forEach(node => {
            node.pageRank = pageRankScores[node.id] || 0.001;
        });
        
        this.updateGraph();
    }
}

// Initialize graph renderer when page loads
let graphRenderer;
document.addEventListener('DOMContentLoaded', () => {
    graphRenderer = new GraphRenderer('graph');
    window.graphRenderer = graphRenderer;
});
