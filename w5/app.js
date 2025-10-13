// app.js
class PageRankApp {
    constructor() {
        this.graph = null;
        this.pageRankScores = null;
        this.selectedNode = null;
        this.isComputing = false;
        
        console.log('🔧 PageRankApp constructor called');
        
        // Обновляем статус
        this.updateGraphStatus('Initializing application...');
        
        // Даем время DOM полностью загрузиться
        setTimeout(() => {
            this.initializeEventListeners();
            this.loadDefaultData();
        }, 100);
    }

    updateGraphStatus(message) {
        const statusElement = document.getElementById('graphStatus');
        if (statusElement) {
            statusElement.innerHTML = message;
        }
    }

    initializeEventListeners() {
        console.log('🔧 Initializing event listeners...');
        
        const computeBtn = document.getElementById('computeBtn');
        const resetBtn = document.getElementById('resetBtn');
        
        if (computeBtn) {
            computeBtn.addEventListener('click', () => this.computePageRank());
            console.log('✅ Compute button listener added');
        } else {
            console.error('❌ Compute button not found!');
        }
        
        if (resetBtn) {
            resetBtn.addEventListener('click', () => this.resetGraph());
            console.log('✅ Reset button listener added');
        } else {
            console.error('❌ Reset button not found!');
        }
    }

    async loadDefaultData() {
        try {
            console.log('📥 Loading default data...');
            this.updateGraphStatus('Loading graph data...');
            
            let csvText;
            try {
                // Пробуем загрузить данные из файла
                const response = await fetch('data/karate.csv');
                if (!response.ok) throw new Error('Failed to load data file');
                csvText = await response.text();
                console.log('✅ CSV data loaded successfully');
            } catch (error) {
                console.warn('⚠️ Using demo data instead:', error);
                csvText = this.getDemoData();
            }
            
            this.graph = this.parseCSVToGraph(csvText);
            console.log('📊 Graph parsed:', this.graph);
            
            // Инициализируем граф
            if (window.graphRenderer) {
                this.updateGraphStatus('Rendering graph...');
                window.graphRenderer.renderGraph(this.graph);
                console.log('✅ Graph rendered');
                this.updateGraphStatus('✅ Graph ready! Click on nodes to see recommendations.');
            } else {
                console.error('❌ Graph renderer not available');
                this.updateGraphStatus('❌ Graph renderer failed to initialize');
            }
            
            this.updateTable();
        } catch (error) {
            console.error('❌ Error loading data:', error);
            this.updateGraphStatus('❌ Error loading graph data');
            this.showError('Error loading graph data: ' + error.message);
        }
    }

    getDemoData() {
        // Демо данные на случай если файл не доступен
        return `1,2
1,3
1,4
2,3
2,4
3,4
4,5
5,6
6,7
7,8
8,9
9,10`;
    }

    parseCSVToGraph(csvText) {
        const edges = [];
        const nodes = new Set();
        
        const lines = csvText.trim().split('\n');
        console.log('📝 Parsing CSV lines:', lines.length);
        
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

        console.log('👥 Found nodes:', Array.from(nodes));
        console.log('🔗 Found edges:', edges.length);

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
        console.log('🔄 Compute PageRank clicked');
        
        if (this.isComputing) {
            console.log('⏳ Already computing, please wait...');
            return;
        }
        
        if (!this.graph) {
            console.log('❌ No graph data available');
            this.showError('No graph data available. Please load data first.');
            return;
        }
        
        this.isComputing = true;
        const computeBtn = document.getElementById('computeBtn');
        computeBtn.disabled = true;
        computeBtn.innerHTML = '<div class="loading"></div>Computing...';

        try {
            console.log('🧮 Starting PageRank computation...');
            this.updateGraphStatus('Computing PageRank scores...');
            
            this.pageRankScores = await computePageRank(this.graph.adjacencyList, 50, 0.85);
            console.log('✅ PageRank computed:', this.pageRankScores);
            
            this.updateTable();
            
            // Update graph visualization with new scores
            if (window.graphRenderer) {
                window.graphRenderer.updatePageRankScores(this.pageRankScores);
            }
            
            // Refresh node details if a node is selected
            if (this.selectedNode !== null) {
                this.showNodeDetails(this.selectedNode);
            }
            
            this.showSuccess('PageRank computation completed!');
            this.updateGraphStatus('✅ PageRank computed! Click on nodes to see recommendations.');
        } catch (error) {
            console.error('❌ Error computing PageRank:', error);
            this.showError('Error computing PageRank scores: ' + error.message);
            this.updateGraphStatus('❌ Error computing PageRank');
        } finally {
            this.isComputing = false;
            computeBtn.disabled = false;
            computeBtn.textContent = 'Compute PageRank';
        }
    }

    resetGraph() {
        console.log('🔄 Reset Graph clicked');
        
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
        this.showSuccess('Graph reset to initial state');
    }

    updateTable() {
        const tableBody = document.getElementById('tableBody');
        if (!tableBody) {
            console.error('❌ Table body not found');
            return;
        }

        tableBody.innerHTML = '';

        if (!this.graph) {
            console.log('❌ No graph data available');
            return;
        }

        // Create array of nodes with their scores and sort by PageRank (descending)
        const nodesWithScores = this.graph.nodes.map(node => ({
            ...node,
            score: this.pageRankScores ? this.pageRankScores[node.id] || 0 : 0,
            friends: this.graph.adjacencyList[node.id] || []
        }));

        // Sort by PageRank score (descending)
        nodesWithScores.sort((a, b) => b.score - a.score);

        console.log('📋 Updating table with', nodesWithScores.length, 'nodes');

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
            
            row.addEventListener('click', () => {
                console.log('🖱️ Table row clicked:', nodeData.id);
                this.selectNode(nodeData.id);
            });
            
            if (this.selectedNode === nodeData.id) {
                row.classList.add('selected');
            }
            
            tableBody.appendChild(row);
        });
    }

    selectNode(nodeId) {
        console.log('🎯 Selecting node:', nodeId);
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
        if (!nodeDetails) {
            console.error('❌ Node details container not found');
            return;
        }

        const friends = this.graph.adjacencyList[nodeId] || [];
        
        let recommendationsHtml = '';
        if (this.pageRankScores) {
            const recommendations = this.getRecommendations(nodeId);
            console.log('💡 Recommendations for node', nodeId, ':', recommendations);
            
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
        console.log('🔗 Connecting nodes:', sourceId, targetId);
        
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
        console.error('❌ Error:', message);
        this.showNotification(message, 'error');
    }

    showSuccess(message) {
        console.log('✅ Success:', message);
        this.showNotification(message, 'success');
    }

    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        const bgColor = type === 'error' ? 
            'linear-gradient(135deg, #e74c3c, #c0392b)' : 
            'linear-gradient(135deg, #2ecc71, #27ae60)';
            
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: ${bgColor};
            color: white;
            padding: 15px 20px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            z-index: 1000;
            font-weight: bold;
            max-width: 300px;
        `;
        notification.textContent = message;
        document.body.appendChild(notification);
        
        setTimeout(() => {
            if (notification.parentNode) {
                document.body.removeChild(notification);
            }
        }, 4000);
    }
}

// Инициализация приложения когда все готово
function initializeApplication() {
    console.log('🚀 Initializing PageRank Application...');
    
    if (typeof d3 === 'undefined') {
        console.error('❌ D3.js not loaded');
        return;
    }
    
    if (typeof tf === 'undefined') {
        console.error('❌ TensorFlow.js not loaded');
        return;
    }
    
    if (typeof computePageRank === 'undefined') {
        console.error('❌ PageRank module not loaded');
        return;
    }
    
    if (typeof GraphRenderer === 'undefined') {
        console.error('❌ GraphRenderer not loaded');
        return;
    }
    
    console.log('✅ All dependencies loaded, creating app instance...');
    window.app = new PageRankApp();
}

// Ждем когда все будет готово
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeApplication);
} else {
    setTimeout(initializeApplication, 100);
}
