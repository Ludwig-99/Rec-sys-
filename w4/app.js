// app.js
class MovieLensApp {
    constructor() {
        this.interactions = [];
        this.items = new Map();
        this.userMap = new Map();
        this.itemMap = new Map();
        this.reverseUserMap = new Map();
        this.reverseItemMap = new Map();
        this.userTopRated = new Map();
        this.model = null;
        
        this.config = {
            maxInteractions: 80000,
            embeddingDim: 32,
            batchSize: 512,
            epochs: 20,
            learningRate: 0.001
        };
        
        this.lossHistory = [];
        this.isTraining = false;
        
        this.initializeUI();
    }
    
    initializeUI() {
        document.getElementById('loadData').addEventListener('click', () => this.loadData());
        document.getElementById('train').addEventListener('click', () => this.train());
        document.getElementById('test').addEventListener('click', () => this.test());
        
        this.updateStatus('Click "Load Data" to start');
    }
    
    async loadData() {
        this.updateStatus('Loading MovieLens 100K dataset...');
        this.updateProgress(10);
        
        try {
            // Load interactions
            this.updateStatus('Loading user ratings...');
            const interactionsResponse = await fetch('data/u.data');
            const interactionsText = await interactionsResponse.text();
            const interactionsLines = interactionsText.trim().split('\n');
            
            this.interactions = interactionsLines.slice(0, this.config.maxInteractions).map(line => {
                const [userId, itemId, rating, timestamp] = line.split('\t');
                return {
                    userId: parseInt(userId),
                    itemId: parseInt(itemId),
                    rating: parseFloat(rating),
                    timestamp: parseInt(timestamp)
                };
            });
            
            this.updateProgress(40);
            
            // Load items
            this.updateStatus('Loading movie information...');
            const itemsResponse = await fetch('data/u.item');
            const itemsText = await itemsResponse.text();
            const itemsLines = itemsText.trim().split('\n');
            
            itemsLines.forEach(line => {
                const parts = line.split('|');
                const itemId = parseInt(parts[0]);
                const title = parts[1];
                const yearMatch = title.match(/\((\d{4})\)$/);
                const year = yearMatch ? parseInt(yearMatch[1]) : null;
                
                this.items.set(itemId, {
                    title: title.replace(/\(\d{4}\)$/, '').trim(),
                    year: year
                });
            });
            
            this.updateProgress(70);
            
            // Create mappings and find users with sufficient ratings
            this.createMappings();
            this.findQualifiedUsers();
            
            this.updateProgress(100);
            this.updateStatus(`✅ Successfully loaded ${this.interactions.length} interactions and ${this.items.size} movies. ${this.userTopRated.size} users have 20+ ratings.`);
            
            document.getElementById('train').disabled = false;
            
        } catch (error) {
            this.updateStatus(`❌ Error loading data: ${error.message}`);
        }
    }
    
    createMappings() {
        // Create user and item mappings to 0-based indices
        const userSet = new Set(this.interactions.map(i => i.userId));
        const itemSet = new Set(this.interactions.map(i => i.itemId));
        
        Array.from(userSet).forEach((userId, index) => {
            this.userMap.set(userId, index);
            this.reverseUserMap.set(index, userId);
        });
        
        Array.from(itemSet).forEach((itemId, index) => {
            this.itemMap.set(itemId, index);
            this.reverseItemMap.set(index, itemId);
        });
        
        // Group interactions by user and find top rated movies
        const userInteractions = new Map();
        this.interactions.forEach(interaction => {
            const userId = interaction.userId;
            if (!userInteractions.has(userId)) {
                userInteractions.set(userId, []);
            }
            userInteractions.get(userId).push(interaction);
        });
        
        // Sort each user's interactions by rating (desc) and timestamp (desc)
        userInteractions.forEach((interactions, userId) => {
            interactions.sort((a, b) => {
                if (b.rating !== a.rating) return b.rating - a.rating;
                return b.timestamp - a.timestamp;
            });
        });
        
        this.userTopRated = userInteractions;
    }
    
    findQualifiedUsers() {
        // Filter users with at least 20 ratings
        const qualifiedUsers = [];
        this.userTopRated.forEach((interactions, userId) => {
            if (interactions.length >= 20) {
                qualifiedUsers.push(userId);
            }
        });
        this.qualifiedUsers = qualifiedUsers;
    }
    
    async train() {
        if (this.isTraining) return;
        
        this.isTraining = true;
        document.getElementById('train').disabled = true;
        this.lossHistory = [];
        
        this.updateStatus('🚀 Initializing Two-Tower model architecture...');
        
        // Initialize model
        this.model = new TwoTowerModel(
            this.userMap.size,
            this.itemMap.size,
            this.config.embeddingDim
        );
        
        // Prepare training data
        const userIndices = this.interactions.map(i => this.userMap.get(i.userId));
        const itemIndices = this.interactions.map(i => this.itemMap.get(i.itemId));
        
        this.updateStatus('🎯 Starting model training with in-batch negative sampling...');
        
        // Training loop
        const numBatches = Math.ceil(userIndices.length / this.config.batchSize);
        
        for (let epoch = 0; epoch < this.config.epochs; epoch++) {
            let epochLoss = 0;
            
            for (let batch = 0; batch < numBatches; batch++) {
                const start = batch * this.config.batchSize;
                const end = Math.min(start + this.config.batchSize, userIndices.length);
                
                const batchUsers = userIndices.slice(start, end);
                const batchItems = itemIndices.slice(start, end);
                
                const loss = await this.model.trainStep(batchUsers, batchItems);
                epochLoss += loss;
                
                this.lossHistory.push(loss);
                this.updateLossChart();
                
                const progress = ((epoch * numBatches + batch) / (this.config.epochs * numBatches)) * 100;
                this.updateProgress(progress);
                
                if (batch % 10 === 0) {
                    this.updateStatus(`📊 Epoch ${epoch + 1}/${this.config.epochs}, Batch ${batch}/${numBatches}, Loss: ${loss.toFixed(4)}`);
                }
                
                // Allow UI to update
                await new Promise(resolve => setTimeout(resolve, 0));
            }
            
            epochLoss /= numBatches;
            this.updateStatus(`✅ Epoch ${epoch + 1}/${this.config.epochs} completed. Average loss: ${epochLoss.toFixed(4)}`);
        }
        
        this.isTraining = false;
        document.getElementById('train').disabled = false;
        document.getElementById('test').disabled = false;
        
        this.updateStatus('🎉 Training completed! Click "Test" to generate personalized recommendations.');
        this.updateProgress(0);
        
        // Visualize embeddings
        this.visualizeEmbeddings();
    }
    
    updateLossChart() {
        const canvas = document.getElementById('lossChart');
        const ctx = canvas.getContext('2d');
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        if (this.lossHistory.length === 0) return;
        
        // Create gradient background
        const gradient = ctx.createLinearGradient(0, 0, canvas.width, 0);
        gradient.addColorStop(0, '#00a8ff');
        gradient.addColorStop(1, '#00d2a8');
        
        const maxLoss = Math.max(...this.lossHistory);
        const minLoss = Math.min(...this.lossHistory);
        const range = maxLoss - minLoss || 1;
        
        ctx.strokeStyle = gradient;
        ctx.lineWidth = 3;
        ctx.lineJoin = 'round';
        ctx.beginPath();
        
        // Smooth line drawing
        this.lossHistory.forEach((loss, index) => {
            const x = (index / this.lossHistory.length) * canvas.width;
            const y = canvas.height - ((loss - minLoss) / range) * canvas.height * 0.9 - 10;
            
            if (index === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });
        
        ctx.stroke();
        
        // Add gradient fill under line
        ctx.lineTo(canvas.width, canvas.height);
        ctx.lineTo(0, canvas.height);
        ctx.closePath();
        
        const fillGradient = ctx.createLinearGradient(0, 0, 0, canvas.height);
        fillGradient.addColorStop(0, 'rgba(0, 168, 255, 0.2)');
        fillGradient.addColorStop(1, 'rgba(0, 210, 168, 0.1)');
        ctx.fillStyle = fillGradient;
        ctx.fill();
        
        // Add labels with modern styling
        ctx.fillStyle = '#2c3e50';
        ctx.font = '12px "Segoe UI"';
        ctx.fillText(`Min: ${minLoss.toFixed(4)}`, 10, canvas.height - 10);
        ctx.fillText(`Max: ${maxLoss.toFixed(4)}`, 10, 25);
        ctx.fillText(`Current: ${this.lossHistory[this.lossHistory.length - 1].toFixed(4)}`, 10, 45);
    }
    
    async visualizeEmbeddings() {
        if (!this.model) return;
        
        this.updateStatus('🔄 Computing embedding visualization with PCA...');
        
        const canvas = document.getElementById('embeddingChart');
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        try {
            // Sample items for visualization
            const sampleSize = Math.min(300, this.itemMap.size);
            const sampleIndices = Array.from({length: sampleSize}, (_, i) => 
                Math.floor(i * this.itemMap.size / sampleSize)
            );
            
            // Get embeddings and compute PCA
            const embeddingsTensor = this.model.getItemEmbeddings();
            const embeddings = embeddingsTensor.arraySync();
            const sampleEmbeddings = sampleIndices.map(i => embeddings[i]);
            
            const projected = this.computePCA(sampleEmbeddings, 2);
            
            // Normalize to canvas coordinates
            const xs = projected.map(p => p[0]);
            const ys = projected.map(p => p[1]);
            
            const xMin = Math.min(...xs);
            const xMax = Math.max(...xs);
            const yMin = Math.min(...ys);
            const yMax = Math.max(...ys);
            
            const xRange = xMax - xMin || 1;
            const yRange = yMax - yMin || 1;
            
            // Draw points with gradient colors
            sampleIndices.forEach((itemIdx, i) => {
                const x = ((projected[i][0] - xMin) / xRange) * (canvas.width - 60) + 30;
                const y = ((projected[i][1] - yMin) / yRange) * (canvas.height - 60) + 30;
                
                // Create radial gradient for each point
                const gradient = ctx.createRadialGradient(x, y, 0, x, y, 8);
                gradient.addColorStop(0, 'rgba(0, 168, 255, 0.8)');
                gradient.addColorStop(1, 'rgba(0, 210, 168, 0.4)');
                
                ctx.beginPath();
                ctx.arc(x, y, 6, 0, 2 * Math.PI);
                ctx.fillStyle = gradient;
                ctx.fill();
                
                // Add subtle glow effect
                ctx.shadowColor = 'rgba(0, 168, 255, 0.3)';
                ctx.shadowBlur = 10;
                ctx.fill();
                ctx.shadowBlur = 0;
            });
            
            // Add title and labels with modern styling
            ctx.fillStyle = '#2c3e50';
            ctx.font = 'bold 16px "Segoe UI"';
            ctx.fillText('Item Embeddings Projection (PCA)', 20, 25);
            ctx.font = '13px "Segoe UI"';
            ctx.fillStyle = '#7f8c8d';
            ctx.fillText(`Visualizing ${sampleSize} movie embeddings in 2D space`, 20, 45);
            
            this.updateStatus('✅ Embedding visualization completed.');
        } catch (error) {
            this.updateStatus(`❌ Error in visualization: ${error.message}`);
        }
    }
    
    computePCA(embeddings, dimensions) {
        // Simple PCA using power iteration
        const n = embeddings.length;
        const dim = embeddings[0].length;
        
        // Center the data
        const mean = Array(dim).fill(0);
        embeddings.forEach(emb => {
            emb.forEach((val, i) => mean[i] += val);
        });
        mean.forEach((val, i) => mean[i] = val / n);
        
        const centered = embeddings.map(emb => 
            emb.map((val, i) => val - mean[i])
        );
        
        // Compute covariance matrix
        const covariance = Array(dim).fill(0).map(() => Array(dim).fill(0));
        centered.forEach(emb => {
            for (let i = 0; i < dim; i++) {
                for (let j = 0; j < dim; j++) {
                    covariance[i][j] += emb[i] * emb[j];
                }
            }
        });
        covariance.forEach(row => row.forEach((val, j) => row[j] = val / n));
        
        // Power iteration for first two components
        const components = [];
        for (let d = 0; d < dimensions; d++) {
            let vector = Array(dim).fill(1/Math.sqrt(dim));
            
            for (let iter = 0; iter < 10; iter++) {
                let newVector = Array(dim).fill(0);
                
                for (let i = 0; i < dim; i++) {
                    for (let j = 0; j < dim; j++) {
                        newVector[i] += covariance[i][j] * vector[j];
                    }
                }
                
                const norm = Math.sqrt(newVector.reduce((sum, val) => sum + val * val, 0));
                vector = newVector.map(val => val / norm);
            }
            
            components.push(vector);
            
            // Deflate the covariance matrix
            for (let i = 0; i < dim; i++) {
                for (let j = 0; j < dim; j++) {
                    covariance[i][j] -= vector[i] * vector[j];
                }
            }
        }
        
        // Project data
        return embeddings.map(emb => {
            return components.map(comp => 
                emb.reduce((sum, val, i) => sum + val * comp[i], 0)
            );
        });
    }
    
    async test() {
        if (!this.model || this.qualifiedUsers.length === 0) {
            this.updateStatus('❌ Model not trained or no qualified users found.');
            return;
        }
        
        this.updateStatus('🎯 Generating personalized recommendations...');
        
        try {
            // Pick random qualified user
            const randomUser = this.qualifiedUsers[Math.floor(Math.random() * this.qualifiedUsers.length)];
            const userInteractions = this.userTopRated.get(randomUser);
            const userIndex = this.userMap.get(randomUser);
            
            // Get user embedding
            const userEmb = this.model.getUserEmbedding(userIndex);
            
            // Get scores for all items
            const allItemScores = await this.model.getScoresForAllItems(userEmb);
            
            // Filter out items the user has already rated
            const ratedItemIds = new Set(userInteractions.map(i => i.itemId));
            const candidateScores = [];
            
            allItemScores.forEach((score, itemIndex) => {
                const itemId = this.reverseItemMap.get(itemIndex);
                if (!ratedItemIds.has(itemId)) {
                    candidateScores.push({ itemId, score, itemIndex });
                }
            });
            
            // Sort by score descending and take top 10
            candidateScores.sort((a, b) => b.score - a.score);
            const topRecommendations = candidateScores.slice(0, 10);
            
            // Display results
            this.displayResults(randomUser, userInteractions, topRecommendations);
            
        } catch (error) {
            this.updateStatus(`❌ Error generating recommendations: ${error.message}`);
        }
    }
    
    displayResults(userId, userInteractions, recommendations) {
        const resultsDiv = document.getElementById('results');
        
        const topRated = userInteractions.slice(0, 10);
        
        let html = `
            <h2>✨ Personalized Recommendations for User ${userId}</h2>
            <div class="side-by-side">
                <div class="recommendation-section">
                    <h3>🎭 User's Top 10 Rated Movies</h3>
                    <table>
                        <thead>
                            <tr>
                                <th>Rank</th>
                                <th>Movie Title</th>
                                <th>Rating</th>
                                <th>Year</th>
                            </tr>
                        </thead>
                        <tbody>
        `;
        
        topRated.forEach((interaction, index) => {
            const item = this.items.get(interaction.itemId);
            const stars = '★'.repeat(Math.round(interaction.rating)) + '☆'.repeat(5 - Math.round(interaction.rating));
            html += `
                <tr>
                    <td><strong>${index + 1}</strong></td>
                    <td>${item.title}</td>
                    <td style="color: #ff6bcb;">${stars}</td>
                    <td>${item.year || 'N/A'}</td>
                </tr>
            `;
        });
        
        html += `
                        </tbody>
                    </table>
                </div>
                <div class="recommendation-section">
                    <h3>🚀 AI Recommended Movies</h3>
                    <table>
                        <thead>
                            <tr>
                                <th>Rank</th>
                                <th>Movie Title</th>
                                <th>Match Score</th>
                                <th>Year</th>
                            </tr>
                        </thead>
                        <tbody>
        `;
        
        recommendations.forEach((rec, index) => {
            const item = this.items.get(rec.itemId);
            const scorePercent = Math.min(100, Math.round(rec.score * 20));
            html += `
                <tr>
                    <td><strong>${index + 1}</strong></td>
                    <td>${item.title}</td>
                    <td style="color: #00d2a8;">${scorePercent}%</td>
                    <td>${item.year || 'N/A'}</td>
                </tr>
            `;
        });
        
        html += `
                        </tbody>
                    </table>
                </div>
            </div>
        `;
        
        resultsDiv.innerHTML = html;
        this.updateStatus('✅ Recommendations generated successfully!');
    }
    
    updateStatus(message) {
        document.getElementById('status').textContent = message;
    }
    
    updateProgress(percent) {
        document.getElementById('loadingProgress').style.width = percent + '%';
    }
}

// Initialize app when page loads
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new MovieLensApp();
});
