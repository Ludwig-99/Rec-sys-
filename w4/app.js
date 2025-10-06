class MovieRecommender {
    constructor() {
        this.interactions = [];
        this.items = new Map();
        this.userMap = new Map();
        this.itemMap = new Map();
        this.reverseUserMap = new Map();
        this.reverseItemMap = new Map();
        this.userRatings = new Map();
        this.model = null;
        
        this.setupEventListeners();
    }

    setupEventListeners() {
        document.getElementById('loadData').addEventListener('click', () => this.loadData());
        document.getElementById('train').addEventListener('click', () => this.train());
        document.getElementById('test').addEventListener('click', () => this.test());
    }

    async loadData() {
        this.updateStatus('Loading data...');
        
        try {
            // Load interactions data
            const response1 = await fetch('data/u.data');
            const data1 = await response1.text();
            this.parseInteractions(data1);
            
            // Load items data
            const response2 = await fetch('data/u.item');
            const data2 = await response2.text();
            this.parseItems(data2);
            
            this.prepareMappings();
            this.prepareUserRatings();
            
            document.getElementById('train').disabled = false;
            this.updateStatus(`Data loaded: ${this.interactions.length} interactions, ${this.items.size} movies, ${this.userMap.size} users`);
        } catch (error) {
            this.updateStatus('Error loading data: ' + error.message);
        }
    }

    parseInteractions(data) {
        const lines = data.trim().split('\n');
        this.interactions = lines.map(line => {
            const [userId, itemId, rating, timestamp] = line.split('\t');
            return {
                userId: parseInt(userId),
                itemId: parseInt(itemId),
                rating: parseFloat(rating),
                timestamp: parseInt(timestamp)
            };
        });
    }

    parseItems(data) {
        const lines = data.trim().split('\n');
        lines.forEach(line => {
            const parts = line.split('|');
            const itemId = parseInt(parts[0]);
            const title = parts[1];
            const yearMatch = title.match(/\((\d{4})\)$/);
            const year = yearMatch ? parseInt(yearMatch[1]) : null;
            
            this.items.set(itemId, {
                title: title,
                year: year
            });
        });
    }

    prepareMappings() {
        // Create user mappings
        const uniqueUsers = [...new Set(this.interactions.map(i => i.userId))];
        uniqueUsers.forEach((userId, index) => {
            this.userMap.set(userId, index);
            this.reverseUserMap.set(index, userId);
        });

        // Create item mappings
        const uniqueItems = [...new Set(this.interactions.map(i => i.itemId))];
        uniqueItems.forEach((itemId, index) => {
            this.itemMap.set(itemId, index);
            this.reverseItemMap.set(index, itemId);
        });
    }

    prepareUserRatings() {
        this.interactions.forEach(interaction => {
            const userId = interaction.userId;
            if (!this.userRatings.has(userId)) {
                this.userRatings.set(userId, []);
            }
            this.userRatings.get(userId).push({
                itemId: interaction.itemId,
                rating: interaction.rating,
                timestamp: interaction.timestamp
            });
        });

        // Sort each user's ratings by rating (desc) and timestamp (desc)
        this.userRatings.forEach(ratings => {
            ratings.sort((a, b) => {
                if (b.rating !== a.rating) return b.rating - a.rating;
                return b.timestamp - a.timestamp;
            });
        });
    }

    async train() {
        this.updateStatus('Training model...');
        document.getElementById('train').disabled = true;
        
        const embeddingDim = 32;
        const batchSize = 512;
        const epochs = 20;
        const learningRate = 0.001;
        
        this.model = new TwoTowerModel(
            this.userMap.size,
            this.itemMap.size,
            embeddingDim
        );

        const lossCtx = document.getElementById('lossChart').getContext('2d');
        this.setupLossChart(lossCtx);

        for (let epoch = 0; epoch < epochs; epoch++) {
            let totalLoss = 0;
            let batchCount = 0;
            
            // Create batches
            const batches = this.createBatches(batchSize);
            
            for (const batch of batches) {
                const loss = await this.model.trainStep(batch.userIndices, batch.itemIndices);
                totalLoss += loss;
                batchCount++;
                
                // Update loss chart every few batches
                if (batchCount % 10 === 0) {
                    this.updateLossChart(lossCtx, epoch + batchCount / batches.length, loss);
                }
                
                // Prevent UI freeze
                await tf.nextFrame();
            }
            
            const avgLoss = totalLoss / batchCount;
            this.updateStatus(`Epoch ${epoch + 1}/${epochs}, Loss: ${avgLoss.toFixed(4)}`);
        }

        document.getElementById('test').disabled = false;
        this.updateStatus('Training completed');
        this.visualizeEmbeddings();
    }

    createBatches(batchSize) {
        const batches = [];
        const shuffled = [...this.interactions].sort(() => Math.random() - 0.5);
        
        for (let i = 0; i < shuffled.length; i += batchSize) {
            const batchInteractions = shuffled.slice(i, i + batchSize);
            const userIndices = batchInteractions.map(i => this.userMap.get(i.userId));
            const itemIndices = batchInteractions.map(i => this.itemMap.get(i.itemId));
            
            batches.push({
                userIndices: tf.tensor1d(userIndices, 'int32'),
                itemIndices: tf.tensor1d(itemIndices, 'int32')
            });
        }
        
        return batches;
    }

    setupLossChart(ctx) {
        ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
        ctx.strokeStyle = '#007acc';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(0, ctx.canvas.height);
    }

    updateLossChart(ctx, x, loss) {
        const xScale = ctx.canvas.width / 25;
        const yScale = ctx.canvas.height / 4;
        
        const xPos = x * xScale;
        const yPos = ctx.canvas.height - (loss * yScale);
        
        ctx.lineTo(xPos, yPos);
        ctx.stroke();
    }

    async visualizeEmbeddings() {
        const ctx = document.getElementById('embeddingChart').getContext('2d');
        ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
        
        // Sample 200 items for visualization
        const sampleSize = Math.min(200, this.itemMap.size);
        const sampleIndices = Array.from({length: sampleSize}, (_, i) => i);
        
        const itemEmbeddings = this.model.getItemEmbeddings();
        const embeddings = [];
        
        for (let i = 0; i < sampleSize; i++) {
            const embedding = itemEmbeddings.slice(i, i + 1);
            embeddings.push(Array.from(embedding.dataSync()));
        }
        
        // Simple 2D projection using PCA approximation
        const projected = this.simpleProjection(embeddings);
        
        // Draw points
        ctx.fillStyle = '#007acc';
        projected.forEach((point, i) => {
            const x = (point[0] + 1) * ctx.canvas.width / 2;
            const y = (point[1] + 1) * ctx.canvas.height / 2;
            
            ctx.beginPath();
            ctx.arc(x, y, 3, 0, 2 * Math.PI);
            ctx.fill();
        });
    }

    simpleProjection(embeddings) {
        // Simple centering and scaling for visualization
        const centered = embeddings.map(emb => {
            const mean = emb.reduce((a, b) => a + b) / emb.length;
            return emb.map(x => x - mean);
        });
        
        // Random projection for simplicity
        const randomVector1 = Array.from({length: embeddings[0].length}, () => Math.random() - 0.5);
        const randomVector2 = Array.from({length: embeddings[0].length}, () => Math.random() - 0.5);
        
        return centered.map(emb => [
            this.dot(emb, randomVector1),
            this.dot(emb, randomVector2)
        ]);
    }

    dot(a, b) {
        return a.reduce((sum, val, i) => sum + val * b[i], 0);
    }

    async test() {
        this.updateStatus('Testing recommendations...');
        
        // Find users with at least 20 ratings
        const qualifiedUsers = Array.from(this.userRatings.entries())
            .filter(([_, ratings]) => ratings.length >= 20)
            .map(([userId]) => userId);
        
        if (qualifiedUsers.length === 0) {
            this.updateStatus('No qualified users found (need users with ≥20 ratings)');
            return;
        }
        
        // Pick random qualified user
        const randomUser = qualifiedUsers[Math.floor(Math.random() * qualifiedUsers.length)];
        const userIndex = this.userMap.get(randomUser);
        
        // Get user embedding
        const userEmbedding = this.model.getUserEmbedding(userIndex);
        
        // Get scores for all items
        const allScores = await this.model.getScoresForAllItems(userEmbedding);
        const scoresArray = Array.from(allScores.dataSync());
        
        // Get user's rated items
        const ratedItems = new Set(this.userRatings.get(randomUser).map(r => r.itemId));
        
        // Filter out rated items and get top recommendations
        const recommendations = scoresArray
            .map((score, index) => ({
                itemId: this.reverseItemMap.get(index),
                score: score
            }))
            .filter(item => !ratedItems.has(item.itemId))
            .sort((a, b) => b.score - a.score)
            .slice(0, 10);
        
        // Get user's top rated movies
        const topRated = this.userRatings.get(randomUser).slice(0, 10);
        
        this.displayResults(topRated, recommendations, randomUser);
        this.updateStatus(`Recommendations generated for user ${randomUser}`);
    }

    displayResults(topRated, recommendations, userId) {
        const resultsDiv = document.getElementById('results');
        
        let html = `<h2>Recommendations for User ${userId}</h2>`;
        html += `<table>`;
        html += `<tr><th>Top 10 Rated Movies</th><th>Top 10 Recommended Movies</th></tr>`;
        html += `<tr><td valign="top"><ol>`;
        
        topRated.forEach(rating => {
            const item = this.items.get(rating.itemId);
            html += `<li>${item.title} (Rating: ${rating.rating})</li>`;
        });
        
        html += `</ol></td><td valign="top"><ol>`;
        
        recommendations.forEach(rec => {
            const item = this.items.get(rec.itemId);
            html += `<li>${item.title} (Score: ${rec.score.toFixed(4)})</li>`;
        });
        
        html += `</ol></td></tr></table>`;
        
        resultsDiv.innerHTML = html;
    }

    updateStatus(message) {
        document.getElementById('status').textContent = 'Status: ' + message;
    }
}

// Initialize the application when the page loads
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new MovieRecommender();
});
