// app.js
class MovieLensApp {
    constructor() {
        this.interactions = [];
        this.items = new Map();
        this.genres = new Map();
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
        
        this.updateStatus('🎬 Click "Load Data" to start exploring MovieLens 100K');
    }
    
    async loadData() {
        this.updateStatus('📥 Loading MovieLens 100K dataset...');
        this.updateProgress(10);
        
        try {
            // Load genres first
            this.updateStatus('📚 Loading genre information...');
            const genresResponse = await fetch('data/u.genre');
            const genresText = await genresResponse.text();
            const genresLines = genresText.trim().split('\n').filter(line => line);
            
            genresLines.forEach(line => {
                const [genreName, genreId] = line.split('|');
                if (genreName && genreId) {
                    this.genres.set(parseInt(genreId), genreName.trim());
                }
            });
            
            this.updateProgress(20);
            
            // Load items with genre information
            this.updateStatus('🎭 Loading movie information and genres...');
            const itemsResponse = await fetch('data/u.item');
            const itemsText = await itemsResponse.text();
            const itemsLines = itemsText.trim().split('\n');
            
            const itemGenres = new Map();
            
            itemsLines.forEach(line => {
                const parts = line.split('|');
                const itemId = parseInt(parts[0]);
                const title = parts[1];
                const yearMatch = title.match(/\((\d{4})\)$/);
                const year = yearMatch ? parseInt(yearMatch[1]) : null;
                
                // Extract genre information (last 19 fields)
                const genreFields = parts.slice(5, 24);
                const movieGenres = [];
                genreFields.forEach((isGenre, index) => {
                    if (isGenre === '1') {
                        movieGenres.push(index); // Genre IDs are 0-based in the dataset
                    }
                });
                
                this.items.set(itemId, {
                    title: title.replace(/\(\d{4}\)$/, '').trim(),
                    year: year,
                    genres: movieGenres.map(genreId => this.genres.get(genreId) || `Genre${genreId}`)
                });
                
                itemGenres.set(itemId, movieGenres.length > 0 ? movieGenres[0] : 0);
            });
            
            this.updateProgress(50);
            
            // Load interactions
            this.updateStatus('⭐ Loading user ratings...');
            const interactionsResponse = await fetch('data/u.data');
            const interactionsText = await interactionsResponse.text();
            const interactionsLines = interactionsText.trim().split('\n');
            
            this.interactions = interactionsLines.slice(0, this.config.maxInteractions).map(line => {
                const [userId, itemId, rating, timestamp] = line.split('\t');
                return {
                    userId: parseInt(userId),
                    itemId: parseInt(itemId),
                    rating: parseFloat(rating),
                    timestamp: parseInt(timestamp),
                    genreId: itemGenres.get(parseInt(itemId)) || 0
                };
            });
            
            this.updateProgress(80);
            
            // Create mappings and find users with sufficient ratings
            this.createMappings();
            this.findQualifiedUsers();
            
            this.updateProgress(100);
            this.updateStatus(`✅ Successfully loaded ${this.interactions.length} interactions, ${this.items.size} movies, and ${this.genres.size} genres. ${this.userTopRated.size} users have 20+ ratings.`);
            
            document.getElementById('train').disabled = false;
            
        } catch (error) {
            this.updateStatus(`❌ Error loading data: ${error.message}`);
            console.error('Data loading error:', error);
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
        
        this.updateStatus('🚀 Initializing Two-Tower model with genre embeddings...');
        
        // Initialize model with genre information
        this.model = new TwoTowerModel(
            this.userMap.size,
            this.itemMap.size,
            this.genres.size,
            this.config.embeddingDim
        );
        
        // Prepare training data
        const userIndices = this.interactions.map(i => this.userMap.get(i.userId));
        const genreIndices = this.interactions.map(i => i.genreId);
        
        this.updateStatus('🎯 Starting model training with in-batch negative sampling...');
        
        // Training loop
        const numBatches = Math.ceil(userIndices.length / this.config.batchSize);
        
        for (let epoch = 0; epoch < this.config.epochs; epoch++) {
            let epochLoss = 0;
            
            for (let batch = 0; batch < numBatches; batch++) {
                const start = batch * this.config.batchSize;
                const end = Math.min(start + this.config.batchSize, userIndices.length);
                
                const batchUsers = userIndices.slice(start, end);
                const batchGenres = genreIndices.slice(start, end);
                
                const loss = await this.model.trainStep(batchUsers, batchGenres);
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
        gradient.addColorStop(0.5, '#ff6bcb');
        gradient.addColorStop(1, '#00d2a8');
        
        const maxLoss = Math.max(...this.lossHistory);
        const minLoss = Math.min(...this.lossHistory);
        const range = maxLoss - minLoss || 1;
        
        ctx.strokeStyle = gradient;
        ctx.lineWidth = 4;
        ctx.lineJoin = 'round';
        ctx.lineCap = 'round';
        ctx.beginPath();
        
        // Smooth line drawing
        this.lossHistory.forEach((loss, index) => {
            const x = (index / this.lossHistory.length) * canvas.width;
            const y = canvas.height - ((loss - minLoss) / range) * canvas.height * 0.9 - 15;
            
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
        fillGradient.addColorStop(0, 'rgba(0, 168, 255, 0.3)');
        fillGradient.addColorStop(1, 'rgba(0, 210, 168, 0.1)');
        ctx.fillStyle = fillGradient;
        ctx.fill();
        
        // Add labels with modern styling
        ctx.fillStyle = '#2c3e50';
        ctx.font = 'bold 14px "Segoe UI"';
        ctx.fillText(`Min Loss: ${minLoss.toFixed(4)}`, 15, canvas.height - 15);
        ctx.fillText(`Max Loss: ${maxLoss.toFixed(4)}`, 15, 30);
        ctx.fillStyle = '#ff6bcb';
        ctx.fillText(`Current: ${this.lossHistory[this.lossHistory.length - 1].toFixed(4)}`, 15, 50);
    }
    
    async visualizeEmbeddings() {
        if (!this.model) return;
        
        this.updateStatus('🔄 Computing embedding visualization with PCA...');
        
        const canvas = document.getElementById('embeddingChart');
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        try {
            // Get all item embeddings (genre-based)
            const itemEmbeddings = await this.model.getItemEmbeddings();
            
            // Sample items for visualization
            const sampleSize = Math.min(300, this.itemMap.size);
            const sampleIndices = Array.from({length: sampleSize}, (_, i) => 
                Math.floor(i * this.itemMap.size / sampleSize)
            );
            
            const sampleEmbeddings = sampleIndices.map(i => itemEmbeddings[i]);
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
            
            // Draw points with gradient colors based on position
            sampleIndices.forEach((itemIdx, i) => {
                const x = ((projected[i][0] - xMin) / xRange) * (canvas.width - 80) + 40;
                const y = ((projected[i][1] - yMin) / yRange) * (canvas.height - 80) + 40;
                
                // Color based on position in the embedding space
                const hue = (x / canvas.width) * 360;
                const saturation = 70 + (y / canvas.height) * 30;
                const lightness = 50;
                
                const gradient = ctx.createRadialGradient(x, y, 0, x, y, 10);
                gradient.addColorStop(0, `hsla(${hue}, ${saturation}%, ${lightness}%, 0.9)`);
                gradient.addColorStop(1, `hsla(${hue}, ${saturation}%, ${lightness + 20}%, 0.4)`);
                
                ctx.beginPath();
                ctx.arc(x, y, 8, 0, 2 * Math.PI);
                ctx.fillStyle = gradient;
                ctx.fill();
                
                // Add subtle glow
                ctx.shadowColor = `hsla(${hue}, ${saturation}%, ${lightness}%, 0.5)`;
                ctx.shadowBlur = 15;
                ctx.fill();
                ctx.shadowBlur = 0;
            });
            
            // Add title and labels
            ctx.fillStyle = '#2c3e50';
            ctx.font = 'bold 18px "Segoe UI"';
            ctx.fillText('Genre Embeddings Projection (PCA)', 20, 30);
            ctx.font = '14px "Segoe UI"';
            ctx.fillStyle = '#7f8c8d';
            ctx.fillText(`Visualizing ${sampleSize} genre embeddings in 2D space`, 20, 55);
            
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
            
            // Get all item embeddings and compute scores
            const itemEmbeddings = await this.model.getItemEmbeddings();
            const allItemScores = await this.model.getScoresForAllItems(userEmb, itemEmbeddings);
            
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
                                <th>Genres</th>
                            </tr>
                        </thead>
                        <tbody>
        `;
        
        topRated.forEach((interaction, index) => {
            const item = this.items.get(interaction.itemId);
            const stars = '★'.repeat(Math.round(interaction.rating)) + '☆'.repeat(5 - Math.round(interaction.rating));
            const genreText = item.genres.slice(0, 2).join(', ');
            html += `
                <tr>
                    <td><strong>#${index + 1}</strong></td>
                    <td>${item.title}</td>
                    <td><span class="stars">${stars}</span></td>
                    <td style="font-size: 0.9em; color: #7f8c8d;">${genreText}</td>
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
                                <th>Genres</th>
                            </tr>
                        </thead>
                        <tbody>
        `;
        
        recommendations.forEach((rec, index) => {
            const item = this.items.get(rec.itemId);
            const scorePercent = Math.min(100, Math.round(rec.score * 25));
            const genreText = item.genres.slice(0, 2).join(', ');
            html += `
                <tr>
                    <td><strong>#${index + 1}</strong></td>
                    <td>${item.title}</td>
                    <td><span class="score">${scorePercent}%</span></td>
                    <td style="font-size: 0.9em; color: #7f8c8d;">${genreText}</td>
                </tr>
            `;
        });
        
        html += `
                        </tbody>
                    </table>
                </div>
            </div>
            <div style="text-align: center; margin-top: 25px; color: #7f8c8d; font-style: italic;">
                Recommendations based on genre preferences and user behavior patterns
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
