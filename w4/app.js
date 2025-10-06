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
        this.models = {};
        
        this.config = {
            maxInteractions: 80000,
            embeddingDim: 32,
            batchSize: 512,
            epochs: 15,
            learningRate: 0.001
        };
        
        this.lossHistory = { basic: [], dl: [], genre: [] };
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
                        movieGenres.push(index);
                    }
                });
                
                this.items.set(itemId, {
                    title: title.replace(/\(\d{4}\)$/, '').trim(),
                    year: year,
                    genres: movieGenres.map(genreId => this.genres.get(genreId) || `Genre${genreId}`),
                    primaryGenre: movieGenres.length > 0 ? movieGenres[0] : 0
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
                const item = this.items.get(parseInt(itemId));
                return {
                    userId: parseInt(userId),
                    itemId: parseInt(itemId),
                    rating: parseFloat(rating),
                    timestamp: parseInt(timestamp),
                    genreId: item ? item.primaryGenre : 0
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
        this.lossHistory = { basic: [], dl: [], genre: [] };
        
        this.updateStatus('🚀 Initializing three model architectures...');
        
        // Initialize all three models
        this.models.basic = new BasicMFModel(
            this.userMap.size,
            this.itemMap.size,
            this.config.embeddingDim
        );
        
        this.models.dl = new TwoTowerDLModel(
            this.userMap.size,
            this.itemMap.size,
            this.config.embeddingDim
        );
        
        this.models.genre = new GenreEnhancedModel(
            this.userMap.size,
            this.genres.size,
            this.config.embeddingDim
        );
        
        // Prepare training data
        const userIndices = this.interactions.map(i => this.userMap.get(i.userId));
        const itemIndices = this.interactions.map(i => this.itemMap.get(i.itemId));
        const genreIndices = this.interactions.map(i => i.genreId);
        
        this.updateStatus('🎯 Starting parallel training of all models...');
        
        // Training loop for all models
        const numBatches = Math.ceil(userIndices.length / this.config.batchSize);
        
        for (let epoch = 0; epoch < this.config.epochs; epoch++) {
            let epochLossBasic = 0;
            let epochLossDL = 0;
            let epochLossGenre = 0;
            
            for (let batch = 0; batch < numBatches; batch++) {
                const start = batch * this.config.batchSize;
                const end = Math.min(start + this.config.batchSize, userIndices.length);
                
                const batchUsers = userIndices.slice(start, end);
                const batchItems = itemIndices.slice(start, end);
                const batchGenres = genreIndices.slice(start, end);
                
                // Train all models in parallel
                const [lossBasic, lossDL, lossGenre] = await Promise.all([
                    this.models.basic.trainStep(batchUsers, batchItems),
                    this.models.dl.trainStep(batchUsers, batchItems),
                    this.models.genre.trainStep(batchUsers, batchGenres)
                ]);
                
                epochLossBasic += lossBasic;
                epochLossDL += lossDL;
                epochLossGenre += lossGenre;
                
                this.lossHistory.basic.push(lossBasic);
                this.lossHistory.dl.push(lossDL);
                this.lossHistory.genre.push(lossGenre);
                
                this.updateLossChart();
                
                const progress = ((epoch * numBatches + batch) / (this.config.epochs * numBatches)) * 100;
                this.updateProgress(progress);
                
                if (batch % 10 === 0) {
                    this.updateStatus(`📊 Epoch ${epoch + 1}/${this.config.epochs}, Batch ${batch}/${numBatches} - Basic: ${lossBasic.toFixed(4)}, DL: ${lossDL.toFixed(4)}, Genre: ${lossGenre.toFixed(4)}`);
                }
                
                // Allow UI to update
                await new Promise(resolve => setTimeout(resolve, 0));
            }
            
            epochLossBasic /= numBatches;
            epochLossDL /= numBatches;
            epochLossGenre /= numBatches;
            
            this.updateStatus(`✅ Epoch ${epoch + 1}/${this.config.epochs} completed. Avg Loss - Basic: ${epochLossBasic.toFixed(4)}, DL: ${epochLossDL.toFixed(4)}, Genre: ${epochLossGenre.toFixed(4)}`);
        }
        
        this.isTraining = false;
        document.getElementById('train').disabled = false;
        document.getElementById('test').disabled = false;
        
        this.updateStatus('🎉 All models trained! Click "Compare" to see recommendations.');
        this.updateProgress(0);
        
        // Visualize embeddings
        this.visualizeEmbeddings();
    }
    
    updateLossChart() {
        const canvas = document.getElementById('lossChart');
        const ctx = canvas.getContext('2d');
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        if (this.lossHistory.basic.length === 0) return;
        
        const colors = {
            basic: '#00a8ff',
            dl: '#ff6bcb',
            genre: '#00d2a8'
        };
        
        const maxLoss = Math.max(
            ...this.lossHistory.basic,
            ...this.lossHistory.dl,
            ...this.lossHistory.genre
        );
        const minLoss = Math.min(
            ...this.lossHistory.basic,
            ...this.lossHistory.dl,
            ...this.lossHistory.genre
        );
        const range = maxLoss - minLoss || 1;
        
        // Draw loss lines for each model
        Object.entries(this.lossHistory).forEach(([model, losses]) => {
            ctx.strokeStyle = colors[model];
            ctx.lineWidth = 3;
            ctx.lineJoin = 'round';
            ctx.lineCap = 'round';
            ctx.beginPath();
            
            losses.forEach((loss, index) => {
                const x = (index / losses.length) * canvas.width;
                const y = canvas.height - ((loss - minLoss) / range) * canvas.height * 0.9 - 15;
                
                if (index === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            });
            
            ctx.stroke();
        });
        
        // Add legend
        ctx.fillStyle = '#2c3e50';
        ctx.font = 'bold 14px "Segoe UI"';
        ctx.fillText('Matrix Factorization', 15, 25);
        ctx.fillStyle = '#ff6bcb';
        ctx.fillText('Deep Learning', 15, 45);
        ctx.fillStyle = '#00d2a8';
        ctx.fillText('Genre-Enhanced', 15, 65);
        
        ctx.fillStyle = '#2c3e50';
        ctx.fillText(`Final Loss - Basic: ${this.lossHistory.basic[this.lossHistory.basic.length - 1].toFixed(4)}`, 15, canvas.height - 45);
        ctx.fillText(`DL: ${this.lossHistory.dl[this.lossHistory.dl.length - 1].toFixed(4)}`, 15, canvas.height - 25);
        ctx.fillText(`Genre: ${this.lossHistory.genre[this.lossHistory.genre.length - 1].toFixed(4)}`, 15, canvas.height - 5);
    }
    
    async visualizeEmbeddings() {
        if (!this.models.basic) return;
        
        this.updateStatus('🔄 Computing embedding visualizations...');
        
        const canvas = document.getElementById('embeddingChart');
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        try {
            // Sample items for visualization
            const sampleSize = Math.min(200, this.itemMap.size);
            const sampleIndices = Array.from({length: sampleSize}, (_, i) => 
                Math.floor(i * this.itemMap.size / sampleSize)
            );
            
            // Get embeddings from basic model
            const basicEmbeddings = this.models.basic.getItemEmbeddings();
            const basicSample = sampleIndices.map(i => basicEmbeddings[i]);
            const basicProjected = this.computePCA(basicSample, 2);
            
            // Normalize coordinates
            const allCoords = [...basicProjected];
            const xs = allCoords.map(p => p[0]);
            const ys = allCoords.map(p => p[1]);
            
            const xMin = Math.min(...xs);
            const xMax = Math.max(...xs);
            const yMin = Math.min(...ys);
            const yMax = Math.max(...ys);
            
            const xRange = xMax - xMin || 1;
            const yRange = yMax - yMin || 1;
            
            // Draw points from basic model
            sampleIndices.forEach((itemIdx, i) => {
                const x = ((basicProjected[i][0] - xMin) / xRange) * (canvas.width - 100) + 50;
                const y = ((basicProjected[i][1] - yMin) / yRange) * (canvas.height - 100) + 50;
                
                const item = this.items.get(this.reverseItemMap.get(itemIdx));
                const hue = (item.primaryGenre / this.genres.size) * 360;
                
                const gradient = ctx.createRadialGradient(x, y, 0, x, y, 8);
                gradient.addColorStop(0, `hsla(${hue}, 80%, 60%, 0.8)`);
                gradient.addColorStop(1, `hsla(${hue}, 80%, 60%, 0.2)`);
                
                ctx.beginPath();
                ctx.arc(x, y, 6, 0, 2 * Math.PI);
                ctx.fillStyle = gradient;
                ctx.fill();
            });
            
            // Add title and info
            ctx.fillStyle = '#2c3e50';
            ctx.font = 'bold 16px "Segoe UI"';
            ctx.fillText('Item Embeddings Visualization (Matrix Factorization)', 20, 25);
            ctx.font = '14px "Segoe UI"';
            ctx.fillStyle = '#7f8c8d';
            ctx.fillText(`Showing ${sampleSize} items colored by genre`, 20, 50);
            
            this.updateStatus('✅ Embedding visualization completed.');
        } catch (error) {
            this.updateStatus(`❌ Error in visualization: ${error.message}`);
        }
    }
    
    computePCA(embeddings, dimensions) {
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
        if (!this.models.basic || this.qualifiedUsers.length === 0) {
            this.updateStatus('❌ Models not trained or no qualified users found.');
            return;
        }
        
        this.updateStatus('🎯 Generating recommendations from all models...');
        
        try {
            // Pick random qualified user
            const randomUser = this.qualifiedUsers[Math.floor(Math.random() * this.qualifiedUsers.length)];
            const userInteractions = this.userTopRated.get(randomUser);
            const userIndex = this.userMap.get(randomUser);
            
            const ratedItemIds = new Set(userInteractions.map(i => i.itemId));
            
            // Get recommendations from all models
            const [basicRecs, dlRecs, genreRecs] = await Promise.all([
                this.getModelRecommendations(this.models.basic, userIndex, ratedItemIds, 'basic'),
                this.getModelRecommendations(this.models.dl, userIndex, ratedItemIds, 'dl'),
                this.getModelRecommendations(this.models.genre, userIndex, ratedItemIds, 'genre')
            ]);
            
            // Display comparison results
            this.displayComparisonResults(randomUser, userInteractions, basicRecs, dlRecs, genreRecs);
            
        } catch (error) {
            this.updateStatus(`❌ Error generating recommendations: ${error.message}`);
        }
    }
    
    async getModelRecommendations(model, userIndex, ratedItemIds, modelType) {
        let scores;
        
        if (modelType === 'genre') {
            // For genre model, we need to handle genre-based recommendations differently
            const genrePredictions = await model.getPredictionsForUser(userIndex, Array.from(this.genres.keys()));
            
            // Find movies matching top genres
            const candidateMovies = [];
            const moviesByGenre = new Map();
            
            this.items.forEach((item, itemId) => {
                if (!ratedItemIds.has(itemId)) {
                    const genreId = item.primaryGenre;
                    if (!moviesByGenre.has(genreId)) {
                        moviesByGenre.set(genreId, []);
                    }
                    moviesByGenre.get(genreId).push({ itemId, item });
                }
            });
            
            genrePredictions.slice(0, 5).forEach(({ genreId, score }) => {
                const movies = moviesByGenre.get(genreId) || [];
                movies.slice(0, 2).forEach(movie => {
                    candidateMovies.push({
                        itemId: movie.itemId,
                        score: score,
                        item: movie.item
                    });
                });
            });
            
            candidateMovies.sort((a, b) => b.score - a.score);
            return candidateMovies.slice(0, 10);
        } else {
            // For basic and DL models
            const userEmb = model.getUserEmbedding(userIndex);
            const allItemScores = await model.getScoresForAllItems(userEmb);
            
            const candidateScores = [];
            allItemScores.forEach((score, itemIndex) => {
                const itemId = this.reverseItemMap.get(itemIndex);
                if (!ratedItemIds.has(itemId)) {
                    candidateScores.push({ 
                        itemId, 
                        score, 
                        item: this.items.get(itemId)
                    });
                }
            });
            
            candidateScores.sort((a, b) => b.score - a.score);
            return candidateScores.slice(0, 10);
        }
    }
    
    displayComparisonResults(userId, userInteractions, basicRecs, dlRecs, genreRecs) {
        const resultsDiv = document.getElementById('results');
        
        const topRated = userInteractions.slice(0, 10);
        
        let html = `
            <h2>✨ Model Comparison for User ${userId}</h2>
            <div style="text-align: center; margin-bottom: 20px; color: var(--text-light);">
                Comparing recommendations from three different architectures
            </div>
            
            <div class="model-comparison">
                <div class="model-section">
                    <h3>Matrix Factorization <span class="model-badge basic-badge">Basic</span></h3>
                    <table>
                        <thead>
                            <tr>
                                <th>Movie</th>
                                <th>Score</th>
                                <th>Genres</th>
                            </tr>
                        </thead>
                        <tbody>
        `;
        
        basicRecs.forEach((rec, index) => {
            const genreText = rec.item.genres.slice(0, 2).join(', ');
            const scorePercent
