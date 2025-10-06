// two-tower.js
// Matrix Factorization Model (Basic - No Deep Learning)
class BasicMFModel {
    constructor(numUsers, numItems, embDim) {
        this.numUsers = numUsers;
        this.numItems = numItems;
        this.embDim = embDim;
        
        // Simple embedding tables - classic matrix factorization
        this.userEmbeddings = tf.variable(
            tf.randomNormal([numUsers, embDim], 0, 0.01), 
            true, 
            'user_embeddings'
        );
        
        this.itemEmbeddings = tf.variable(
            tf.randomNormal([numItems, embDim], 0, 0.01), 
            true, 
            'item_embeddings'
        );
        
        this.optimizer = tf.train.adam(0.001);
    }
    
    // Simple embedding lookup for users
    userForward(userIndices) {
        return tf.gather(this.userEmbeddings, userIndices);
    }
    
    // Simple embedding lookup for items  
    itemForward(itemIndices) {
        return tf.gather(this.itemEmbeddings, itemIndices);
    }
    
    // Dot product scoring - efficient for retrieval
    score(userEmbeddings, itemEmbeddings) {
        return tf.sum(tf.mul(userEmbeddings, itemEmbeddings), -1);
    }
    
    async trainStep(userIndices, itemIndices) {
        return await tf.tidy(() => {
            // In-batch negative sampling: use all items in batch as negatives
            const loss = () => {
                const userEmbs = this.userForward(userIndices);
                const itemEmbs = this.itemForward(itemIndices);
                
                // Compute similarity matrix: batch_size x batch_size
                const logits = tf.matMul(userEmbs, itemEmbs, false, true);
                
                // Labels: diagonal elements are positive pairs
                const labels = tf.oneHot(
                    tf.range(0, userIndices.length, 1, 'int32'), 
                    userIndices.length
                );
                
                // Softmax cross entropy encourages positive pairs to score higher than negatives
                const loss = tf.losses.softmaxCrossEntropy(labels, logits);
                return loss;
            };
            
            const { value, grads } = this.optimizer.computeGradients(loss);
            this.optimizer.applyGradients(grads);
            
            return value.dataSync()[0];
        });
    }
    
    getUserEmbedding(userIndex) {
        return tf.tidy(() => {
            return this.userForward([userIndex]).squeeze().arraySync();
        });
    }
    
    async getScoresForAllItems(userEmbedding) {
        return await tf.tidy(() => {
            const userEmbTensor = tf.tensor1d(userEmbedding);
            const scores = tf.dot(this.itemEmbeddings, userEmbTensor);
            return scores.dataSync();
        });
    }
    
    getItemEmbeddings() {
        return this.itemEmbeddings.arraySync();
    }
}

// Deep Learning Two-Tower Model with MLP
class TwoTowerDLModel {
    constructor(numUsers, numItems, embDim, hiddenDim = 64) {
        this.numUsers = numUsers;
        this.numItems = numItems;
        this.embDim = embDim;
        this.hiddenDim = hiddenDim;
        
        // User Tower: MLP with one hidden layer
        this.userTower = tf.sequential({
            layers: [
                tf.layers.embedding({
                    inputDim: numUsers,
                    outputDim: embDim,
                    name: 'user_embedding'
                }),
                tf.layers.dense({
                    units: hiddenDim,
                    activation: 'relu',
                    name: 'user_hidden'
                }),
                tf.layers.dense({
                    units: embDim,
                    activation: 'linear',
                    name: 'user_output'
                })
            ]
        });
        
        // Item Tower: MLP with one hidden layer
        this.itemTower = tf.sequential({
            layers: [
                tf.layers.embedding({
                    inputDim: numItems,
                    outputDim: embDim,
                    name: 'item_embedding'
                }),
                tf.layers.dense({
                    units: hiddenDim,
                    activation: 'relu',
                    name: 'item_hidden'
                }),
                tf.layers.dense({
                    units: embDim,
                    activation: 'linear',
                    name: 'item_output'
                })
            ]
        });
        
        this.optimizer = tf.train.adam(0.001);
    }
    
    // User tower forward pass through MLP
    userForward(userIndices) {
        return tf.tidy(() => {
            const userTensor = tf.tensor1d(userIndices, 'int32');
            return this.userTower.apply(userTensor);
        });
    }
    
    // Item tower forward pass through MLP
    itemForward(itemIndices) {
        return tf.tidy(() => {
            const itemTensor = tf.tensor1d(itemIndices, 'int32');
            return this.itemTower.apply(itemTensor);
        });
    }
    
    // Dot product scoring between user and item embeddings
    score(userEmbeddings, itemEmbeddings) {
        return tf.sum(tf.mul(userEmbeddings, itemEmbeddings), -1);
    }
    
    async trainStep(userIndices, itemIndices) {
        return await tf.tidy(() => {
            const loss = () => {
                const userEmbs = this.userForward(userIndices);
                const itemEmbs = this.itemForward(itemIndices);
                
                // In-batch negative sampling with deep representations
                const logits = tf.matMul(userEmbs, itemEmbs, false, true);
                
                const labels = tf.oneHot(
                    tf.range(0, userIndices.length, 1, 'int32'), 
                    userIndices.length
                );
                
                const loss = tf.losses.softmaxCrossEntropy(labels, logits);
                return loss;
            };
            
            const { value, grads } = this.optimizer.computeGradients(loss);
            this.optimizer.applyGradients(grads);
            
            return value.dataSync()[0];
        });
    }
    
    getUserEmbedding(userIndex) {
        return tf.tidy(() => {
            return this.userForward([userIndex]).squeeze().arraySync();
        });
    }
    
    async getScoresForAllItems(userEmbedding) {
        return await tf.tidy(() => {
            // Get all item embeddings through the item tower
            const allItemIndices = Array.from({length: this.numItems}, (_, i) => i);
            const itemTensor = tf.tensor1d(allItemIndices, 'int32');
            const itemEmbs = this.itemTower.apply(itemTensor);
            
            const userEmbTensor = tf.tensor2d([userEmbedding]);
            const scores = tf.matMul(userEmbTensor, itemEmbs, false, true);
            return scores.dataSync();
        });
    }
    
    getItemEmbeddings() {
        return tf.tidy(() => {
            const allItemIndices = Array.from({length: this.numItems}, (_, i) => i);
            const itemTensor = tf.tensor1d(allItemIndices, 'int32');
            return this.itemTower.apply(itemTensor).arraySync();
        });
    }
}

// Genre-Enhanced Model using genre features
class GenreEnhancedModel {
    constructor(numUsers, numGenres, embDim, hiddenDim = 64) {
        this.numUsers = numUsers;
        this.numGenres = numGenres;
        this.embDim = embDim;
        this.hiddenDim = hiddenDim;
        
        // User Tower: MLP architecture
        this.userTower = tf.sequential({
            layers: [
                tf.layers.embedding({
                    inputDim: numUsers,
                    outputDim: embDim,
                    name: 'user_embedding'
                }),
                tf.layers.dense({
                    units: hiddenDim,
                    activation: 'relu',
                    name: 'user_hidden'
                }),
                tf.layers.dense({
                    units: embDim,
                    activation: 'linear',
                    name: 'user_output'
                })
            ]
        });
        
        // Item Tower: Uses genre embeddings instead of item IDs
        this.itemTower = tf.sequential({
            layers: [
                tf.layers.embedding({
                    inputDim: numGenres,
                    outputDim: hiddenDim,
                    name: 'genre_embedding'
                }),
                tf.layers.dense({
                    units: hiddenDim,
                    activation: 'relu',
                    name: 'item_hidden'
                }),
                tf.layers.dense({
                    units: embDim,
                    activation: 'linear',
                    name: 'item_output'
                })
            ]
        });
        
        this.optimizer = tf.train.adam(0.001);
    }
    
    // User tower forward pass
    userForward(userIndices) {
        return tf.tidy(() => {
            const userTensor = tf.tensor1d(userIndices, 'int32');
            return this.userTower.apply(userTensor);
        });
    }
    
    // Item tower forward pass with genre features
    itemForward(genreIndices) {
        return tf.tidy(() => {
            const genreTensor = tf.tensor1d(genreIndices, 'int32');
            return this.itemTower.apply(genreTensor);
        });
    }
    
    // Dot product scoring
    score(userEmbeddings, itemEmbeddings) {
        return tf.sum(tf.mul(userEmbeddings, itemEmbeddings), -1);
    }
    
    async trainStep(userIndices, genreIndices) {
        return await tf.tidy(() => {
            const loss = () => {
                const userEmbs = this.userForward(userIndices);
                const itemEmbs = this.itemForward(genreIndices);
                
                // In-batch negative sampling with genre-based items
                const logits = tf.matMul(userEmbs, itemEmbs, false, true);
                
                const labels = tf.oneHot(
                    tf.range(0, userIndices.length, 1, 'int32'), 
                    userIndices.length
                );
                
                const loss = tf.losses.softmaxCrossEntropy(labels, logits);
                return loss;
            };
            
            const { value, grads } = this.optimizer.computeGradients(loss);
            this.optimizer.applyGradients(grads);
            
            return value.dataSync()[0];
        });
    }
    
    getUserEmbedding(userIndex) {
        return tf.tidy(() => {
            return this.userForward([userIndex]).squeeze().arraySync();
        });
    }
    
    async getPredictionsForUser(userIndex, allGenres) {
        return await tf.tidy(() => {
            const userIndices = Array(allGenres.length).fill(userIndex);
            const userEmbs = this.userForward(userIndices);
            const itemEmbs = this.itemForward(allGenres);
            
            // Compute scores for all genres
            const scores = this.score(userEmbs, itemEmbs).dataSync();
            
            // Combine genres with their scores
            const genreScores = allGenres.map((genreId, index) => ({
                genreId: genreId,
                score: scores[index]
            }));
            
            // Sort by score descending
            genreScores.sort((a, b) => b.score - a.score);
            
            return genreScores;
        });
    }
    
    getGenreEmbeddings() {
        return tf.tidy(() => {
            const allGenreIndices = Array.from({length: this.numGenres}, (_, i) => i);
            const genreTensor = tf.tensor1d(allGenreIndices, 'int32');
            return this.itemTower.apply(genreTensor).arraySync();
        });
    }
}
