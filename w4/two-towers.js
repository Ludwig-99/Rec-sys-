// two-tower.js
class TwoTowerModel {
    constructor(numUsers, numItems, numGenres, embDim, hiddenDim = 64) {
        this.numUsers = numUsers;
        this.numItems = numItems;
        this.numGenres = numGenres;
        this.embDim = embDim;
        this.hiddenDim = hiddenDim;
        
        // Build user tower
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
        
        // Build item tower (genre-based)
        this.itemTower = tf.sequential({
            layers: [
                tf.layers.embedding({
                    inputDim: numGenres,
                    outputDim: hiddenDim,
                    name: 'genre_embedding'
                }),
                tf.layers.dense({
                    units: embDim,
                    activation: 'linear',
                    name: 'item_output'
                })
            ]
        });
        
        // Adam optimizer
        this.optimizer = tf.train.adam(0.001);
        
        // Store genre embeddings for inference
        this.genreEmbeddings = this.itemTower.layers[0];
    }
    
    // User tower forward pass
    userForward(userIndices) {
        return tf.tidy(() => {
            const userTensor = tf.tensor1d(userIndices, 'int32');
            return this.userTower.apply(userTensor);
        });
    }
    
    // Item tower forward pass (genre-based)
    itemForward(genreIndices) {
        return tf.tidy(() => {
            const genreTensor = tf.tensor1d(genreIndices, 'int32');
            return this.itemTower.apply(genreTensor);
        });
    }
    
    // Scoring function: dot product between user and item embeddings
    score(userEmbeddings, itemEmbeddings) {
        return tf.sum(tf.mul(userEmbeddings, itemEmbeddings), -1);
    }
    
    async trainStep(userIndices, genreIndices) {
        return await tf.tidy(() => {
            // In-batch sampled softmax loss
            const loss = () => {
                const userEmbs = this.userForward(userIndices);
                const itemEmbs = this.itemForward(genreIndices);
                
                // Compute similarity matrix: batch_size x batch_size
                const logits = tf.matMul(userEmbs, itemEmbs, false, true);
                
                // Labels: diagonal elements are positives
                const labels = tf.oneHot(
                    tf.range(0, userIndices.length, 1, 'int32'), 
                    userIndices.length
                );
                
                // Softmax cross entropy loss
                const loss = tf.losses.softmaxCrossEntropy(labels, logits);
                return loss;
            };
            
            // Compute gradients and update both towers
            const { value, grads } = this.optimizer.computeGradients(loss);
            this.optimizer.applyGradients(grads);
            
            return value.dataSync()[0];
        });
    }
    
    getUserEmbedding(userIndex) {
        return tf.tidy(() => {
            return this.userForward([userIndex]).squeeze();
        });
    }
    
    async getItemEmbeddings() {
        return await tf.tidy(() => {
            // Generate embeddings for all genres
            const allGenreIndices = Array.from({length: this.numGenres}, (_, i) => i);
            const genreTensor = tf.tensor1d(allGenreIndices, 'int32');
            const embeddings = this.itemTower.apply(genreTensor);
            return embeddings.arraySync();
        });
    }
    
    async getScoresForAllItems(userEmbedding, itemEmbeddings) {
        return await tf.tidy(() => {
            const userEmbTensor = tf.tensor2d([userEmbedding]);
            const itemEmbTensor = tf.tensor2d(itemEmbeddings);
            
            // Compute dot products with all item embeddings
            const scores = tf.matMul(userEmbTensor, itemEmbTensor, false, true);
            return scores.dataSync();
        });
    }
}
