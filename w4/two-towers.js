// two-tower.js
/**
 * Deep Learning Two-Tower Model with MLP Architecture
 * User Tower: MLP with embedding + hidden layers for user representation
 * Item Tower: MLP with item embedding + genre features for item representation
 * Uses dot product scoring for efficient retrieval
 */
class TwoTowerModel {
    constructor(numUsers, numItems, numGenres, embDim) {
        this.numUsers = numUsers;
        this.numItems = numItems;
        this.numGenres = numGenres;
        this.embDim = embDim;
        
        // User Tower: MLP architecture for learning user representations
        // Embedding layer captures user IDs, MLP layers learn non-linear patterns
        this.userTower = tf.sequential({
            layers: [
                // User embedding layer: maps user IDs to dense vectors
                tf.layers.embedding({
                    inputDim: numUsers,
                    outputDim: embDim,
                    name: 'user_embedding'
                }),
                // First hidden layer: learns non-linear user feature interactions
                tf.layers.dense({
                    units: 64,
                    activation: 'relu',
                    name: 'user_hidden1'
                }),
                // Output layer: produces final user representation in shared space
                tf.layers.dense({
                    units: embDim,
                    activation: 'linear',
                    name: 'user_output'
                })
            ]
        });
        
        // Item Tower: MLP architecture with genre features for item representation
        // Combines item ID embeddings with genre information for richer representations
        this.itemTower = tf.sequential({
            layers: [
                // Item embedding layer: maps item IDs to initial dense vectors
                tf.layers.embedding({
                    inputDim: numItems,
                    outputDim: embDim,
                    name: 'item_embedding'
                }),
                // Genre embedding layer: adds genre features as additional context
                // Genre embeddings are concatenated with item embeddings
                tf.layers.dense({
                    units: embDim + 16, // Extra capacity for genre information
                    activation: 'relu',
                    name: 'item_hidden1'
                }),
                // Output layer: produces final item representation in shared space
                tf.layers.dense({
                    units: embDim,
                    activation: 'linear',
                    name: 'item_output'
                })
            ]
        });
        
        // Genre embedding layer for item features
        this.genreEmbedding = tf.layers.embedding({
            inputDim: numGenres,
            outputDim: 16, // Compact genre representations
            name: 'genre_embedding'
        });
        
        // Adam optimizer for stable training with MLP layers
        this.optimizer = tf.train.adam(0.001);
    }
    
    /**
     * User Tower Forward Pass
     * Processes user indices through MLP to get user embeddings
     * MLP allows learning complex user preference patterns
     */
    userForward(userIndices) {
        return tf.tidy(() => {
            const userTensor = tf.tensor1d(userIndices, 'int32');
            return this.userTower.apply(userTensor);
        });
    }
    
    /**
     * Item Tower Forward Pass with Genre Features
     * Processes item indices and genre information through MLP
     * Genre features provide additional semantic context for items
     */
    itemForward(itemIndices, genreIndices) {
        return tf.tidy(() => {
            const itemTensor = tf.tensor1d(itemIndices, 'int32');
            const genreTensor = tf.tensor1d(genreIndices, 'int32');
            
            // Get base item embeddings
            const itemEmbs = this.itemTower.layers[0].apply(itemTensor);
            
            // Get genre embeddings and concatenate with item embeddings
            const genreEmbs = this.genreEmbedding.apply(genreTensor);
            const combined = tf.concat([itemEmbs, genreEmbs], -1);
            
            // Pass through remaining MLP layers
            let output = combined;
            for (let i = 1; i < this.itemTower.layers.length; i++) {
                output = this.itemTower.layers[i].apply(output);
            }
            
            return output;
        });
    }
    
    /**
     * Scoring Function: Dot Product Similarity
     * Efficient computation of user-item affinity in shared latent space
     * Dot product is computationally efficient and well-suited for retrieval
     */
    score(userEmbeddings, itemEmbeddings) {
        return tf.sum(tf.mul(userEmbeddings, itemEmbeddings), -1);
    }
    
    /**
     * Training Step with In-Batch Negative Sampling
     * Uses all items in batch as negatives for each user
     * Diagonal elements are positive pairs, off-diagonals are negatives
     * This is efficient and provides good gradient signal
     */
    async trainStep(userIndices, itemIndices, genreIndices) {
        return await tf.tidy(() => {
            const loss = () => {
                const userEmbs = this.userForward(userIndices);
                const itemEmbs = this.itemForward(itemIndices, genreIndices);
                
                // Compute similarity matrix: each user scored against all items in batch
                // This creates in-batch negatives automatically
                const logits = tf.matMul(userEmbs, itemEmbs, false, true);
                
                // Labels: diagonal elements are positive user-item pairs
                const labels = tf.oneHot(
                    tf.range(0, userIndices.length, 1, 'int32'), 
                    userIndices.length
                );
                
                // Softmax cross entropy loss encourages positive pairs to have higher scores
                // than negative pairs, learning meaningful user-item affinities
                const loss = tf.losses.softmaxCrossEntropy(labels, logits);
                return loss;
            };
            
            // Compute gradients and update all MLP layers and embeddings
            const { value, grads } = this.optimizer.computeGradients(loss);
            this.optimizer.applyGradients(grads);
            
            return value.dataSync()[0];
        });
    }
    
    /**
     * Get User Embedding for Inference
     * Forward pass through user MLP tower for a specific user
     */
    getUserEmbedding(userIndex) {
        return tf.tidy(() => {
            return this.userForward([userIndex]).squeeze().arraySync();
        });
    }
    
    /**
     * Score All Items for a User
     * Efficient batched computation of user-item affinities
     * Uses matrix multiplication for scalability
     */
    async getScoresForAllItems(userEmbedding) {
        return await tf.tidy(() => {
            // Generate embeddings for all items (with default genre 0)
            const allItemIndices = Array.from({length: this.numItems}, (_, i) => i);
            const allGenreIndices = Array(this.numItems).fill(0);
            const itemEmbs = this.itemForward(allItemIndices, allGenreIndices);
            
            // Compute dot products with user embedding
            const userEmbTensor = tf.tensor2d([userEmbedding]);
            const scores = tf.matMul(userEmbTensor, itemEmbs, false, true);
            return scores.dataSync();
        });
    }
    
    /**
     * Get Item Embeddings for Visualization
     * Returns all item embeddings through the item tower
     */
    getItemEmbeddings() {
        return tf.tidy(() => {
            const allItemIndices = Array.from({length: this.numItems}, (_, i) => i);
            const allGenreIndices = Array(this.numItems).fill(0);
            return this.itemForward(allItemIndices, allGenreIndices).arraySync();
        });
    }
}
