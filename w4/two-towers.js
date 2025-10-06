// two-tower.js
/**
 * Enhanced Two-Tower Model for Movie Recommendation with Genre Features
 * 
 * Two-tower architecture learns separate embeddings for users and items.
 * The item tower is enhanced with genre information to better capture
 * movie characteristics and improve recommendation quality.
 */
class TwoTowerModel {
    constructor(numUsers, numItems, numGenres, embeddingDim, hiddenDim = 64) {
        this.numUsers = numUsers;
        this.numItems = numItems;
        this.numGenres = numGenres;
        this.embeddingDim = embeddingDim;
        this.hiddenDim = hiddenDim;
        
        // User Tower: maps user IDs to embeddings through neural network
        // Learns user preferences from their interaction history
        this.userTower = tf.sequential({
            layers: [
                // Embedding layer converts sparse user IDs to dense vectors
                tf.layers.embedding({
                    inputDim: numUsers,
                    outputDim: embeddingDim,
                    inputLength: 1,
                    name: 'user_embedding'
                }),
                tf.layers.flatten(),
                // First hidden layer captures complex user preference patterns
                tf.layers.dense({
                    units: hiddenDim,
                    activation: 'relu',
                    name: 'user_hidden1'
                }),
                // Second hidden layer for deeper pattern learning
                tf.layers.dense({
                    units: hiddenDim / 2,
                    activation: 'relu', 
                    name: 'user_hidden2'
                }),
                // Output embedding for similarity computation
                tf.layers.dense({
                    units: embeddingDim,
                    activation: 'linear',
                    name: 'user_output'
                })
            ]
        });
        
        // Item Tower: enhanced with genre features for better movie representation
        // Combines item ID embeddings with genre information
        this.itemTower = tf.sequential({
            layers: [
                // Custom layer to handle multiple inputs (item ID + genre)
                {
                    className: 'ItemTowerInput',
                    call: (inputs) => {
                        const [itemInput, genreInput] = inputs;
                        
                        // Item embedding branch
                        const itemEmbedding = tf.layers.embedding({
                            inputDim: numItems,
                            outputDim: embeddingDim,
                            inputLength: 1
                        }).apply(itemInput);
                        
                        const flatItem = tf.layers.flatten().apply(itemEmbedding);
                        
                        // Genre embedding branch  
                        const genreEmbedding = tf.layers.embedding({
                            inputDim: numGenres,
                            outputDim: Math.floor(embeddingDim / 4),
                            inputLength: 1
                        }).apply(genreInput);
                        
                        const flatGenre = tf.layers.flatten().apply(genreEmbedding);
                        
                        // Concatenate item and genre embeddings
                        return tf.layers.concatenate().apply([flatItem, flatGenre]);
                    }
                },
                // First hidden layer processes combined features
                tf.layers.dense({
                    units: hiddenDim,
                    activation: 'relu',
                    name: 'item_hidden1'
                }),
                // Second hidden layer for deeper feature learning
                tf.layers.dense({
                    units: hiddenDim / 2,
                    activation: 'relu',
                    name: 'item_hidden2'
                }),
                // Output embedding for similarity computation
                tf.layers.dense({
                    units: embeddingDim,
                    activation: 'linear',
                    name: 'item_output'
                })
            ]
        });
        
        // Adam optimizer for stable training with adaptive learning rates
        this.optimizer = tf.train.adam(0.001);
    }
    
    /**
     * User tower forward pass: convert user IDs to embeddings
     * The neural network learns to represent users in embedding space
     * based on their interaction patterns
     */
    userForward(userIndices) {
        return tf.tidy(() => {
            const userTensor = tf.tensor1d(userIndices, 'int32').expandDims(-1);
            return this.userTower.apply(userTensor);
        });
    }
    
    /**
     * Item tower forward pass: convert item IDs and genres to embeddings  
     * Enhanced with genre information to better capture movie characteristics
     * and improve representation learning
     */
    itemForward(itemIndices, genreIndices) {
        return tf.tidy(() => {
            const itemTensor = tf.tensor1d(itemIndices, 'int32').expandDims(-1);
            const genreTensor = tf.tensor1d(genreIndices, 'int32').expandDims(-1);
            
            // For the custom layer structure, we need to handle the dual input
            const itemEmbedding = tf.layers.embedding({
                inputDim: this.numItems,
                outputDim: this.embeddingDim
            }).apply(itemTensor);
            
            const flatItem = tf.layers.flatten().apply(itemEmbedding);
            
            const genreEmbedding = tf.layers.embedding({
                inputDim: this.numGenres,
                outputDim: Math.floor(this.embeddingDim / 4)
            }).apply(genreTensor);
            
            const flatGenre = tf.layers.flatten().apply(genreEmbedding);
            
            const concatenated = tf.layers.concatenate().apply([flatItem, flatGenre]);
            
            // Pass through the rest of the item tower
            const hidden1 = tf.layers.dense({
                units: this.hiddenDim,
                activation: 'relu'
            }).apply(concatenated);
            
            const hidden2 = tf.layers.dense({
                units: this.hiddenDim / 2,
                activation: 'relu'
            }).apply(hidden1);
            
            const output = tf.layers.dense({
                units: this.embeddingDim,
                activation: 'linear'
            }).apply(hidden2);
            
            return output;
        });
    }
    
    /**
     * Scoring function: dot product between user and item embeddings
     * Dot product is computationally efficient and commonly used in 
     * retrieval systems for large-scale recommendation
     */
    score(userEmbeddings, itemEmbeddings) {
        return tf.sum(tf.mul(userEmbeddings, itemEmbeddings), -1);
    }
    
    /**
     * Training step with in-batch sampled softmax loss
     * 
     * In-batch negatives: for each positive (user, item) pair in the batch,
     * all other items in the batch serve as negative examples. This is
     * computationally efficient and provides good signal for learning.
     * 
     * The loss encourages positive pairs to have higher similarity scores
     * than negative pairs through softmax cross-entropy.
     */
    async trainStep(userIndices, itemIndices, genreIndices) {
        return await tf.tidy(() => {
            // Forward pass through both towers
            const userEmbs = this.userForward(userIndices);
            const itemEmbs = this.itemForward(itemIndices, genreIndices);
            
            // Compute similarity matrix: batch_size x batch_size
            // Each element (i,j) is similarity between user i and item j
            const logits = tf.matMul(userEmbs, itemEmbs, false, true);
            
            // Labels: diagonal elements are positive pairs
            // We want user i to be most similar to item i in the batch
            const labels = tf.oneHot(
                tf.range(0, userIndices.length, 1, 'int32'), 
                userIndices.length
            );
            
            // Softmax cross entropy loss
            // This creates a multi-class classification problem where
            // the positive item is the target class among all batch items
            const loss = tf.losses.softmaxCrossEntropy(labels, logits);
            
            // Compute gradients and update both towers
            const variables = [
                ...this.userTower.getWeights(),
                // Include all item tower components
                ...this.getItemTowerWeights()
            ];
            
            const grads = tf.grads(() => loss);
            const gradArrays = grads(variables);
            
            // Apply gradients
            this.optimizer.applyGradients(gradArrays.map((grad, i) => ({
                tensor: variables[i],
                gradTensor: grad
            })));
            
            return loss.dataSync()[0];
        });
    }
    
    /**
     * Get all weights for the item tower (including embeddings)
     */
    getItemTowerWeights() {
        const weights = [];
        
        // Item embedding weights
        const itemEmbeddingLayer = tf.layers.embedding({
            inputDim: this.numItems,
            outputDim: this.embeddingDim
        });
        weights.push(...itemEmbeddingLayer.getWeights());
        
        // Genre embedding weights  
        const genreEmbeddingLayer = tf.layers.embedding({
            inputDim: this.numGenres,
            outputDim: Math.floor(this.embeddingDim / 4)
        });
        weights.push(...genreEmbeddingLayer.getWeights());
        
        // Dense layer weights
        const hidden1 = tf.layers.dense({
            units: this.hiddenDim,
            activation: 'relu'
        });
        weights.push(...hidden1.getWeights());
        
        const hidden2 = tf.layers.dense({
            units: this.hiddenDim / 2,
            activation: 'relu'
        });
        weights.push(...hidden2.getWeights());
        
        const output = tf.layers.dense({
            units: this.embeddingDim,
            activation: 'linear'
        });
        weights.push(...output.getWeights());
        
        return weights;
    }
    
    /**
     * Get user embedding for inference
     */
    getUserEmbedding(userIndex) {
        return tf.tidy(() => {
            return this.userForward([userIndex]).squeeze();
        });
    }
    
    /**
     * Compute scores for all items given a user embedding
     * Efficient batched computation for recommendation
     */
    async getScoresForAllItems(userEmbedding) {
        return await tf.tidy(() => {
            // Get all item embeddings (this would need to be computed)
            // For now, we'll create a simple version that computes scores
            // In a real implementation, you'd precompute item embeddings
            
            // Simple implementation: create dummy item embeddings for demonstration
            const allItemIndices = Array.from({length: this.numItems}, (_, i) => i);
            const dummyGenreIndices = Array.from({length: this.numItems}, () => 0);
            const allItemEmbs = this.itemForward(allItemIndices, dummyGenreIndices);
            
            // Compute dot product with all item embeddings
            const scores = tf.dot(allItemEmbs, userEmbedding);
            return scores.dataSync();
        });
    }
    
    /**
     * Get item embeddings for visualization
     */
    getItemEmbeddings() {
        return tf.tidy(() => {
            // Create embeddings for all items with default genre
            const allItemIndices = Array.from({length: this.numItems}, (_, i) => i);
            const dummyGenreIndices = Array.from({length: this.numItems}, () => 0);
            return this.itemForward(allItemIndices, dummyGenreIndices);
        });
    }
}
