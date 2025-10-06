// two-tower.js
/**
 * Two-Tower Model for Movie Recommendation
 * 
 * Two-tower architecture learns separate embeddings for users and items,
 * then computes similarity via dot product. This is efficient for retrieval
 * and scales to large catalogs.
 */
class TwoTowerModel {
    constructor(numUsers, numItems, embeddingDim, hiddenDim = 64) {
        this.numUsers = numUsers;
        this.numItems = numItems;
        this.embeddingDim = embeddingDim;
        this.hiddenDim = hiddenDim;
        
        // User Tower: maps user IDs to embeddings through neural network
        // The hidden layers allow learning non-linear user preferences
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
                // Hidden layer captures complex user preference patterns
                tf.layers.dense({
                    units: hiddenDim,
                    activation: 'relu',
                    name: 'user_hidden'
                }),
                // Output embedding for similarity computation
                tf.layers.dense({
                    units: embeddingDim,
                    activation: 'linear',
                    name: 'user_output'
                })
            ]
        });
        
        // Item Tower: maps item IDs to embeddings through neural network  
        // Learns movie characteristics and attributes in latent space
        this.itemTower = tf.sequential({
            layers: [
                // Embedding layer converts sparse item IDs to dense vectors
                tf.layers.embedding({
                    inputDim: numItems,
                    outputDim: embeddingDim,
                    inputLength: 1,
                    name: 'item_embedding'
                }),
                tf.layers.flatten(),
                // Hidden layer captures complex item characteristics
                tf.layers.dense({
                    units: hiddenDim,
                    activation: 'relu',
                    name: 'item_hidden'
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
     * Item tower forward pass: convert item IDs to embeddings  
     * The neural network learns to represent items in embedding space
     * based on how users interact with them
     */
    itemForward(itemIndices) {
        return tf.tidy(() => {
            const itemTensor = tf.tensor1d(itemIndices, 'int32').expandDims(-1);
            return this.itemTower.apply(itemTensor);
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
    async trainStep(userIndices, itemIndices) {
        return await tf.tidy(() => {
            // Forward pass through both towers
            const userEmbs = this.userForward(userIndices);
            const itemEmbs = this.itemForward(itemIndices);
            
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
            const loss = () => tf.losses.softmaxCrossEntropy(labels, logits);
            
            // Compute gradients and update both towers
            const { value, grads } = this.optimizer.computeGradients(loss);
            
            this.optimizer.applyGradients(g
