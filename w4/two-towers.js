class TwoTowerModel {
    constructor(numUsers, numItems, embDim, numGenres = 18) {
        this.numUsers = numUsers;
        this.numItems = numItems;
        this.embDim = embDim;
        this.numGenres = numGenres;
        
        console.log(`🚀 Initializing Two-Tower model with ${numUsers} users, ${numItems} items, ${numGenres} genres, embedding dim: ${embDim}`);
        
        // Simple Two-Tower architecture without complex MLP to avoid freezing
        this.userEmbedding = tf.variable(
            tf.randomNormal([numUsers, embDim], 0, 0.01), 
            true, 
            'userEmbedding'
        );
        
        // Item embedding combined with genre information
        this.itemEmbedding = tf.variable(
            tf.randomNormal([numItems, embDim], 0, 0.01),
            true,
            'itemEmbedding'
        );
        
        // Genre embeddings (optional - can be removed if causing issues)
        this.genreEmbedding = tf.variable(
            tf.randomNormal([numGenres, 8], 0, 0.01), // Small genre embeddings
            true,
            'genreEmbedding'
        );
        
        this.optimizer = tf.train.adam(0.001);
        
        console.log('✅ Model initialized successfully');
    }

    // Simple user forward pass - just embedding lookup
    userForward(userIndices) {
        return tf.tidy(() => {
            return tf.gather(this.userEmbedding, userIndices);
        });
    }

    // Item forward pass with optional genre information
    itemForward(itemIndices, itemGenres = null) {
        return tf.tidy(() => {
            const baseEmbedding = tf.gather(this.itemEmbedding, itemIndices);
            
            // If genre information is provided, combine with base embedding
            if (itemGenres && this.genreEmbedding) {
                try {
                    const genreEmb = tf.gather(this.genreEmbedding, itemGenres);
                    const genreMean = tf.mean(genreEmb, 1); // Average genre embeddings
                    return tf.add(baseEmbedding, genreMean);
                } catch (error) {
                    console.warn('Genre embedding failed, using base embedding:', error);
                    return baseEmbedding;
                }
            }
            
            return baseEmbedding;
        });
    }

    // Score function using dot product
    score(userEmbeddings, itemEmbeddings) {
        return tf.tidy(() => {
            // Simple dot product without normalization for stability
            return tf.sum(tf.mul(userEmbeddings, itemEmbeddings), -1);
        });
    }

    // Training step with in-batch negatives
    async trainStep(userIndices, itemIndices) {
        return tf.tidy(() => {
            const userEmbs = this.userForward(userIndices);
            const itemEmbs = this.itemForward(itemIndices);
            
            // Compute similarity matrix
            const scores = tf.matMul(userEmbs, itemEmbs, false, true);
            
            // Labels: diagonal is positive
            const labels = tf.oneHot(
                tf.range(0, userIndices.shape[0]), 
                itemIndices.shape[0]
            );
            
            // Softmax cross entropy loss
            const loss = tf.losses.softmaxCrossEntropy(labels, scores);
            
            // Gradient update
            const gradients = this.optimizer.computeGradients(() => loss);
            this.optimizer.applyGradients(gradients);
            
            return loss.dataSync()[0];
        });
    }

    // Get user embedding for inference
    getUserEmbedding(userIndex) {
        return tf.tidy(() => {
            return this.userForward(tf.tensor1d([userIndex], 'int32'));
        });
    }

    // Get all item embeddings
    getItemEmbeddings() {
        return tf.tidy(() => {
            const allItemIndices = tf.range(0, this.numItems, 1, 'int32');
            return this.itemForward(allItemIndices);
        });
    }

    // Get scores for user against all items
    async getScoresForAllItems(userEmbedding) {
        return tf.tidy(() => {
            const itemEmbeddings = this.getItemEmbeddings();
            return tf.matMul(userEmbedding, itemEmbeddings, false, true).squeeze();
        });
    }

    // Cleanup
    dispose() {
        if (this.userEmbedding) this.userEmbedding.dispose();
        if (this.itemEmbedding) this.itemEmbedding.dispose();
        if (this.genreEmbedding) this.genreEmbedding.dispose();
        if (this.optimizer) this.optimizer.dispose();
    }
}
