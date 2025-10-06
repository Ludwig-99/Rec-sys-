class TwoTowerModel {
    constructor(numUsers, numItems, embDim) {
        this.numUsers = numUsers;
        this.numItems = numItems;
        this.embDim = embDim;
        
        // Two-Tower architecture with one hidden layer as required
        // User tower: user_id -> embedding -> hidden layer -> user representation
        this.userEmbedding = tf.variable(
            tf.randomNormal([numUsers, embDim], 0, 0.05), 
            true, 
            'userEmbedding'
        );
        
        // Item tower: item_id -> embedding -> hidden layer -> item representation  
        this.itemEmbedding = tf.variable(
            tf.randomNormal([numItems, embDim], 0, 0.05),
            true,
            'itemEmbedding'
        );
        
        // Hidden layers for both towers to add deep learning capability
        // User tower hidden layer weights
        this.userHiddenWeights = tf.variable(
            tf.randomNormal([embDim, embDim], 0, 0.05),
            true,
            'userHiddenWeights'
        );
        
        this.userHiddenBias = tf.variable(
            tf.zeros([embDim]),
            true,
            'userHiddenBias'
        );
        
        // Item tower hidden layer weights
        this.itemHiddenWeights = tf.variable(
            tf.randomNormal([embDim, embDim], 0, 0.05),
            true,
            'itemHiddenWeights'
        );
        
        this.itemHiddenBias = tf.variable(
            tf.zeros([embDim]),
            true,
            'itemHiddenBias'
        );
        
        this.optimizer = tf.train.adam(0.001);
    }

    // User tower forward pass with one hidden layer
    userForward(userIndices) {
        return tf.tidy(() => {
            // Look up user embeddings
            const userEmb = tf.gather(this.userEmbedding, userIndices);
            
            // Pass through hidden layer with ReLU activation
            const hiddenOutput = tf.relu(
                tf.add(tf.matMul(userEmb, this.userHiddenWeights), this.userHiddenBias)
            );
            
            return hiddenOutput;
        });
    }

    // Item tower forward pass with one hidden layer  
    itemForward(itemIndices) {
        return tf.tidy(() => {
            // Look up item embeddings
            const itemEmb = tf.gather(this.itemEmbedding, itemIndices);
            
            // Pass through hidden layer with ReLU activation
            const hiddenOutput = tf.relu(
                tf.add(tf.matMul(itemEmb, this.itemHiddenWeights), this.itemHiddenBias)
            );
            
            return hiddenOutput;
        });
    }

    // Score function using dot product similarity
    // Dot product measures cosine similarity when embeddings are normalized
    score(userEmbeddings, itemEmbeddings) {
        return tf.tidy(() => {
            // Normalize embeddings for stable training
            const normalizedUser = tf.div(userEmbeddings, tf.norm(userEmbeddings, 2, -1, true));
            const normalizedItem = tf.div(itemEmbeddings, tf.norm(itemEmbeddings, 2, -1, true));
            
            // Compute dot product along the last dimension
            return tf.sum(tf.mul(normalizedUser, normalizedItem), -1);
        });
    }

    // Training step using in-batch sampled softmax loss
    // This creates a multi-class classification problem where the positive item
    // should score higher than all other items in the batch
    async trainStep(userIndices, itemIndices) {
        return tf.tidy(() => {
            const userEmbs = this.userForward(userIndices);
            const itemEmbs = this.itemForward(itemIndices);
            
            // Compute scores for all user-item pairs in batch
            // This creates a matrix where each user should score highest with their positive item
            const scores = tf.matMul(userEmbs, itemEmbs, false, true);
            
            // Create labels: diagonal elements are positives
            const labels = tf.oneHot(
                tf.range(0, userIndices.shape[0]), 
                itemIndices.shape[0]
            );
            
            // Compute softmax cross entropy loss
            const loss = tf.losses.softmaxCrossEntropy(labels, scores);
            
            // Apply gradients
            this.optimizer.minimize(() => loss);
            
            return loss.dataSync()[0];
        });
    }

    // Get user embedding for inference
    getUserEmbedding(userIndex) {
        return tf.tidy(() => {
            return this.userForward(tf.tensor1d([userIndex], 'int32'));
        });
    }

    // Get all item embeddings for scoring
    getItemEmbeddings() {
        return tf.tidy(() => {
            const allItemIndices = tf.range(0, this.numItems, 1, 'int32');
            return this.itemForward(allItemIndices);
        });
    }

    // Get scores for a user against all items
    async getScoresForAllItems(userEmbedding) {
        return tf.tidy(() => {
            const itemEmbeddings = this.getItemEmbeddings();
            
            // Expand user embedding to match item dimensions for batch scoring
            const expandedUser = userEmbedding.expandDims(1);
            const expandedItems = itemEmbeddings.expandDims(0);
            
            // Compute scores using the same scoring function
            return this.score(expandedUser, expandedItems).squeeze();
        });
    }
}
