class TwoTowerModel {
    constructor(numUsers, numItems, embDim = 32) {
        this.numUsers = numUsers;
        this.numItems = numItems;
        this.embDim = embDim;
        
        console.log(`🚀 Initializing Two-Tower model: ${numUsers} users, ${numItems} items, dim: ${embDim}`);
        
        // Simple embeddings - no complex MLP to avoid issues
        this.userEmbedding = tf.variable(
            tf.randomNormal([numUsers, embDim], 0, 0.01), 
            true, 
            'userEmbedding'
        );
        
        this.itemEmbedding = tf.variable(
            tf.randomNormal([numItems, embDim], 0, 0.01),
            true,
            'itemEmbedding'
        );
        
        this.optimizer = tf.train.adam(0.001);
        
        console.log('✅ Model initialized successfully');
    }

    userForward(userIndices) {
        return tf.tidy(() => {
            return tf.gather(this.userEmbedding, userIndices);
        });
    }

    itemForward(itemIndices) {
        return tf.tidy(() => {
            return tf.gather(this.itemEmbedding, itemIndices);
        });
    }

    score(userEmbeddings, itemEmbeddings) {
        return tf.tidy(() => {
            return tf.sum(tf.mul(userEmbeddings, itemEmbeddings), -1);
        });
    }

    async trainStep(userIndices, itemIndices) {
        return tf.tidy(() => {
            const userEmbs = this.userForward(userIndices);
            const itemEmbs = this.itemForward(itemIndices);
            
            const scores = tf.matMul(userEmbs, itemEmbs, false, true);
            const labels = tf.oneHot(
                tf.range(0, userIndices.shape[0]), 
                itemIndices.shape[0]
            );
            
            const loss = tf.losses.softmaxCrossEntropy(labels, scores);
            
            const gradients = this.optimizer.computeGradients(() => loss);
            this.optimizer.applyGradients(gradients);
            
            return loss.dataSync()[0];
        });
    }

    getUserEmbedding(userIndex) {
        return tf.tidy(() => {
            return this.userForward(tf.tensor1d([userIndex], 'int32'));
        });
    }

    getItemEmbeddings() {
        return tf.tidy(() => {
            const allItemIndices = tf.range(0, this.numItems, 1, 'int32');
            return this.itemForward(allItemIndices);
        });
    }

    async getScoresForAllItems(userEmbedding) {
        return tf.tidy(() => {
            const itemEmbeddings = this.getItemEmbeddings();
            return tf.matMul(userEmbedding, itemEmbeddings, false, true).squeeze();
        });
    }

    dispose() {
        [this.userEmbedding, this.itemEmbedding].forEach(tensor => {
            if (tensor) tensor.dispose();
        });
        if (this.optimizer) this.optimizer.dispose();
    }
}
