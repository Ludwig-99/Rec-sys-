// two-tower.js
class MLPGenreModel {
    constructor(numUsers, numGenres, embDim, hiddenDim = 64) {
        this.numUsers = numUsers;
        this.numGenres = numGenres;
        this.embDim = embDim;
        this.hiddenDim = hiddenDim;
        
        // Build MLP model with concatenated user and genre embeddings
        this.model = tf.sequential({
            layers: [
                // The model will take two inputs: user indices and genre indices
                // We'll handle concatenation in the forward pass
                tf.layers.dense({
                    inputShape: [embDim + hiddenDim], // Concatenated embeddings
                    units: hiddenDim,
                    activation: 'relu',
                    name: 'hidden_layer'
                }),
                tf.layers.dense({
                    units: 1,
                    activation: 'sigmoid',
                    name: 'output_layer'
                })
            ]
        });
        
        // Separate embedding layers
        this.userEmbedding = tf.layers.embedding({
            inputDim: numUsers,
            outputDim: embDim,
            name: 'user_embedding'
        });
        
        this.genreEmbedding = tf.layers.embedding({
            inputDim: numGenres,
            outputDim: hiddenDim,
            name: 'genre_embedding'
        });
        
        // Adam optimizer
        this.optimizer = tf.train.adam(0.001);
    }
    
    // Forward pass: concatenate user and genre embeddings, then pass through MLP
    forward(userIndices, genreIndices) {
        return tf.tidy(() => {
            const userTensor = tf.tensor1d(userIndices, 'int32');
            const genreTensor = tf.tensor1d(genreIndices, 'int32');
            
            const userEmbs = this.userEmbedding.apply(userTensor);
            const genreEmbs = this.genreEmbedding.apply(genreTensor);
            
            // Concatenate embeddings along the last dimension
            const concatenated = tf.concat([userEmbs, genreEmbs], -1);
            
            return this.model.apply(concatenated).squeeze();
        });
    }
    
    // Score function: returns prediction probability
    score(predictions) {
        return predictions;
    }
    
    async trainStep(userIndices, genreIndices, labels) {
        return await tf.tidy(() => {
            const loss = () => {
                const predictions = this.forward(userIndices, genreIndices);
                const labelTensor = tf.tensor1d(labels, 'float32');
                
                // Binary crossentropy loss for rating prediction
                const loss = tf.losses.sigmoidCrossEntropy(labelTensor, predictions);
                return loss;
            };
            
            // Compute gradients and update model
            const { value, grads } = this.optimizer.computeGradients(loss);
            this.optimizer.applyGradients(grads);
            
            return value.dataSync()[0];
        });
    }
    
    // Get predictions for a user across all genres
    async getPredictionsForUser(userIndex, allGenres) {
        return await tf.tidy(() => {
            const userIndices = Array(allGenres.length).fill(userIndex);
            const predictions = this.forward(userIndices, allGenres);
            const scores = predictions.dataSync();
            
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
    
    // Get genre embeddings for visualization
    async getGenreEmbeddings() {
        return await tf.tidy(() => {
            const allGenreIndices = Array.from({length: this.numGenres}, (_, i) => i);
            const genreTensor = tf.tensor1d(allGenreIndices, 'int32');
            const embeddings = this.genreEmbedding.apply(genreTensor);
            return embeddings.arraySync();
        });
    }
    
    // Get user embedding for a specific user
    getUserEmbedding(userIndex) {
        return tf.tidy(() => {
            const userTensor = tf.tensor1d([userIndex], 'int32');
            return this.userEmbedding.apply(userTensor).squeeze().arraySync();
        });
    }
}
