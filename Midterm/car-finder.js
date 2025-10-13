// car-finder.js
/**
 * Car Finder Model - Two-Tower Architecture for Car Recommendations
 * 
 * Learns embeddings for cars based on their features and matches them
 * with user preferences to provide personalized recommendations.
 */
class CarFinderModel {
    constructor(numBrands, numFuelTypes, numTransmissions, numSellerTypes, embeddingDim, hiddenDim = 64) {
        this.numBrands = numBrands;
        this.numFuelTypes = numFuelTypes;
        this.numTransmissions = numTransmissions;
        this.numSellerTypes = numSellerTypes;
        this.embeddingDim = embeddingDim;
        this.hiddenDim = hiddenDim;
        
        // Initialize embeddings and layers
        this.brandEmbedding = tf.layers.embedding({
            inputDim: numBrands,
            outputDim: Math.floor(embeddingDim / 4)
        });
        
        this.fuelEmbedding = tf.layers.embedding({
            inputDim: numFuelTypes,
            outputDim: Math.floor(embeddingDim / 8)
        });
        
        this.transmissionEmbedding = tf.layers.embedding({
            inputDim: numTransmissions,
            outputDim: Math.floor(embeddingDim / 8)
        });
        
        this.sellerEmbedding = tf.layers.embedding({
            inputDim: numSellerTypes,
            outputDim: Math.floor(embeddingDim / 8)
        });
        
        this.carDense1 = tf.layers.dense({
            units: hiddenDim,
            activation: 'relu'
        });
        
        this.carDense2 = tf.layers.dense({
            units: Math.floor(hiddenDim / 2),
            activation: 'relu'
        });
        
        this.carOutput = tf.layers.dense({
            units: embeddingDim,
            activation: 'linear'
        });
        
        this.userDense = tf.layers.dense({
            units: hiddenDim,
            activation: 'relu'
        });
        
        this.optimizer = tf.train.adam(0.001);
    }
    
    /**
     * Car tower forward pass: convert car features to embeddings
     */
    carForward(carIndices, brandIndices, fuelIndices, transmissionIndices, sellerIndices) {
        return tf.tidy(() => {
            // Create numerical features from car indices (simulating continuous features)
            const numericalFeatures = tf.ones([carIndices.length, 3]); // Placeholder for real numerical features
            
            // Get categorical embeddings
            const brandEmb = this.brandEmbedding.apply(tf.tensor1d(brandIndices, 'int32'));
            const fuelEmb = this.fuelEmbedding.apply(tf.tensor1d(fuelIndices, 'int32'));
            const transmissionEmb = this.transmissionEmbedding.apply(tf.tensor1d(transmissionIndices, 'int32'));
            const sellerEmb = this.sellerEmbedding.apply(tf.tensor1d(sellerIndices, 'int32'));
            
            // Flatten all embeddings
            const flatBrand = tf.layers.flatten().apply(brandEmb);
            const flatFuel = tf.layers.flatten().apply(fuelEmb);
            const flatTransmission = tf.layers.flatten().apply(transmissionEmb);
            const flatSeller = tf.layers.flatten().apply(sellerEmb);
            
            // Concatenate all features
            const concatenated = tf.layers.concatenate().apply([
                numericalFeatures,
                flatBrand,
                flatFuel,
                flatTransmission,
                flatSeller
            ]);
            
            // Pass through dense layers
            const hidden1 = this.carDense1.apply(concatenated);
            const hidden2 = this.carDense2.apply(hidden1);
            const output = this.carOutput.apply(hidden2);
            
            return output;
        });
    }
    
    /**
     * User tower forward pass: process user preference vector
     */
    userForward(userVector) {
        return tf.tidy(() => {
            const userTensor = tf.tensor2d([userVector]);
            return this.userDense.apply(userTensor);
        });
    }
    
    /**
     * Scoring function: cosine similarity between user and car embeddings
     */
    score(userEmbeddings, carEmbeddings) {
        return tf.tidy(() => {
            // Normalize embeddings for cosine similarity
            const userNorm = tf.norm(userEmbeddings, 2, -1, true);
            const carNorm = tf.norm(carEmbeddings, 2, -1, true);
            
            const normalizedUser = tf.div(userEmbeddings, userNorm);
            const normalizedCar = tf.div(carEmbeddings, carNorm);
            
            // Cosine similarity
            return tf.sum(tf.mul(normalizedUser, normalizedCar), -1);
        });
    }
    
    /**
     * Training step with in-batch negative sampling
     */
    async trainStep(carIndices, brandIndices, fuelIndices, transmissionIndices, sellerIndices) {
        return await tf.tidy(() => {
            // Forward pass through car tower
            const carEmbs = this.carForward(carIndices, brandIndices, fuelIndices, transmissionIndices, sellerIndices);
            
            // Create synthetic user preferences based on car features
            const userVectors = carEmbs.arraySync().map(emb => {
                // Create user vector that prefers similar cars
                return emb.map(val => val + (Math.random() - 0.5) * 0.1); // Add small noise
            });
            
            const userTensor = tf.tensor2d(userVectors);
            const userEmbs = this.userDense.apply(userTensor);
            
            // Compute similarity matrix
            const logits = this.score(userEmbs, carEmbs);
            
            // Labels: prefer similar cars (diagonal elements)
            const labels = tf.oneHot(
                tf.range(0, carIndices.length, 1, 'int32'),
                carIndices.length
            );
            
            // Softmax cross entropy loss
            const loss = tf.losses.softmaxCrossEntropy(labels, logits.expandDims(-1));
            
            // Get all trainable variables
            const variables = this.getTrainableVariables();
            
            // Compute gradients
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
     * Get all trainable variables
     */
    getTrainableVariables() {
        const variables = [];
        
        // Car tower variables
        variables.push(...this.brandEmbedding.getWeights());
        variables.push(...this.fuelEmbedding.getWeights());
        variables.push(...this.transmissionEmbedding.getWeights());
        variables.push(...this.sellerEmbedding.getWeights());
        variables.push(...this.carDense1.getWeights());
        variables.push(...this.carDense2.getWeights());
        variables.push(...this.carOutput.getWeights());
        
        // User tower variables
        variables.push(...this.userDense.getWeights());
        
        return variables;
    }
    
    /**
     * Get car embeddings for all cars
     */
    getCarEmbeddings() {
        return tf.tidy(() => {
            // Create dummy indices for demonstration
            const dummyIndices = Array.from({length: 20}, (_, i) => i);
            const dummyBrands = Array.from({length: 20}, (_, i) => i % this.numBrands);
            const dummyFuels = Array.from({length: 20}, (_, i) => i % this.numFuelTypes);
            const dummyTransmissions = Array.from({length: 20}, (_, i) => i % this.numTransmissions);
            const dummySellers = Array.from({length: 20}, (_, i) => i % this.numSellerTypes);
            
            return this.carForward(dummyIndices, dummyBrands, dummyFuels, dummyTransmissions, dummySellers);
        });
    }
    
    /**
     * Get scores for all cars given a user vector
     */
    async getScoresForAllCars(userVector) {
        return await tf.tidy(() => {
            // Get user embedding
            const userEmb = this.userForward(userVector);
            
            // Get all car embeddings
            const carEmbs = this.getCarEmbeddings();
            
            // Compute scores
            const scores = this.score(userEmb, carEmbs);
            
            return scores.dataSync();
        });
    }
}
