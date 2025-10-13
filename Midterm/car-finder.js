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
        
        // Car Tower: learns embeddings based on car features
        this.carTower = {
            brandEmbedding: tf.layers.embedding({
                inputDim: numBrands,
                outputDim: Math.floor(embeddingDim / 4),
                name: 'brand_embedding'
            }),
            fuelEmbedding: tf.layers.embedding({
                inputDim: numFuelTypes,
                outputDim: Math.floor(embeddingDim / 8),
                name: 'fuel_embedding'
            }),
            transmissionEmbedding: tf.layers.embedding({
                inputDim: numTransmissions,
                outputDim: Math.floor(embeddingDim / 8),
                name: 'transmission_embedding'
            }),
            sellerEmbedding: tf.layers.embedding({
                inputDim: numSellerTypes,
                outputDim: Math.floor(embeddingDim / 8),
                name: 'seller_embedding'
            }),
            dense1: tf.layers.dense({
                units: hiddenDim,
                activation: 'relu',
                name: 'car_dense1'
            }),
            dense2: tf.layers.dense({
                units: Math.floor(hiddenDim / 2),
                activation: 'relu',
                name: 'car_dense2'
            }),
            output: tf.layers.dense({
                units: embeddingDim,
                activation: 'linear',
                name: 'car_output'
            })
        };
        
        // User Tower: processes user preferences
        this.userTower = tf.layers.dense({
            units: hiddenDim,
            activation: 'relu',
            name: 'user_dense1'
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
            const brandEmb = this.carTower.brandEmbedding.apply(tf.tensor1d(brandIndices, 'int32'));
            const fuelEmb = this.carTower.fuelEmbedding.apply(tf.tensor1d(fuelIndices, 'int32'));
            const transmissionEmb = this.carTower.transmissionEmbedding.apply(tf.tensor1d(transmissionIndices, 'int32'));
            const sellerEmb = this.carTower.sellerEmbedding.apply(tf.tensor1d(sellerIndices, 'int32'));
            
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
            const hidden1 = this.carTower.dense1.apply(concatenated);
            const hidden2 = this.carTower.dense2.apply(hidden1);
            const output = this.carTower.output.apply(hidden2);
            
            return output;
        });
    }
    
    /**
     * User tower forward pass: process user preference vector
     */
    userForward(userVector) {
        return tf.tidy(() => {
            const userTensor = tf.tensor2d([userVector]);
            return this.userTower.apply(userTensor);
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
            const userEmbs = this.userTower.apply(userTensor);
            
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
        variables.push(...this.carTower.brandEmbedding.getWeights());
        variables.push(...this.carTower.fuelEmbedding.getWeights());
        variables.push(...this.carTower.transmissionEmbedding.getWeights());
        variables.push(...this.carTower.sellerEmbedding.getWeights());
        variables.push(...this.carTower.dense1.getWeights());
        variables.push(...this.carTower.dense2.getWeights());
        variables.push(...this.carTower.output.getWeights());
        
        // User tower variables
        variables.push(...this.userTower.getWeights());
        
        return variables;
    }
    
    /**
     * Get car embeddings for all cars
     */
    getCarEmbeddings() {
        return tf.tidy(() => {
            // Create dummy indices for all possible combinations
            // In a real implementation, you'd pass actual car data
            const dummyIndices = Array.from({length: this.numBrands * 2}, (_, i) => i % this.numBrands);
            const dummyBrands = Array.from({length: this.numBrands * 2}, (_, i) => i % this.numBrands);
            const dummyFuels = Array.from({length: this.numBrands * 2}, (_, i) => i % this.numFuelTypes);
            const dummyTransmissions = Array.from({length: this.numBrands * 2}, (_, i) => i % this.numTransmissions);
            const dummySellers = Array.from({length: this.numBrands * 2}, (_, i) => i % this.numSellerTypes);
            
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
            
            // Get all car embeddings (simplified - using dummy data)
            const carEmbs = this.getCarEmbeddings();
            
            // Compute scores
            const scores = this.score(userEmb, carEmbs);
            
            return scores.dataSync();
        });
    }
}
