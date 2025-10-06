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
                    outputDim: embeddingDim
