// Global variables
let model;
let userMapping = new Map(); // Map user IDs to indices
let movieMapping = new Map(); // Map movie IDs to indices
let reverseUserMapping = new Map(); // Map indices back to user IDs
let reverseMovieMapping = new Map(); // Map indices back to movie IDs

// Training configuration
const LATENT_DIM = 10;
const EPOCHS = 10;
const BATCH_SIZE = 64;
const LEARNING_RATE = 0.001;

// Initialize application when window loads
window.onload = async function() {
    try {
        document.getElementById('result').innerHTML = 
            '<div class="loading">Loading movie data<span class="dots"></span></div>';
        
        // Load data first
        await loadData();
        
        // Populate dropdowns
        populateUserDropdown();
        populateMovieDropdown();
        
        // Create mappings for model
        createMappings();
        
        // Start model training
        await trainModel();
        
    } catch (error) {
        console.error('Initialization error:', error);
        document.getElementById('result').innerHTML = 
            '<span style="color: #e74c3c;">Error during initialization. Check console for details.</span>';
    }
};

/**
 * Create mappings between original IDs and model indices
 */
function createMappings() {
    // Create user mappings
    const uniqueUsers = [...new Set(ratings.map(r => r.userId))].sort((a, b) => a - b);
    uniqueUsers.forEach((userId, index) => {
        userMapping.set(userId, index);
        reverseUserMapping.set(index, userId);
    });
    
    // Create movie mappings
    const uniqueMovies = [...new Set(ratings.map(r => r.itemId))].sort((a, b) => a - b);
    uniqueMovies.forEach((movieId, index) => {
        movieMapping.set(movieId, index);
        reverseMovieMapping.set(index, movieId);
    });
}

/**
 * Populate user dropdown with available users
 */
function populateUserDropdown() {
    const selectElement = document.getElementById('user-select');
    
    // Clear existing options except the first placeholder
    while (selectElement.options.length > 1) {
        selectElement.remove(1);
    }
    
    // Get unique users and sort them
    const uniqueUsers = [...new Set(ratings.map(r => r.userId))].sort((a, b) => a - b);
    
    // Add users to dropdown (limit to first 100 for performance)
    uniqueUsers.slice(0, 100).forEach(userId => {
        const option = document.createElement('option');
        option.value = userId;
        option.textContent = `User ${userId}`;
        selectElement.appendChild(option);
    });
}

/**
 * Populate movie dropdown with available movies
 */
function populateMovieDropdown() {
    const selectElement = document.getElementById('movie-select');
    
    // Clear existing options except the first placeholder
    while (selectElement.options.length > 1) {
        selectElement.remove(1);
    }
    
    // Sort movies alphabetically by title
    const sortedMovies = [...movies].sort((a, b) => a.title.localeCompare(b.title));
    
    // Add movies to dropdown
    sortedMovies.forEach(movie => {
        const option = document.createElement('option');
        option.value = movie.id;
        option.textContent = movie.title;
        selectElement.appendChild(option);
    });
}

/**
 * Create the Matrix Factorization model
 */
function createModel() {
    // User input - shape [null, 1]
    const userInput = tf.input({shape: [1], name: 'userInput'});
    
    // Movie input - shape [null, 1]
    const movieInput = tf.input({shape: [1], name: 'movieInput'});
    
    // User embedding layer
    const userEmbedding = tf.layers.embedding({
        inputDim: numUsers,
        outputDim: LATENT_DIM,
        name: 'userEmbedding'
    }).apply(userInput);
    
    // Movie embedding layer
    const movieEmbedding = tf.layers.embedding({
        inputDim: numMovies,
        outputDim: LATENT_DIM,
        name: 'movieEmbedding'
    }).apply(movieInput);
    
    // Flatten embeddings
    const userFlatten = tf.layers.flatten().apply(userEmbedding);
    const movieFlatten = tf.layers.flatten().apply(movieEmbedding);
    
    // Dot product of user and movie embeddings
    const dotProduct = tf.layers.dot({axes: 1}).apply([userFlatten, movieFlatten]);
    
    // Reshape to get a single output value
    const prediction = tf.layers.dense({
        units: 1,
        activation: 'linear',
        name: 'prediction'
    }).apply(dotProduct);
    
    // Create and return the model
    const model = tf.model({
        inputs: [userInput, movieInput],
        outputs: prediction
    });
    
    return model;
}

/**
 * Train the Matrix Factorization model
 */
async function trainModel() {
    try {
        document.getElementById('training-status').textContent = 'Creating model...';
        document.getElementById('result').innerHTML = 'Creating matrix factorization model...';
        
        // Create the model
        model = createModel();
        
        // Compile the model
        model.compile({
            optimizer: tf.train.adam(LEARNING_RATE),
            loss: 'meanSquaredError',
            metrics: ['mse']
        });
        
        // Prepare training data
        document.getElementById('training-status').textContent = 'Preparing training data...';
        
        const userIndices = ratings.map(r => userMapping.get(r.userId));
        const movieIndices = ratings.map(r => movieMapping.get(r.itemId));
        const ratingValues = ratings.map(r => r.rating);
        
        const userTensor = tf.tensor1d(userIndices, 'int32');
        const movieTensor = tf.tensor1d(movieIndices, 'int32');
        const ratingTensor = tf.tensor1d(ratingValues, 'float32');
        
        // Train the model
        document.getElementById('training-status').textContent = 'Training model...';
        document.getElementById('result').innerHTML = 'Training matrix factorization model. This may take a moment...';
        
        await model.fit([userTensor, movieTensor], ratingTensor, {
            epochs: EPOCHS,
            batchSize: BATCH_SIZE,
            validationSplit: 0.1,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    const progress = ((epoch + 1) / EPOCHS) * 100;
                    document.getElementById('progress-fill').style.width = `${progress}%`;
                    document.getElementById('progress-text').textContent = `${Math.round(progress)}%`;
                    document.getElementById('training-status').textContent = 
                        `Epoch ${epoch + 1}/${EPOCHS} - Loss: ${logs.loss.toFixed(4)}`;
                    
                    // Update result with training progress
                    if (epoch === 0 || (epoch + 1) % 2 === 0) {
                        document.getElementById('result').innerHTML = 
                            `Training progress: ${epoch + 1}/${EPOCHS} epochs completed. Loss: ${logs.loss.toFixed(4)}`;
                    }
                },
                onTrainEnd: () => {
                    document.getElementById('training-status').textContent = 'Training completed!';
                    document.getElementById('result').innerHTML = 
                        '<span style="color: #27ae60;">Model training completed! Select a user and movie to predict ratings.</span>';
                    
                    // Enable predict button
                    document.getElementById('predict-btn').disabled = false;
                }
            }
        });
        
        // Clean up tensors
        userTensor.dispose();
        movieTensor.dispose();
        ratingTensor.dispose();
        
    } catch (error) {
        console.error('Training error:', error);
        document.getElementById('result').innerHTML = 
            '<span style="color: #e74c3c;">Error during model training. Check console for details.</span>';
        document.getElementById('training-status').textContent = 'Training failed';
    }
}

/**
 * Predict rating for selected user and movie
 */
async function predictRating() {
    const resultElement = document.getElementById('result');
    
    // Get selected values
    const userSelect = document.getElementById('user-select');
    const movieSelect = document.getElementById('movie-select');
    
    const selectedUserId = parseInt(userSelect.value);
    const selectedMovieId = parseInt(movieSelect.value);
    
    // Validate selection
    if (isNaN(selectedUserId) || isNaN(selectedMovieId)) {
        resultElement.innerHTML = '<span style="color: #e74c3c;">Please select both a user and a movie.</span>';
        return;
    }
    
    // Check if model is trained
    if (!model) {
        resultElement.innerHTML = '<span style="color: #e74c3c;">Model is not trained yet. Please wait.</span>';
        return;
    }
    
    try {
        // Get movie title
        const movie = movies.find(m => m.id === selectedMovieId);
        const movieTitle = movie ? movie.title : `Movie ${selectedMovieId}`;
        
        // Show loading
        resultElement.innerHTML = '<div class="loading">Calculating prediction<span class="dots"></span></div>';
        
        // Get mapped indices
        const userIndex = userMapping.get(selectedUserId);
        const movieIndex = movieMapping.get(selectedMovieId);
        
        if (userIndex === undefined || movieIndex === undefined) {
            resultElement.innerHTML = '<span style="color: #e74c3c;">Invalid user or movie selection.</span>';
            return;
        }
        
        // Create input tensors
        const userTensor = tf.tensor2d([[userIndex]], [1, 1], 'int32');
        const movieTensor = tf.tensor2d([[movieIndex]], [1, 1], 'int32');
        
        // Make prediction
        const prediction = model.predict([userTensor, movieTensor]);
        const predictedRating = await prediction.data();
        const finalRating = Math.min(5, Math.max(0.5, predictedRating[0])).toFixed(1);
        
        // Display result
        resultElement.innerHTML = `
            <div class="prediction-result">
                <div class="prediction-header">Rating Prediction:</div>
                <div class="prediction-value">${finalRating}/5.0</div>
                <div class="prediction-details">
                    User ${selectedUserId} would likely rate<br>
                    <strong>"${movieTitle}"</strong><br>
                    ${finalRating} stars
                </div>
            </div>
        `;
        
        // Clean up tensors
        userTensor.dispose();
        movieTensor.dispose();
        prediction.dispose();
        
    } catch (error) {
        console.error('Prediction error:', error);
        resultElement.innerHTML = '<span style="color: #e74c3c;">Error during prediction. Check console for details.</span>';
    }
}

// Add CSS for loading animation and prediction styling
const style = document.createElement('style');
style.textContent = `
    .loading {
        display: flex;
        align-items: center;
        justify-content: center;
        color: #3498db;
        font-weight: 500;
    }
    
    .dots::after {
        content: '';
        animation: dots 1.5s infinite;
    }
    
    @keyframes dots {
        0%, 20% { content: '.'; }
        40% { content: '..'; }
        60%, 100% { content: '...'; }
    }
    
    .prediction-result {
        text-align: center;
    }
    
    .prediction-header {
        font-size: 16px;
        color: #7f8c8d;
        margin-bottom: 10px;
    }
    
    .prediction-value {
        font-size: 32px;
        font-weight: 700;
        color: #e74c3c;
        margin: 10px 0;
    }
    
    .prediction-details {
        font-size: 14px;
        color: #2c3e50;
        line-height: 1.5;
        margin-top: 15px;
    }
`;
document.head.appendChild(style);
