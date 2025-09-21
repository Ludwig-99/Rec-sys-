// Initialize application when window loads
window.onload = async function() {
    try {
        await loadData();
        populateMoviesDropdown();
        document.getElementById('result').innerText = 
            "Data loaded. Please select a movie and click 'Get Recommendations'.";
    } catch (error) {
        // Error handling is already done in data.js
        console.error('Initialization error:', error);
    }
};

/**
 * Populates the movie dropdown with sorted movie titles
 */
function populateMoviesDropdown() {
    const selectElement = document.getElementById('movie-select');
    
    // Clear existing options except the first placeholder
    while (selectElement.options.length > 1) {
        selectElement.remove(1);
    }
    
    // Sort movies alphabetically by title
    const sortedMovies = [...movies].sort((a, b) => 
        a.title.localeCompare(b.title)
    );
    
    // Add movies to dropdown
    sortedMovies.forEach(movie => {
        const option = document.createElement('option');
        option.value = movie.id;
        option.textContent = movie.title;
        selectElement.appendChild(option);
    });
}

/**
 * Converts genre array to a binary vector representation
 * @param {Array} genres - Array of genre names
 * @returns {Array} Binary vector representation of genres
 */
function genresToVector(genres) {
    // Create a vector with the same length as genreNames
    const vector = new Array(genreNames.length).fill(0);
    
    // Set 1 for each genre present in the movie
    genres.forEach(genre => {
        const index = genreNames.indexOf(genre);
        if (index !== -1) {
            vector[index] = 1;
        }
    });
    
    return vector;
}

/**
 * Calculates cosine similarity between two vectors
 * @param {Array} vecA - First vector
 * @param {Array} vecB - Second vector
 * @returns {number} Cosine similarity score
 */
function cosineSimilarity(vecA, vecB) {
    // Calculate dot product
    let dotProduct = 0;
    for (let i = 0; i < vecA.length; i++) {
        dotProduct += vecA[i] * vecB[i];
    }
    
    // Calculate magnitudes
    const magnitudeA = Math.sqrt(vecA.reduce((sum, val) => sum + val * val, 0));
    const magnitudeB = Math.sqrt(vecB.reduce((sum, val) => sum + val * val, 0));
    
    // Avoid division by zero
    if (magnitudeA === 0 || magnitudeB === 0) {
        return 0;
    }
    
    // Return cosine similarity
    return dotProduct / (magnitudeA * magnitudeB);
}

/**
 * Calculates and displays movie recommendations based on selected movie
 */
function getRecommendations() {
    const resultElement = document.getElementById('result');
    
    // Get selected movie ID
    const selectElement = document.getElementById('movie-select');
    const selectedMovieId = parseInt(selectElement.value);
    
    // Validate selection
    if (isNaN(selectedMovieId)) {
        resultElement.innerText = "Please select a movie first.";
        return;
    }
    
    // Find the liked movie
    const likedMovie = movies.find(movie => movie.id === selectedMovieId);
    if (!likedMovie) {
        resultElement.innerText = "Error: Selected movie not found.";
        return;
    }
    
    // Show loading message
    resultElement.innerText = "Calculating recommendations...";
    
    // Use setTimeout to allow UI to update before heavy computation
    setTimeout(() => {
        try {
            // Convert liked movie's genres to vector representation
            const likedMovieVector = genresToVector(likedMovie.genres);
            
            // Filter out the liked movie from candidates
            const candidateMovies = movies.filter(movie => movie.id !== likedMovie.id);
            
            // Calculate cosine similarity for each candidate movie
            const scoredMovies = candidateMovies.map(candidate => {
                // Convert candidate movie's genres to vector representation
                const candidateVector = genresToVector(candidate.genres);
                
                // Calculate cosine similarity
                const score = cosineSimilarity(likedMovieVector, candidateVector);
                
                return { ...candidate, score };
            });
            
            // Sort by score in descending order
            scoredMovies.sort((a, b) => b.score - a.score);
            
            // Get top 2 recommendations
            const topRecommendations = scoredMovies.slice(0, 2);
            
            // Display results
            if (topRecommendations.length > 0) {
                const recommendationTitles = topRecommendations.map(movie => movie.title);
                resultElement.innerHTML = 
                    `Because you liked <strong>${likedMovie.title}</strong>, we recommend:<br>` +
                    `<strong>${recommendationTitles.join('</strong>, <strong>')}</strong>`;
            } else {
                resultElement.innerText = 
                    `No recommendations found for '${likedMovie.title}'.`;
            }
        } catch (error) {
            console.error('Error calculating recommendations:', error);
            resultElement.innerText = "An error occurred while calculating recommendations.";
        }
    }, 10);
}
