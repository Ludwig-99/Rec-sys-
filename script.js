// Initialize application when window loads
window.onload = async function() {
    try {
        // Add a slight delay for visual effect
        document.getElementById('result').innerHTML = 
            '<div class="loading">Loading movie data<span class="dots"></span></div>';
        
        await loadData();
        populateMoviesDropdown();
        document.getElementById('result').innerHTML = 
            '<span style="color: #27ae60;">Data loaded successfully!</span> Select a movie and click "Get Recommendations".';
    } catch (error) {
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
        resultElement.innerHTML = '<span style="color: #e74c3c;">Please select a movie first.</span>';
        return;
    }
    
    // Find the liked movie
    const likedMovie = movies.find(movie => movie.id === selectedMovieId);
    if (!likedMovie) {
        resultElement.innerHTML = '<span style="color: #e74c3c;">Error: Selected movie not found.</span>';
        return;
    }
    
    // Show loading animation
    resultElement.innerHTML = '<div class="loading">Calculating recommendations<span class="dots"></span></div>';
    
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
                const recommendationTitles = topRecommendations.map(movie => 
                    `<span class="movie-title">${movie.title}</span>`
                );
                
                resultElement.innerHTML = 
                    `Because you liked <span class="liked-movie">${likedMovie.title}</span>, we recommend:<br><br>` +
                    `<div class="recommendation-list">${recommendationTitles.join('<br>')}</div>`;
            } else {
                resultElement.innerHTML = 
                    `No recommendations found for <span class="liked-movie">${likedMovie.title}</span>.`;
            }
        } catch (error) {
            console.error('Error calculating recommendations:', error);
            resultElement.innerHTML = '<span style="color: #e74c3c;">An error occurred while calculating recommendations.</span>';
        }
    }, 800); // Slightly longer delay for a more polished feel
}

// Add CSS for loading animation
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
    
    .liked-movie {
        color: #e74c3c;
        font-weight: 600;
    }
    
    .movie-title {
        color: #27ae60;
        font-weight: 600;
    }
    
    .recommendation-list {
        line-height: 1.8;
        margin-top: 10px;
    }
`;
document.head.appendChild(style);
