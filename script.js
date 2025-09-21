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
            // Create a set of liked movie's genres for faster lookup
            const likedGenresSet = new Set(likedMovie.genres);
            
            // Filter out the liked movie from candidates
            const candidateMovies = movies.filter(movie => movie.id !== likedMovie.id);
            
            // Calculate Jaccard similarity for each candidate movie
            const scoredMovies = candidateMovies.map(candidate => {
                const candidateGenresSet = new Set(candidate.genres);
                
                // Calculate intersection
                const intersection = new Set(
                    [...likedGenresSet].filter(genre => candidateGenresSet.has(genre))
                );
                
                // Calculate union
                const union = new Set([...likedGenresSet, ...candidateGenresSet]);
                
                // Calculate Jaccard similarity
                const score = union.size > 0 ? intersection.size / union.size : 0;
                
                return { ...candidate, score };
            });
            
            // Sort by score in descending order
            scoredMovies.sort((a, b) => b.score - a.score);
            
            // Get top 2 recommendations
            const topRecommendations = scoredMovies.slice(0, 2);
            
            // Display results
            if (topRecommendations.length > 0) {
                const recommendationTitles = topRecommendations.map(movie => movie.title);
                resultElement.innerText = 
                    `Because you liked '${likedMovie.title}', we recommend: ${recommendationTitles.join(', ')}`;
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
