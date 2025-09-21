// Global variables for movie and rating data
let movies = [];
let ratings = [];

// Genre names in the order they appear in the u.item file
const genreNames = [
    "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
    "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western"
];

/**
 * Asynchronously loads and parses movie and rating data
 */
async function loadData() {
    try {
        // Load and parse movie data
        const moviesResponse = await fetch('u.item');
        if (!moviesResponse.ok) {
            throw new Error(`Failed to load movie data: ${moviesResponse.status}`);
        }
        const moviesText = await moviesResponse.text();
        parseItemData(moviesText);

        // Load and parse rating data
        const ratingsResponse = await fetch('u.data');
        if (!ratingsResponse.ok) {
            throw new Error(`Failed to load rating data: ${ratingsResponse.status}`);
        }
        const ratingsText = await ratingsResponse.text();
        parseRatingData(ratingsText);
    } catch (error) {
        console.error('Error loading data:', error);
        document.getElementById('result').innerText = 
            `Error: ${error.message}. Please make sure u.item and u.data files are in the correct location.`;
        throw error; // Re-throw to allow script.js to handle the error
    }
}

/**
 * Parses movie data from the u.item file format
 * @param {string} text - Raw text content from u.item file
 */
function parseItemData(text) {
    const lines = text.split('\n');
    
    for (const line of lines) {
        if (line.trim() === '') continue;
        
        const fields = line.split('|');
        if (fields.length < 5) continue; // Skip invalid lines
        
        const id = parseInt(fields[0]);
        const title = fields[1];
        const genres = [];
        
        // Extract genre information (fields 5-23)
        for (let i = 0; i < 18; i++) {
            const genreIndex = i + 5;
            if (genreIndex < fields.length && fields[genreIndex] === '1') {
                genres.push(genreNames[i]);
            }
        }
        
        movies.push({ id, title, genres });
    }
}

/**
 * Parses rating data from the u.data file format
 * @param {string} text - Raw text content from u.data file
 */
function parseRatingData(text) {
    const lines = text.split('\n');
    
    for (const line of lines) {
        if (line.trim() === '') continue;
        
        const fields = line.split('\t');
        if (fields.length < 4) continue; // Skip invalid lines
        
        const userId = parseInt(fields[0]);
        const itemId = parseInt(fields[1]);
        const rating = parseFloat(fields[2]);
        const timestamp = parseInt(fields[3]);
        
        ratings.push({ userId, itemId, rating, timestamp });
    }
}
