async loadData() {
    this.updateStatus('Loading data...');
    
    try {
        // Load interactions data
        const response1 = await fetch('data/u.data');
        const data1 = await response1.text();
        this.parseInteractions(data1);
        
        // Load items data
        const response2 = await fetch('data/u.item');
        const data2 = await response2.text();
        
        // Load genres data
        const response3 = await fetch('data/u.genre');
        const genreData = await response3.text();
        this.parseGenres(genreData);
        
        this.parseItems(data2);
        
        this.prepareMappings();
        this.prepareUserRatings();
        
        document.getElementById('train').disabled = false;
        this.updateStatus(`Data loaded: ${this.interactions.length} interactions, ${this.items.size} movies, ${this.userMap.size} users, ${this.genres.size} genres`);
    } catch (error) {
        this.updateStatus('Error loading data: ' + error.message);
        console.error('Error details:', error);
    }
}

parseGenres(data) {
    this.genres = new Map();
    const lines = data.trim().split('\n');
    
    lines.forEach(line => {
        if (line.trim()) {
            const parts = line.split('|');
            if (parts.length >= 2) {
                const genreId = parseInt(parts[1]); // ID is in second position
                const genreName = parts[0].trim();
                if (!isNaN(genreId) && genreName) {
                    this.genres.set(genreId, genreName);
                }
            }
        }
    });
    
    console.log('Loaded genres:', Array.from(this.genres.entries()));
}

parseItems(data) {
    const lines = data.trim().split('\n');
    lines.forEach(line => {
        const parts = line.split('|');
        if (parts.length >= 24) {
            const itemId = parseInt(parts[0]);
            const title = parts[1];
            const yearMatch = title.match(/\((\d{4})\)$/);
            const year = yearMatch ? parseInt(yearMatch[1]) : null;
            
            // Parse genres (columns 6-24 in u.item format represent genre flags)
            const genreFlags = parts.slice(5, 24).map(val => parseInt(val));
            const movieGenres = genreFlags
                .map((flag, index) => flag === 1 ? index : -1)
                .filter(idx => idx !== -1);
            
            this.items.set(itemId, {
                title: title,
                year: year,
                genres: movieGenres
            });
        }
    });
    
    console.log('Sample item with genres:', Array.from(this.items.entries())[0]);
}
