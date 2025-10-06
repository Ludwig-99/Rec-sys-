async loadData() {
    this.updateStatus('Loading data...');
    
    try {
        // Load interactions data
        const response1 = await fetch('data/u.data');
        if (!response1.ok) throw new Error('Failed to load u.data');
        const data1 = await response1.text();
        this.parseInteractions(data1);
        
        // Load items data
        const response2 = await fetch('data/u.item');
        if (!response2.ok) throw new Error('Failed to load u.item');
        const data2 = await response2.text();
        
        // Load genres data
        const response3 = await fetch('data/u.genre');
        if (!response3.ok) {
            console.warn('u.genre not found, using default genres');
            this.setupDefaultGenres();
        } else {
            const genreData = await response3.text();
            this.parseGenres(genreData);
        }
        
        this.parseItems(data2);
        this.prepareMappings();
        this.prepareUserRatings();
        
        document.getElementById('train').disabled = false;
        this.updateStatus(`Data loaded: ${this.interactions.length} interactions, ${this.items.size} movies, ${this.userMap.size} users, ${this.genres.size} genres`);
        
        console.log('Data loaded successfully');
    } catch (error) {
        this.updateStatus('Error loading data: ' + error.message);
        console.error('Error details:', error);
    }
}

parseGenres(data) {
    this.genres = new Map();
    const lines = data.trim().split('\n');
    
    console.log('Parsing genres file:', lines);
    
    lines.forEach(line => {
        if (line.trim()) {
            const parts = line.split('|');
            if (parts.length >= 2) {
                const genreName = parts[0].trim();
                const genreId = parseInt(parts[1]);
                if (!isNaN(genreId) && genreName) {
                    this.genres.set(genreId, genreName);
                }
            }
        }
    });
    
    console.log('Loaded genres:', Array.from(this.genres.entries()));
}

setupDefaultGenres() {
    this.genres = new Map([
        [0, "unknown"],
        [1, "Action"],
        [2, "Adventure"],
        [3, "Animation"],
        [4, "Children's"],
        [5, "Comedy"],
        [6, "Crime"],
        [7, "Documentary"],
        [8, "Drama"],
        [9, "Fantasy"],
        [10, "Film-Noir"],
        [11, "Horror"],
        [12, "Musical"],
        [13, "Mystery"],
        [14, "Romance"],
        [15, "Sci-Fi"],
        [16, "Thriller"],
        [17, "War"],
        [18, "Western"]
    ]);
}

parseItems(data) {
    const lines = data.trim().split('\n');
    this.items = new Map();
    
    console.log('Parsing items, total lines:', lines.length);
    
    lines.forEach((line, index) => {
        try {
            const parts = line.split('|');
            if (parts.length >= 24) {
                const itemId = parseInt(parts[0]);
                const title = parts[1];
                const yearMatch = title.match(/\((\d{4})\)$/);
                const year = yearMatch ? parseInt(yearMatch[1]) : null;
                
                // Parse genres (columns 6-24 in u.item format represent genre flags)
                const genreFlags = parts.slice(5, 24).map(val => parseInt(val));
                const movieGenres = [];
                
                genreFlags.forEach((flag, index) => {
                    if (flag === 1) {
                        movieGenres.push(index);
                    }
                });
                
                this.items.set(itemId, {
                    title: title,
                    year: year,
                    genres: movieGenres
                });
            }
        } catch (error) {
            console.warn(`Error parsing line ${index}:`, error);
        }
    });
    
    console.log('Successfully parsed items:', this.items.size);
    if (this.items.size > 0) {
        const firstItem = Array.from(this.items.entries())[0];
        console.log('Sample item:', firstItem);
    }
}
