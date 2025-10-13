// app.js
class CarFinderApp {
    constructor() {
        this.cars = [];
        this.userPreferences = {
            maxBudget: 15,
            fuelType: 'any',
            transmission: 'any',
            sellerType: 'any',
            maxKms: 50000
        };
        
        this.carMap = new Map();
        this.reverseCarMap = new Map();
        this.fuelTypeMap = new Map();
        this.transmissionMap = new Map();
        this.sellerTypeMap = new Map();
        this.brandMap = new Map();
        
        this.model = null;
        
        this.config = {
            embeddingDim: 32,
            batchSize: 128,
            epochs: 15,
            learningRate: 0.001
        };
        
        this.lossHistory = [];
        this.isTraining = false;
        
        this.initializeUI();
    }
    
    initializeUI() {
        document.getElementById('loadData').addEventListener('click', () => this.loadData());
        document.getElementById('train').addEventListener('click', () => this.train());
        document.getElementById('findCars').addEventListener('click', () => this.findCars());
        
        // Setup preference listeners
        document.getElementById('maxBudget').addEventListener('input', (e) => {
            this.userPreferences.maxBudget = parseFloat(e.target.value);
        });
        document.getElementById('fuelType').addEventListener('change', (e) => {
            this.userPreferences.fuelType = e.target.value;
        });
        document.getElementById('transmission').addEventListener('change', (e) => {
            this.userPreferences.transmission = e.target.value;
        });
        document.getElementById('sellerType').addEventListener('change', (e) => {
            this.userPreferences.sellerType = e.target.value;
        });
        document.getElementById('maxKms').addEventListener('input', (e) => {
            this.userPreferences.maxKms = parseInt(e.target.value);
        });
        
        this.updateStatus('Ready to load car data. Click "Load Car Data" to begin.');
    }
    
    async loadData() {
        const loadBtn = document.getElementById('loadData');
        const loadingSpinner = loadBtn.querySelector('.loading');
        
        loadBtn.disabled = true;
        loadingSpinner.style.display = 'inline-block';
        loadBtn.innerHTML = loadingSpinner.outerHTML + ' Loading...';
        
        this.updateStatus('📥 Loading car data...');
        
        try {
            // Use embedded car data instead of loading from CSV
            this.cars = this.getCarData();
            this.createMappings();
            
            this.updateStatus(`✅ Successfully loaded ${this.cars.length} cars. Found ${this.brandMap.size} brands, ${this.fuelTypeMap.size} fuel types.`);
            
            document.getElementById('train').disabled = false;
            document.getElementById('findCars').disabled = false;
            
        } catch (error) {
            this.updateStatus(`❌ Error loading data: ${error.message}`);
        } finally {
            loadBtn.disabled = false;
            loadingSpinner.style.display = 'none';
            loadBtn.textContent = 'Load Car Data';
        }
    }
    
    getCarData() {
        // Embedded car data from the CSV
        const carData = [
            {carName: "ritz", year: 2014, sellingPrice: 3.35, presentPrice: 5.59, kmsDriven: 27000, fuelType: "Petrol", sellerType: "Dealer", transmission: "Manual", owner: 0},
            {carName: "sx4", year: 2013, sellingPrice: 4.75, presentPrice: 9.54, kmsDriven: 43000, fuelType: "Diesel", sellerType: "Dealer", transmission: "Manual", owner: 0},
            {carName: "ciaz", year: 2017, sellingPrice: 7.25, presentPrice: 9.85, kmsDriven: 6900, fuelType: "Petrol", sellerType: "Dealer", transmission: "Manual", owner: 0},
            {carName: "wagon r", year: 2011, sellingPrice: 2.85, presentPrice: 4.15, kmsDriven: 5200, fuelType: "Petrol", sellerType: "Dealer", transmission: "Manual", owner: 0},
            {carName: "swift", year: 2014, sellingPrice: 4.6, presentPrice: 6.87, kmsDriven: 42450, fuelType: "Diesel", sellerType: "Dealer", transmission: "Manual", owner: 0},
            {carName: "vitara brezza", year: 2018, sellingPrice: 9.25, presentPrice: 9.83, kmsDriven: 2071, fuelType: "Diesel", sellerType: "Dealer", transmission: "Manual", owner: 0},
            {carName: "ciaz", year: 2015, sellingPrice: 6.75, presentPrice: 8.12, kmsDriven: 18796, fuelType: "Petrol", sellerType: "Dealer", transmission: "Manual", owner: 0},
            {carName: "s cross", year: 2015, sellingPrice: 6.5, presentPrice: 8.61, kmsDriven: 33429, fuelType: "Diesel", sellerType: "Dealer", transmission: "Manual", owner: 0},
            {carName: "ciaz", year: 2016, sellingPrice: 8.75, presentPrice: 8.89, kmsDriven: 20273, fuelType: "Diesel", sellerType: "Dealer", transmission: "Manual", owner: 0},
            {carName: "ciaz", year: 2015, sellingPrice: 7.45, presentPrice: 8.92, kmsDriven: 42367, fuelType: "Diesel", sellerType: "Dealer", transmission: "Manual", owner: 0},
            {carName: "alto 800", year: 2017, sellingPrice: 2.85, presentPrice: 3.6, kmsDriven: 2135, fuelType: "Petrol", sellerType: "Dealer", transmission: "Manual", owner: 0},
            {carName: "ciaz", year: 2015, sellingPrice: 6.85, presentPrice: 10.38, kmsDriven: 51000, fuelType: "Diesel", sellerType: "Dealer", transmission: "Manual", owner: 0},
            {carName: "ciaz", year: 2015, sellingPrice: 7.5, presentPrice: 9.94, kmsDriven: 15000, fuelType: "Petrol", sellerType: "Dealer", transmission: "Automatic", owner: 0},
            {carName: "ertiga", year: 2015, sellingPrice: 6.1, presentPrice: 7.71, kmsDriven: 26000, fuelType: "Petrol", sellerType: "Dealer", transmission: "Manual", owner: 0},
            {carName: "dzire", year: 2009, sellingPrice: 2.25, presentPrice: 7.21, kmsDriven: 77427, fuelType: "Petrol", sellerType: "Dealer", transmission: "Manual", owner: 0},
            {carName: "ertiga", year: 2016, sellingPrice: 7.75, presentPrice: 10.79, kmsDriven: 43000, fuelType: "Diesel", sellerType: "Dealer", transmission: "Manual", owner: 0},
            {carName: "ertiga", year: 2015, sellingPrice: 7.25, presentPrice: 10.79, kmsDriven: 41678, fuelType: "Diesel", sellerType: "Dealer", transmission: "Manual", owner: 0},
            {carName: "wagon r", year: 2015, sellingPrice: 3.25, presentPrice: 5.09, kmsDriven: 35500, fuelType: "CNG", sellerType: "Dealer", transmission: "Manual", owner: 0},
            {carName: "sx4", year: 2010, sellingPrice: 2.65, presentPrice: 7.98, kmsDriven: 41442, fuelType: "Petrol", sellerType: "Dealer", transmission: "Manual", owner: 0},
            {carName: "alto k10", year: 2016, sellingPrice: 2.85, presentPrice: 3.95, kmsDriven: 25000, fuelType: "Petrol", sellerType: "Dealer", transmission: "Manual", owner: 0}
        ];
        
        // Add brand and age information
        return carData.map(car => ({
            ...car,
            brand: this.extractBrand(car.carName),
            age: new Date().getFullYear() - car.year
        }));
    }
    
    extractBrand(carName) {
        const brands = ['Maruti', 'Hyundai', 'Honda', 'Toyota', 'Ford', 'BMW', 'Mercedes', 'Audi', 'Mahindra', 'Tata', 'Renault', 'Volkswagen', 'Skoda', 'Kia', 'MG', 'Jeep', 'Volvo', 'Jaguar', 'Land Rover', 'Porsche', 'Ferrari', 'Lamborghini', 'Mitsubishi', 'Nissan', 'Chevrolet', 'Fiat', 'Force', 'Isuzu', 'Mini'];
        
        for (let brand of brands) {
            if (carName.toLowerCase().includes(brand.toLowerCase())) {
                return brand;
            }
        }
        
        // Extract first word as brand for unknown brands
        return carName.split(' ')[0];
    }
    
    createMappings() {
        // Create mappings for categorical features
        const fuelTypes = [...new Set(this.cars.map(car => car.fuelType))];
        const transmissions = [...new Set(this.cars.map(car => car.transmission))];
        const sellerTypes = [...new Set(this.cars.map(car => car.sellerType))];
        const brands = [...new Set(this.cars.map(car => car.brand))];
        
        fuelTypes.forEach((type, index) => this.fuelTypeMap.set(type, index));
        transmissions.forEach((type, index) => this.transmissionMap.set(type, index));
        sellerTypes.forEach((type, index) => this.sellerTypeMap.set(type, index));
        brands.forEach((brand, index) => this.brandMap.set(brand, index));
        
        // Create car mappings
        this.cars.forEach((car, index) => {
            this.carMap.set(index, car);
            this.reverseCarMap.set(car, index);
        });
    }
    
    async train() {
        if (this.isTraining) return;
        
        this.isTraining = true;
        document.getElementById('train').disabled = true;
        this.lossHistory = [];
        
        this.updateStatus('🔄 Initializing Car Recommendation Model...');
        
        // Initialize model
        this.model = new CarFinderModel(
            this.brandMap.size,
            this.fuelTypeMap.size,
            this.transmissionMap.size,
            this.sellerTypeMap.size,
            this.config.embeddingDim
        );
        
        // Prepare training data
        const carIndices = Array.from({length: this.cars.length}, (_, i) => i);
        const brandIndices = this.cars.map(car => this.brandMap.get(car.brand));
        const fuelIndices = this.cars.map(car => this.fuelTypeMap.get(car.fuelType));
        const transmissionIndices = this.cars.map(car => this.transmissionMap.get(car.transmission));
        const sellerIndices = this.cars.map(car => this.sellerTypeMap.get(car.sellerType));
        
        this.updateStatus('🚀 Starting training with car features...');
        
        // Training loop
        const numBatches = Math.ceil(carIndices.length / this.config.batchSize);
        
        for (let epoch = 0; epoch < this.config.epochs; epoch++) {
            let epochLoss = 0;
            
            for (let batch = 0; batch < numBatches; batch++) {
                const start = batch * this.config.batchSize;
                const end = Math.min(start + this.config.batchSize, carIndices.length);
                
                const batchCars = carIndices.slice(start, end);
                const batchBrands = brandIndices.slice(start, end);
                const batchFuels = fuelIndices.slice(start, end);
                const batchTransmissions = transmissionIndices.slice(start, end);
                const batchSellers = sellerIndices.slice(start, end);
                
                const loss = await this.model.trainStep(
                    batchCars, batchBrands, batchFuels, batchTransmissions, batchSellers
                );
                epochLoss += loss;
                
                this.lossHistory.push(loss);
                this.updateLossChart();
                
                if (batch % 5 === 0) {
                    this.updateStatus(`📚 Epoch ${epoch + 1}/${this.config.epochs}, Batch ${batch}/${numBatches}, Loss: ${loss.toFixed(4)}`);
                }
                
                // Allow UI to update
                await new Promise(resolve => setTimeout(resolve, 0));
            }
            
            epochLoss /= numBatches;
            this.updateStatus(`🎉 Epoch ${epoch + 1}/${this.config.epochs} completed. Average loss: ${epochLoss.toFixed(4)}`);
        }
        
        this.isTraining = false;
        document.getElementById('train').disabled = false;
        
        this.updateStatus('🏆 Training completed! Click "Find Cars" to get recommendations.');
        
        // Visualize embeddings
        this.visualizeEmbeddings();
    }
    
    updateLossChart() {
        const canvas = document.getElementById('lossChart');
        const ctx = canvas.getContext('2d');
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        if (this.lossHistory.length === 0) return;
        
        // Create gradient background
        const gradient = ctx.createLinearGradient(0, 0, 0, canvas.height);
        gradient.addColorStop(0, 'rgba(110, 202, 220, 0.1)');
        gradient.addColorStop(1, 'rgba(164, 215, 225, 0.05)');
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        const maxLoss = Math.max(...this.lossHistory);
        const minLoss = Math.min(...this.lossHistory);
        const range = maxLoss - minLoss || 1;
        
        // Draw smoothed line
        ctx.strokeStyle = '#6ecadc';
        ctx.lineWidth = 3;
        ctx.beginPath();
        
        this.lossHistory.forEach((loss, index) => {
            const x = (index / this.lossHistory.length) * canvas.width;
            const y = canvas.height - ((loss - minLoss) / range) * canvas.height;
            
            if (index === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });
        
        ctx.stroke();
        
        // Add labels
        ctx.fillStyle = '#2c3e50';
        ctx.font = '12px Segoe UI';
        ctx.fillText(`Min: ${minLoss.toFixed(4)}`, 10, canvas.height - 10);
        ctx.fillText(`Max: ${maxLoss.toFixed(4)}`, 10, 20);
        ctx.fillText(`Current: ${this.lossHistory[this.lossHistory.length - 1].toFixed(4)}`, canvas.width - 100, 20);
    }
    
    async visualizeEmbeddings() {
        if (!this.model) return;
        
        this.updateStatus('🎨 Computing PCA projection for car embeddings...');
        
        const canvas = document.getElementById('embeddingChart');
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        try {
            // Get embeddings and compute PCA
            const embeddingsTensor = this.model.getCarEmbeddings();
            const embeddings = await embeddingsTensor.array();
            
            const projected = this.computePCA(embeddings, 2);
            
            // Normalize to canvas coordinates
            const xs = projected.map(p => p[0]);
            const ys = projected.map(p => p[1]);
            
            const xMin = Math.min(...xs);
            const xMax = Math.max(...xs);
            const yMin = Math.min(...ys);
            const yMax = Math.max(...ys);
            
            const xRange = xMax - xMin || 1;
            const yRange = yMax - yMin || 1;
            
            // Draw background gradient
            const gradient = ctx.createRadialGradient(
                canvas.width / 2, canvas.height / 2, 0,
                canvas.width / 2, canvas.height / 2, Math.max(canvas.width, canvas.height) / 2
            );
            gradient.addColorStop(0, 'rgba(164, 215, 225, 0.1)');
            gradient.addColorStop(1, 'rgba(110, 202, 220, 0.05)');
            ctx.fillStyle = gradient;
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            // Color by price range
            const priceRanges = [
                { max: 5, color: '#6ecadc' },    // Low price
                { max: 15, color: '#4bb5c3' },   // Medium price
                { max: 25, color: '#3498db' },   // High price
                { max: Infinity, color: '#2980b9' } // Luxury
            ];
            
            this.cars.forEach((car, i) => {
                const priceRange = priceRanges.find(range => car.presentPrice <= range.max);
                const color = priceRange ? priceRange.color : '#2980b9';
                
                const x = ((projected[i][0] - xMin) / xRange) * (canvas.width - 60) + 30;
                const y = ((projected[i][1] - yMin) / yRange) * (canvas.height - 60) + 30;
                
                const pointGradient = ctx.createRadialGradient(x, y, 0, x, y, 6);
                pointGradient.addColorStop(0, color + 'CC');
                pointGradient.addColorStop(1, color + '66');
                
                ctx.fillStyle = pointGradient;
                ctx.beginPath();
                ctx.arc(x, y, 4, 0, 2 * Math.PI);
                ctx.fill();
            });
            
            // Add title and labels
            ctx.fillStyle = '#2c3e50';
            ctx.font = '16px Segoe UI';
            ctx.fillText('Car Embeddings Projection (PCA) - Colored by Price', 20, 30);
            ctx.font = '12px Segoe UI';
            ctx.fillText(`Visualizing ${this.cars.length} cars - similar cars cluster together`, 20, 50);
            
            // Add legend
            ctx.fillStyle = '#6ecadc';
            ctx.fillRect(20, canvas.height - 80, 10, 10);
            ctx.fillStyle = '#2c3e50';
            ctx.fillText('Budget (<5L)', 35, canvas.height - 70);
            
            ctx.fillStyle = '#4bb5c3';
            ctx.fillRect(120, canvas.height - 80, 10, 10);
            ctx.fillStyle = '#2c3e50';
            ctx.fillText('Mid-range (5-15L)', 135, canvas.height - 70);
            
            ctx.fillStyle = '#3498db';
            ctx.fillRect(240, canvas.height - 80, 10, 10);
            ctx.fillStyle = '#2c3e50';
            ctx.fillText('Premium (15-25L)', 255, canvas.height - 70);
            
            ctx.fillStyle = '#2980b9';
            ctx.fillRect(360, canvas.height - 80, 10, 10);
            ctx.fillStyle = '#2c3e50';
            ctx.fillText('Luxury (>25L)', 375, canvas.height - 70);
            
            this.updateStatus('✅ Embedding visualization completed.');
        } catch (error) {
            this.updateStatus(`❌ Error in visualization: ${error.message}`);
        }
    }
    
    computePCA(embeddings, dimensions) {
        const n = embeddings.length;
        const dim = embeddings[0].length;
        
        // Center the data
        const mean = Array(dim).fill(0);
        embeddings.forEach(emb => {
            emb.forEach((val, i) => mean[i] += val);
        });
        mean.forEach((val, i) => mean[i] = val / n);
        
        const centered = embeddings.map(emb => 
            emb.map((val, i) => val - mean[i])
        );
        
        // Compute covariance matrix
        const covariance = Array(dim).fill(0).map(() => Array(dim).fill(0));
        centered.forEach(emb => {
            for (let i = 0; i < dim; i++) {
                for (let j = 0; j < dim; j++) {
                    covariance[i][j] += emb[i] * emb[j];
                }
            }
        });
        covariance.forEach(row => row.forEach((val, j) => row[j] = val / n));
        
        // Power iteration for first two components
        const components = [];
        for (let d = 0; d < dimensions; d++) {
            let vector = Array(dim).fill(1/Math.sqrt(dim));
            
            for (let iter = 0; iter < 10; iter++) {
                let newVector = Array(dim).fill(0);
                
                for (let i = 0; i < dim; i++) {
                    for (let j = 0; j < dim; j++) {
                        newVector[i] += covariance[i][j] * vector[j];
                    }
                }
                
                const norm = Math.sqrt(newVector.reduce((sum, val) => sum + val * val, 0));
                vector = newVector.map(val => val / norm);
            }
            
            components.push(vector);
            
            // Deflate the covariance matrix
            for (let i = 0; i < dim; i++) {
                for (let j = 0; j < dim; j++) {
                    covariance[i][j] -= vector[i] * vector[j];
                }
            }
        }
        
        // Project data
        return embeddings.map(emb => {
            return components.map(comp => 
                emb.reduce((sum, val, i) => sum + val * comp[i], 0)
            );
        });
    }
    
    async findCars() {
        if (!this.model) {
            this.updateStatus('❌ Model not trained yet.');
            return;
        }
        
        this.updateStatus('🔍 Finding cars matching your preferences...');
        
        try {
            // Create user preference vector
            const userVector = this.createUserVector();
            
            // Get scores for all cars
            const allCarScores = await this.model.getScoresForAllCars(userVector);
            
            // Filter and rank cars based on preferences
            const candidateCars = [];
            
            allCarScores.forEach((score, carIndex) => {
                const car = this.cars[carIndex];
                
                // Apply filters
                if (car.presentPrice > this.userPreferences.maxBudget) return;
                if (this.userPreferences.fuelType !== 'any' && car.fuelType !== this.userPreferences.fuelType) return;
                if (this.userPreferences.transmission !== 'any' && car.transmission !== this.userPreferences.transmission) return;
                if (this.userPreferences.sellerType !== 'any' && car.sellerType !== this.userPreferences.sellerType) return;
                if (car.kmsDriven > this.userPreferences.maxKms) return;
                
                candidateCars.push({ 
                    car, 
                    score,
                    valueScore: this.calculateValueScore(car)
                });
            });
            
            // Sort by combined score (model score + value score)
            candidateCars.sort((a, b) => {
                const scoreA = a.score + a.valueScore;
                const scoreB = b.score + b.valueScore;
                return scoreB - scoreA;
            });
            
            const topRecommendations = candidateCars.slice(0, 10);
            
            // Display results
            this.displayResults(topRecommendations);
            
        } catch (error) {
            this.updateStatus(`❌ Error finding cars: ${error.message}`);
        }
    }
    
    createUserVector() {
        // Create a synthetic user vector based on preferences
        const vector = Array(this.config.embeddingDim).fill(0);
        
        // Bias towards preferred features
        if (this.userPreferences.fuelType !== 'any') {
            vector[0] = 1.0; // Fuel type importance
        }
        if (this.userPreferences.transmission !== 'any') {
            vector[1] = 0.8; // Transmission importance
        }
        
        // Budget preference (lower budget = higher value in lower dimensions)
        const budgetRatio = Math.max(0, 1 - (this.userPreferences.maxBudget / 50));
        vector[2] = budgetRatio;
        
        // Kilometer preference (lower kms = higher value)
        const kmRatio = Math.max(0, 1 - (this.userPreferences.maxKms / 200000));
        vector[3] = kmRatio;
        
        return vector;
    }
    
    calculateValueScore(car) {
        // Calculate value score based on price, age, and kilometers
        const priceScore = Math.max(0, 1 - (car.presentPrice / this.userPreferences.maxBudget));
        const ageScore = Math.max(0, 1 - (car.age / 20));
        const kmScore = Math.max(0, 1 - (car.kmsDriven / this.userPreferences.maxKms));
        
        return (priceScore * 0.5 + ageScore * 0.3 + kmScore * 0.2) * 2;
    }
    
    displayResults(recommendations) {
        const resultsDiv = document.getElementById('results');
        
        let html = `
            <h2 style="color: #6ecadc; margin-bottom: 20px;">🚗 Top Car Recommendations</h2>
            <div style="margin-bottom: 20px; padding: 15px; background: linear-gradient(135deg, #e8f4f8, #d4edf2); border-radius: 10px; border-left: 4px solid #6ecadc;">
                <strong>🎯 Your Preferences:</strong> 
                Max Budget: ₹${this.userPreferences.maxBudget}L | 
                Fuel: ${this.userPreferences.fuelType} | 
                Transmission: ${this.userPreferences.transmission} | 
                Seller: ${this.userPreferences.sellerType} | 
                Max Kms: ${this.userPreferences.maxKms.toLocaleString()}
            </div>
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Car</th>
                        <th>Brand</th>
                        <th>Year</th>
                        <th>Price (L)</th>
                        <th>Kms</th>
                        <th>Fuel</th>
                        <th>Transmission</th>
                        <th>Match Score</th>
                    </tr>
                </thead>
                <tbody>
        `;
        
        recommendations.forEach((rec, index) => {
            const car = rec.car;
            const totalScore = Math.min(100, (rec.score + rec.valueScore) * 25); // Normalize to 0-100
            
            html += `
                <tr>
                    <td><strong>${index + 1}</strong></td>
                    <td>${car.carName}</td>
                    <td>${car.brand}</td>
                    <td>${car.year}</td>
                    <td style="color: ${car.presentPrice <= this.userPreferences.maxBudget ? '#27ae60' : '#e74c3c'}">
                        ₹${car.presentPrice.toFixed(2)}L
                    </td>
                    <td>${car.kmsDriven.toLocaleString()}</td>
                    <td>${car.fuelType}</td>
                    <td>${car.transmission}</td>
                    <td>
                        <div style="background: #ecf0f1; border-radius: 10px; height: 8px; margin: 5px 0;">
                            <div style="background: linear-gradient(90deg, #6ecadc, #4bb5c3); width: ${totalScore}%; height: 100%; border-radius: 10px;"></div>
                        </div>
                        ${totalScore.toFixed(1)}%
                    </td>
                </tr>
            `;
        });
        
        html += `
                </tbody>
            </table>
            <div style="margin-top: 20px; padding: 15px; background: linear-gradient(135deg, #ffeaa7, #fdcb6e); border-radius: 10px; border-left: 4px solid #f39c12;">
                <strong>💡 Pro Tip:</strong> Consider test driving the top recommendations and comparing insurance costs before making a decision.
            </div>
        `;
        
        resultsDiv.innerHTML = html;
        this.updateStatus(`✅ Found ${recommendations.length} cars matching your preferences!`);
    }
    
    updateStatus(message) {
        document.getElementById('status').textContent = message;
    }
}

// Initialize app when page loads
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new CarFinderApp();
});
