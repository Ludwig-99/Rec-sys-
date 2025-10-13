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
        
        // Setup preference listeners with immediate feedback
        document.getElementById('maxBudget').addEventListener('input', (e) => {
            this.userPreferences.maxBudget = parseFloat(e.target.value);
            const progress = (this.userPreferences.maxBudget / 100) * 100;
            e.target.nextElementSibling.querySelector('.progress-fill').style.width = `${progress}%`;
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
            const progress = (this.userPreferences.maxKms / 200000) * 100;
            e.target.nextElementSibling.querySelector('.progress-fill').style.width = `${progress}%`;
        });
        
        this.updateStatus('✅ Ready to load car data. Click "Load Car Data" to begin.');
    }
    
    async loadData() {
        const loadBtn = document.getElementById('loadData');
        const loadingSpinner = loadBtn.querySelector('.loading');
        
        loadBtn.disabled = true;
        loadingSpinner.style.display = 'inline-block';
        loadBtn.innerHTML = loadingSpinner.outerHTML + ' Loading Car Database...';
        
        this.updateStatus('📥 Loading car database...');
        
        try {
            // Simulate loading delay for better UX
            await new Promise(resolve => setTimeout(resolve, 1500));
            
            // Use embedded car data
            this.cars = this.getCarData();
            this.createMappings();
            
            this.updateStatus(`✅ Successfully loaded ${this.cars.length} premium vehicles. Found ${this.brandMap.size} brands across ${this.fuelTypeMap.size} fuel types.`);
            
            document.getElementById('train').disabled = false;
            document.getElementById('findCars').disabled = false;
            
            // Update button states
            loadBtn.innerHTML = '<span class="icon">✅</span>Data Loaded!';
            
        } catch (error) {
            this.updateStatus(`❌ Error loading data: ${error.message}`);
            console.error('Data loading error:', error);
        } finally {
            setTimeout(() => {
                loadBtn.disabled = false;
                loadingSpinner.style.display = 'none';
                loadBtn.innerHTML = '<span class="icon">📥</span>Load Car Data';
            }, 2000);
        }
    }
    
    getCarData() {
        // Comprehensive car dataset
        const carData = [
            // Maruti Suzuki Cars
            {carName: "Maruti Swift", year: 2020, sellingPrice: 6.5, presentPrice: 8.2, kmsDriven: 15000, fuelType: "Petrol", sellerType: "Dealer", transmission: "Manual", owner: 0},
            {carName: "Maruti Baleno", year: 2019, sellingPrice: 7.2, presentPrice: 9.1, kmsDriven: 22000, fuelType: "Petrol", sellerType: "Dealer", transmission: "Automatic", owner: 0},
            {carName: "Maruti Dzire", year: 2021, sellingPrice: 8.1, presentPrice: 10.5, kmsDriven: 8000, fuelType: "Petrol", sellerType: "Dealer", transmission: "Manual", owner: 0},
            {carName: "Maruti Vitara Brezza", year: 2020, sellingPrice: 9.5, presentPrice: 12.8, kmsDriven: 18000, fuelType: "Diesel", sellerType: "Dealer", transmission: "Automatic", owner: 0},
            {carName: "Maruti Ertiga", year: 2019, sellingPrice: 8.8, presentPrice: 11.2, kmsDriven: 25000, fuelType: "Petrol", sellerType: "Dealer", transmission: "Manual", owner: 0},
            
            // Hyundai Cars
            {carName: "Hyundai Creta", year: 2021, sellingPrice: 12.5, presentPrice: 16.8, kmsDriven: 12000, fuelType: "Petrol", sellerType: "Dealer", transmission: "Automatic", owner: 0},
            {carName: "Hyundai i20", year: 2020, sellingPrice: 7.8, presentPrice: 9.9, kmsDriven: 14000, fuelType: "Petrol", sellerType: "Dealer", transmission: "Manual", owner: 0},
            {carName: "Hyundai Verna", year: 2019, sellingPrice: 9.2, presentPrice: 12.1, kmsDriven: 21000, fuelType: "Diesel", sellerType: "Dealer", transmission: "Automatic", owner: 0},
            {carName: "Hyundai Venue", year: 2021, sellingPrice: 8.9, presentPrice: 11.5, kmsDriven: 9000, fuelType: "Petrol", sellerType: "Dealer", transmission: "Manual", owner: 0},
            
            // Honda Cars
            {carName: "Honda City", year: 2020, sellingPrice: 11.2, presentPrice: 14.8, kmsDriven: 16000, fuelType: "Petrol", sellerType: "Dealer", transmission: "Automatic", owner: 0},
            {carName: "Honda Amaze", year: 2019, sellingPrice: 6.8, presentPrice: 8.9, kmsDriven: 19000, fuelType: "Diesel", sellerType: "Dealer", transmission: "Manual", owner: 0},
            {carName: "Honda WR-V", year: 2018, sellingPrice: 7.5, presentPrice: 9.8, kmsDriven: 28000, fuelType: "Petrol", sellerType: "Dealer", transmission: "Manual", owner: 0},
            
            // Toyota Cars
            {carName: "Toyota Innova", year: 2019, sellingPrice: 18.5, presentPrice: 25.2, kmsDriven: 32000, fuelType: "Diesel", sellerType: "Dealer", transmission: "Automatic", owner: 0},
            {carName: "Toyota Fortuner", year: 2020, sellingPrice: 28.9, presentPrice: 36.5, kmsDriven: 15000, fuelType: "Diesel", sellerType: "Dealer", transmission: "Automatic", owner: 0},
            {carName: "Toyota Glanza", year: 2021, sellingPrice: 7.9, presentPrice: 10.1, kmsDriven: 7000, fuelType: "Petrol", sellerType: "Dealer", transmission: "Manual", owner: 0},
            
            // Tata Cars
            {carName: "Tata Nexon", year: 2020, sellingPrice: 8.2, presentPrice: 10.8, kmsDriven: 13000, fuelType: "Petrol", sellerType: "Dealer", transmission: "Manual", owner: 0},
            {carName: "Tata Harrier", year: 2021, sellingPrice: 14.8, presentPrice: 19.2, kmsDriven: 8000, fuelType: "Diesel", sellerType: "Dealer", transmission: "Automatic", owner: 0},
            {carName: "Tata Altroz", year: 2020, sellingPrice: 6.9, presentPrice: 8.7, kmsDriven: 11000, fuelType: "Petrol", sellerType: "Dealer", transmission: "Manual", owner: 0},
            
            // Kia Cars
            {carName: "Kia Seltos", year: 2021, sellingPrice: 12.1, presentPrice: 15.9, kmsDriven: 6000, fuelType: "Petrol", sellerType: "Dealer", transmission: "Automatic", owner: 0},
            {carName: "Kia Sonet", year: 2020, sellingPrice: 9.5, presentPrice: 12.3, kmsDriven: 10000, fuelType: "Diesel", sellerType: "Dealer", transmission: "Manual", owner: 0},
            
            // Individual Seller Cars
            {carName: "Maruti Wagon R", year: 2018, sellingPrice: 3.8, presentPrice: 5.2, kmsDriven: 35000, fuelType: "CNG", sellerType: "Individual", transmission: "Manual", owner: 1},
            {carName: "Hyundai i10", year: 2017, sellingPrice: 3.2, presentPrice: 4.5, kmsDriven: 42000, fuelType: "Petrol", sellerType: "Individual", transmission: "Manual", owner: 0},
            {carName: "Honda Jazz", year: 2016, sellingPrice: 4.8, presentPrice: 6.9, kmsDriven: 38000, fuelType: "Petrol", sellerType: "Individual", transmission: "Automatic", owner: 1}
        ];
        
        // Add brand and age information
        return carData.map(car => ({
            ...car,
            brand: this.extractBrand(car.carName),
            age: new Date().getFullYear() - car.year
        }));
    }
    
    extractBrand(carName) {
        const brands = ['Maruti', 'Hyundai', 'Honda', 'Toyota', 'Tata', 'Kia', 'Ford', 'Mahindra', 'Renault', 'Volkswagen'];
        
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
        const trainBtn = document.getElementById('train');
        trainBtn.disabled = true;
        trainBtn.innerHTML = '<span class="icon">⏳</span>Training AI...';
        
        this.lossHistory = [];
        this.updateStatus('🧠 Initializing AI model architecture...');
        
        try {
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
            
            this.updateStatus('🚀 Starting AI training with advanced feature learning...');
            
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
                    
                    if (batch % 3 === 0) {
                        this.updateStatus(`📚 Epoch ${epoch + 1}/${this.config.epochs} • Batch ${batch + 1}/${numBatches} • Loss: ${loss.toFixed(4)}`);
                    }
                    
                    // Allow UI to update
                    await new Promise(resolve => setTimeout(resolve, 50));
                }
                
                epochLoss /= numBatches;
                this.updateStatus(`🎉 Epoch ${epoch + 1}/${this.config.epochs} completed • Average Loss: ${epochLoss.toFixed(4)}`);
            }
            
            this.updateStatus('🏆 AI training completed! Ready to find your perfect car match.');
            trainBtn.innerHTML = '<span class="icon">✅</span>AI Trained!';
            
        } catch (error) {
            this.updateStatus(`❌ Training error: ${error.message}`);
            console.error('Training error:', error);
        } finally {
            this.isTraining = false;
            setTimeout(() => {
                trainBtn.disabled = false;
                trainBtn.innerHTML = '<span class="icon">🧠</span>Train AI Model';
            }, 2000);
        }
        
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
        
        // Add labels with better styling
        ctx.fillStyle = '#2c3e50';
        ctx.font = '14px -apple-system, BlinkMacSystemFont, sans-serif';
        ctx.fillText(`Min Loss: ${minLoss.toFixed(4)}`, 15, canvas.height - 15);
        ctx.fillText(`Max Loss: ${maxLoss.toFixed(4)}`, 15, 25);
        ctx.fillText(`Current: ${this.lossHistory[this.lossHistory.length - 1].toFixed(4)}`, canvas.width - 120, 25);
    }
    
    async visualizeEmbeddings() {
        if (!this.model) return;
        
        this.updateStatus('🎨 Generating car similarity visualization...');
        
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
            
            // Color by brand
            const brandColors = {
                'Maruti': '#6ecadc',
                'Hyundai': '#4bb5c3', 
                'Honda': '#3498db',
                'Toyota': '#2980b9',
                'Tata': '#27ae60',
                'Kia': '#9b59b6',
                'Ford': '#e74c3c',
                'Mahindra': '#e67e22',
                'Renault': '#f1c40f',
                'Volkswagen': '#1abc9c'
            };
            
            this.cars.forEach((car, i) => {
                const color = brandColors[car.brand] || '#95a5a6';
                
                const x = ((projected[i][0] - xMin) / xRange) * (canvas.width - 80) + 40;
                const y = ((projected[i][1] - yMin) / yRange) * (canvas.height - 80) + 40;
                
                const pointGradient = ctx.createRadialGradient(x, y, 0, x, y, 8);
                pointGradient.addColorStop(0, color + 'DD');
                pointGradient.addColorStop(1, color + '66');
                
                ctx.fillStyle = pointGradient;
                ctx.beginPath();
                ctx.arc(x, y, 6, 0, 2 * Math.PI);
                ctx.fill();
                
                // Add glow effect
                ctx.shadowColor = color + '80';
                ctx.shadowBlur = 10;
                ctx.fill();
                ctx.shadowBlur = 0;
            });
            
            // Add title and labels
            ctx.fillStyle = '#2c3e50';
            ctx.font = 'bold 18px -apple-system, BlinkMacSystemFont, sans-serif';
            ctx.fillText('Car Similarity Map - AI Clustering', 30, 35);
            ctx.font = '14px -apple-system, BlinkMacSystemFont, sans-serif';
            ctx.fillText(`Visualizing ${this.cars.length} vehicles - Similar cars cluster together`, 30, 60);
            
            this.updateStatus('✅ Car similarity visualization completed.');
        } catch (error) {
            this.updateStatus(`❌ Visualization error: ${error.message}`);
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
            this.updateStatus('❌ Please train the AI model first.');
            return;
        }
        
        const findBtn = document.getElementById('findCars');
        findBtn.disabled = true;
        findBtn.innerHTML = '<span class="icon">🔍</span>Finding Matches...';
        
        this.updateStatus('🔍 Analyzing your preferences and finding perfect car matches...');
        
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
            
            const topRecommendations = candidateCars.slice(0, 8);
            
            // Display results
            this.displayResults(topRecommendations);
            
            findBtn.innerHTML = '<span class="icon">✅</span>Matches Found!';
            
        } catch (error) {
            this.updateStatus(`❌ Error finding cars: ${error.message}`);
        } finally {
            setTimeout(() => {
                findBtn.disabled = false;
                findBtn.innerHTML = '<span class="icon">🔍</span>Find Perfect Cars';
            }, 2000);
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
            <h2 style="color: var(--primary); margin-bottom: 25px; font-size: 2em; font-weight: 400;">
                <span class="icon">🚗</span>Your Perfect Car Matches
            </h2>
            <div style="margin-bottom: 25px; padding: 20px; background: linear-gradient(135deg, rgba(232, 244, 248, 0.9), rgba(212, 237, 242, 0.9)); border-radius: 20px; border-left: 6px solid var(--primary);">
                <strong style="font-size: 16px;"><span class="icon">🎯</span> Your Preferences:</strong> 
                <div style="margin-top: 10px; display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; font-size: 14px;">
                    <div>💰 <strong>Budget:</strong> ₹${this.userPreferences.maxBudget}L</div>
                    <div>⛽ <strong>Fuel:</strong> ${this.userPreferences.fuelType}</div>
                    <div>⚙️ <strong>Transmission:</strong> ${this.userPreferences.transmission}</div>
                    <div>🏪 <strong>Seller:</strong> ${this.userPreferences.sellerType}</div>
                    <div>📊 <strong>Max Kms:</strong> ${this.userPreferences.maxKms.toLocaleString()}</div>
                </div>
            </div>
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Car Model</th>
                        <th>Brand</th>
                        <th>Year</th>
                        <th>Price (L)</th>
                        <th>Kms</th>
                        <th>Fuel</th>
                        <th>Trans</th>
                        <th>Match Score</th>
                    </tr>
                </thead>
                <tbody>
        `;
        
        recommendations.forEach((rec, index) => {
            const car = rec.car;
            const totalScore = Math.min(100, (rec.score + rec.valueScore) * 25);
            const scoreColor = totalScore >= 80 ? '#27ae60' : totalScore >= 60 ? '#f39c12' : '#e74c3c';
            
            html += `
                <tr>
                    <td style="font-weight: 600; color: var(--primary);">#${index + 1}</td>
                    <td style="font-weight: 500;">${car.carName}</td>
                    <td>${car.brand}</td>
                    <td>${car.year}</td>
                    <td style="color: ${car.presentPrice <= this.userPreferences.maxBudget ? '#27ae60' : '#e74c3c'}; font-weight: 500;">
                        ₹${car.presentPrice.toFixed(1)}L
                    </td>
                    <td>${car.kmsDriven.toLocaleString()}</td>
                    <td>${car.fuelType}</td>
                    <td>${car.transmission}</td>
                    <td>
                        <div style="background: #ecf0f1; border-radius: 10px; height: 8px; margin: 5px 0; position: relative;">
                            <div style="background: linear-gradient(90deg, #6ecadc, #4bb5c3); width: ${totalScore}%; height: 100%; border-radius: 10px;"></div>
                        </div>
                        <span style="color: ${scoreColor}; font-weight: 600;">${totalScore.toFixed(1)}%</span>
                    </td>
                </tr>
            `;
        });
        
        html += `
                </tbody>
            </table>
            <div style="margin-top: 25px; padding: 20px; background: linear-gradient(135deg, rgba(255, 234, 167, 0.9), rgba(253, 203, 110, 0.9)); border-radius: 20px; border-left: 6px solid #f39c12;">
                <strong style="font-size: 16px;"><span class="icon">💡</span> Pro Tip:</strong> 
                <span style="font-size: 14px;">Schedule test drives for your top 3 matches and compare insurance quotes before making your final decision.</span>
            </div>
        `;
        
        resultsDiv.innerHTML = html;
        this.updateStatus(`✅ Found ${recommendations.length} perfect car matches for you!`);
    }
    
    updateStatus(message) {
        document.getElementById('status').innerHTML = message;
    }
}

// Initialize app when page loads
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new CarFinderApp();
});
