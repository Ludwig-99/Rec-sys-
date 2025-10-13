let model;
let carData = [];
let carEmbeddings = {};
let isModelTrained = false;

// Load and parse CSV data
async function loadData() {
    const status = document.getElementById('status');
    status.textContent = 'Loading car data...';
    
    try {
        const response = await fetch('car data.csv');
        const csvText = await response.text();
        
        // Parse CSV
        const lines = csvText.split('\n').slice(1); // Skip header
        carData = lines.filter(line => line.trim()).map(line => {
            const [Car_Name, Year, Selling_Price, Present_Price, Kms_Driven, Fuel_Type, Seller_Type, Transmission, Owner] = line.split(',');
            
            return {
                Car_Name: Car_Name.trim(),
                Year: parseInt(Year),
                Selling_Price: parseFloat(Selling_Price),
                Present_Price: parseFloat(Present_Price),
                Kms_Driven: parseInt(Kms_Driven),
                Fuel_Type: Fuel_Type.trim(),
                Seller_Type: Seller_Type.trim(),
                Transmission: Transmission.trim(),
                Owner: parseInt(Owner)
            };
        }).filter(car => car.Car_Name); // Remove empty entries

        populateCarSelect();
        status.textContent = `Loaded ${carData.length} cars`;
        
    } catch (error) {
        status.textContent = 'Error loading data: ' + error.message;
    }
}

function populateCarSelect() {
    const select = document.getElementById('carSelect');
    select.innerHTML = '<option value="">Select a car...</option>';
    
    carData.forEach((car, index) => {
        const option = document.createElement('option');
        option.value = index;
        option.textContent = `${car.Car_Name} (${car.Year}) - $${car.Selling_Price}`;
        select.appendChild(option);
    });
}

// Two Tower Model Architecture
function createTwoTowerModel(inputDim, embeddingDim = 32) {
    // User tower - represents user preferences
    const userInput = tf.input({shape: [inputDim], name: 'user_input'});
    const userTower = tf.layers.dense({units: 64, activation: 'relu'}).apply(userInput);
    const userTower2 = tf.layers.dense({units: embeddingDim, activation: 'linear'}).apply(userTower);
    const userEmbedding = tf.layers.l2Normalize().apply(userTower2);
    
    // Item tower - represents car features
    const itemInput = tf.input({shape: [inputDim], name: 'item_input'});
    const itemTower = tf.layers.dense({units: 64, activation: 'relu'}).apply(itemInput);
    const itemTower2 = tf.layers.dense({units: embeddingDim, activation: 'linear'}).apply(itemTower);
    const itemEmbedding = tf.layers.l2Normalize().apply(itemTower2);
    
    // Dot product for similarity
    const dotProduct = tf.layers.dot({axes: -1, normalize: true}).apply([userEmbedding, itemEmbedding]);
    
    const model = tf.model({
        inputs: [userInput, itemInput],
        outputs: dotProduct,
        name: 'TwoTowerRecommender'
    });
    
    model.compile({
        optimizer: 'adam',
        loss: 'meanSquaredError'
    });
    
    return model;
}

// Preprocess car features
function preprocessFeatures(car) {
    // One-hot encoding for categorical features
    const fuelTypes = ['Petrol', 'Diesel', 'CNG'];
    const sellerTypes = ['Dealer', 'Individual'];
    const transmissions = ['Manual', 'Automatic'];
    
    const features = [
        // Normalized numerical features
        (car.Year - 2000) / 20, // Normalize year
        car.Selling_Price / 40, // Normalize selling price
        car.Present_Price / 100, // Normalize present price
        car.Kms_Driven / 250000, // Normalize kilometers
        car.Owner / 3, // Normalize owner count
        
        // One-hot encoded categorical features
        ...fuelTypes.map(ft => ft === car.Fuel_Type ? 1 : 0),
        ...sellerTypes.map(st => st === car.Seller_Type ? 1 : 0),
        ...transmissions.map(t => t === car.Transmission ? 1 : 0)
    ];
    
    return tf.tensor1d(features);
}

// Train the model
async function trainModel() {
    const status = document.getElementById('status');
    
    if (carData.length === 0) {
        status.textContent = 'Please load data first';
        return;
    }
    
    status.textContent = 'Training model...';
    
    const inputDim = 5 + 3 + 2 + 2; // Numerical + fuelTypes + sellerTypes + transmissions
    
    if (!model) {
        model = createTwoTowerModel(inputDim);
    }
    
    // Create training data (positive pairs: similar cars)
    const userInputs = [];
    const itemInputs = [];
    const targets = [];
    
    for (let i = 0; i < carData.length; i++) {
        const userFeatures = preprocessFeatures(carData[i]);
        
        // Find similar cars based on price and type
        for (let j = 0; j < carData.length; j++) {
            if (i !== j) {
                const itemFeatures = preprocessFeatures(carData[j]);
                
                // Calculate similarity score
                const priceDiff = Math.abs(carData[i].Selling_Price - carData[j].Selling_Price);
                const yearDiff = Math.abs(carData[i].Year - carData[j].Year);
                const fuelMatch = carData[i].Fuel_Type === carData[j].Fuel_Type ? 1 : 0;
                
                const similarity = Math.exp(-(priceDiff / 10 + yearDiff / 5)) * (fuelMatch + 1) / 2;
                
                userInputs.push(userFeatures);
                itemInputs.push(itemFeatures);
                targets.push(similarity);
            }
        }
    }
    
    const userTensor = tf.stack(userInputs);
    const itemTensor = tf.stack(itemInputs);
    const targetTensor = tf.tensor1d(targets);
    
    // Train the model
    await model.fit([userTensor, itemTensor], targetTensor, {
        epochs: 50,
        batchSize: 32,
        validationSplit: 0.2,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                status.textContent = `Training... Epoch ${epoch + 1}/50, Loss: ${logs.loss.toFixed(4)}`;
            }
        }
    });
    
    // Precompute embeddings for all cars
    status.textContent = 'Computing embeddings...';
    for (let i = 0; i < carData.length; i++) {
        const features = preprocessFeatures(carData[i]);
        const embedding = model.layers[6].apply(features.reshape([1, -1])); // Get item embedding
        carEmbeddings[i] = embedding;
    }
    
    isModelTrained = true;
    status.textContent = 'Model trained successfully!';
    
    // Clean up
    userTensor.dispose();
    itemTensor.dispose();
    targetTensor.dispose();
    tf.dispose(userInputs);
    tf.dispose(itemInputs);
}

// Get recommendations
async function getRecommendations() {
    const status = document.getElementById('status');
    const results = document.getElementById('results');
    const carSelect = document.getElementById('carSelect');
    
    if (!isModelTrained) {
        status.textContent = 'Please train the model first';
        return;
    }
    
    const selectedIndex = carSelect.value;
    if (!selectedIndex) {
        status.textContent = 'Please select a car';
        return;
    }
    
    status.textContent = 'Finding recommendations...';
    results.innerHTML = '';
    
    const selectedCar = carData[selectedIndex];
    const userFeatures = preprocessFeatures(selectedCar);
    const userEmbedding = model.layers[3].apply(userFeatures.reshape([1, -1])); // Get user embedding
    
    // Calculate similarities with all cars
    const similarities = [];
    for (let i = 0; i < carData.length; i++) {
        if (i != selectedIndex) {
            const similarity = tf.dot(userEmbedding, carEmbeddings[i]).dataSync()[0];
            similarities.push({index: i, similarity: similarity});
        }
    }
    
    // Sort by similarity (descending)
    similarities.sort((a, b) => b.similarity - a.similarity);
    
    // Display top 5 recommendations
    status.textContent = `Found ${similarities.length} recommendations`;
    
    const topRecommendations = similarities.slice(0, 5);
    
    results.innerHTML = `
        <h3>Because you're interested in: ${selectedCar.Car_Name} (${selectedCar.Year})</h3>
        <p>Price: $${selectedCar.Selling_Price} | ${selectedCar.Fuel_Type} | ${selectedCar.Transmission}</p>
        <h4>Recommended Cars:</h4>
    `;
    
    topRecommendations.forEach(rec => {
        const car = carData[rec.index];
        const carCard = document.createElement('div');
        carCard.className = 'car-card';
        carCard.innerHTML = `
            <strong>${car.Car_Name} (${car.Year})</strong><br>
            Price: $${car.Selling_Price} | Present: $${car.Present_Price}<br>
            ${car.Kms_Driven} km | ${car.Fuel_Type} | ${car.Transmission}<br>
            Seller: ${car.Seller_Type} | Previous Owners: ${car.Owner}<br>
            <small>Similarity: ${rec.similarity.toFixed(3)}</small>
        `;
        results.appendChild(carCard);
    });
    
    // Clean up
    userFeatures.dispose();
    userEmbedding.dispose();
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', loadData);
