// app.js
let interactions = [];
let cars = new Map();
let userToCars = new Map();
let carIdToIndex = new Map();
let userIdToIndex = new Map();
let indexToCarId = [];
let indexToUserId = [];
let model = null;
let carEmbeddingsMatrix = null;

const config = {
  maxInteractions: 5000,
  embeddingDim: 32,
  epochs: 30,
  batchSize: 64,
  learningRate: 0.01,
  sampleCarCount: 1000
};

const statusEl = document.getElementById('status');
const loadBtn = document.getElementById('loadBtn');
const trainBtn = document.getElementById('trainBtn');
const testBtn = document.getElementById('testBtn');
const lossCanvas = document.getElementById('lossChart');
const embedCanvas = document.getElementById('embeddingCanvas');
const resultsEl = document.getElementById('results');

const ctxLoss = lossCanvas.getContext('2d');
const ctxEmbed = embedCanvas.getContext('2d');

// Utility: parse CSV
function parseCSV(text) {
  const lines = text.trim().split('\n');
  const headers = lines[0].split(',');
  return lines.slice(1).map(line => {
    const values = line.split(',');
    const obj = {};
    headers.forEach((h, i) => obj[h.trim()] = values[i].trim());
    return obj;
  });
}

// Load data from /data/
async function loadData() {
  try {
    statusEl.textContent = 'Loading interactions.csv...';
    const interResp = await fetch('data/interactions.csv');
    const interText = await interResp.text();
    const interRows = parseCSV(interText);

    statusEl.textContent = 'Loading cars.csv...';
    const carsResp = await fetch('data/cars.csv');
    const carsText = await carsResp.text();
    const carRows = parseCSV(carsText);

    // Build cars map
    cars.clear();
    carRows.forEach(row => {
      cars.set(row.car_id, {
        make: row.make || 'Unknown',
        model: row.model || 'Unknown',
        year: row.year || '',
        bodyType: row.body_type || '',
        priceRange: row.price_range || ''
      });
    });

    // Build interactions
    interactions = interRows.slice(0, config.maxInteractions).map(r => ({
      userId: r.user_id,
      carId: r.car_id,
      rating: parseFloat(r.rating),
      ts: parseInt(r.timestamp)
    }));

    // Index users and cars
    const uniqueUsers = [...new Set(interactions.map(i => i.userId))];
    const uniqueCars = [...new Set(interactions.map(i => i.carId))];

    userIdToIndex.clear();
    carIdToIndex.clear();
    indexToUserId = uniqueUsers;
    indexToCarId = uniqueCars;

    uniqueUsers.forEach((id, idx) => userIdToIndex.set(id, idx));
    uniqueCars.forEach((id, idx) => carIdToIndex.set(id, idx));

    // Build user → rated cars
    userToCars.clear();
    interactions.forEach(i => {
      if (!userToCars.has(i.userId)) userToCars.set(i.userId, []);
      userToCars.get(i.userId).push({
        carId: i.carId,
        rating: i.rating,
        ts: i.ts
      });
    });

    // Sort by rating (desc) then timestamp (desc)
    for (let [userId, carsList] of userToCars.entries()) {
      carsList.sort((a, b) => b.rating - a.rating || b.ts - a.ts);
    }

    statusEl.textContent = `Loaded ${interactions.length} interactions, ${uniqueUsers.length} users, ${uniqueCars.length} cars.`;
    trainBtn.disabled = false;
  } catch (err) {
    statusEl.textContent = `Error loading data: ${err.message}`;
    console.error(err);
  }
}

// Draw loss curve
let lossHistory = [];
function drawLossChart(loss) {
  lossHistory.push(loss);
  ctxLoss.clearRect(0, 0, lossCanvas.width, lossCanvas.height);
  const w = lossCanvas.width;
  const h = lossCanvas.height;
  const maxLoss = Math.max(...lossHistory);
  const minLoss = Math.min(...lossHistory);
  const range = maxLoss - minLoss || 1;

  ctxLoss.beginPath();
  ctxLoss.moveTo(0, h);
  lossHistory.forEach((l, i) => {
    const x = (i / (lossHistory.length - 1 || 1)) * w;
    const y = h - ((l - minLoss) / range) * h;
    if (i === 0) ctxLoss.moveTo(x, y);
    else ctxLoss.lineTo(x, y);
  });
  ctxLoss.strokeStyle = '#1a5276';
  ctxLoss.lineWidth = 2;
  ctxLoss.stroke();
}

// Approximate PCA for 2D projection
function approximatePCA(embeddings, nComponents = 2) {
  const n = embeddings.shape[0];
  const d = embeddings.shape[1];
  const data = embeddings.dataSync();

  // Center data
  const mean = new Array(d).fill(0);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < d; j++) {
      mean[j] += data[i * d + j];
    }
  }
  for (let j = 0; j < d; j++) mean[j] /= n;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < d; j++) {
      data[i * d + j] -= mean[j];
    }
  }

  // Compute covariance (approx via random projection)
  const proj = new Array(nComponents).fill(null).map(() => 
    new Array(d).fill(null).map(() => Math.random() - 0.5)
  );

  const result = new Array(n).fill(null).map(() => new Array(nComponents).fill(0));
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < nComponents; k++) {
      for (let j = 0; j < d; j++) {
        result[i][k] += data[i * d + j] * proj[k][j];
      }
    }
  }

  return result;
}

// Draw embedding projection
async function drawEmbeddingProjection() {
  if (!carEmbeddingsMatrix) return;

  const sampleIndices = [];
  const total = carEmbeddingsMatrix.shape[0];
  const count = Math.min(config.sampleCarCount, total);
  while (sampleIndices.length < count) {
    const idx = Math.floor(Math.random() * total);
    if (!sampleIndices.includes(idx)) sampleIndices.push(idx);
  }

  const sampled = tf.gather(carEmbeddingsMatrix, sampleIndices);
  const proj2d = approximatePCA(sampled, 2);
  sampled.dispose();

  ctxEmbed.clearRect(0, 0, embedCanvas.width, embedCanvas.height);
  const w = embedCanvas.width;
  const h = embedCanvas.height;
  const xs = proj2d.map(p => p[0]);
  const ys = proj2d.map(p => p[1]);
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  const yMin = Math.min(...ys), yMax = Math.max(...ys);
  const xRange = xMax - xMin || 1;
  const yRange = yMax - yMin || 1;

  ctxEmbed.font = '12px Arial';
  proj2d.forEach((p, i) => {
    const carId = indexToCarId[sampleIndices[i]];
    const car = cars.get(carId);
    if (!car) return;

    const x = ((p[0] - xMin) / xRange) * w;
    const y = ((p[1] - yMin) / yRange) * h;

    ctxEmbed.fillStyle = '#1a5276';
    ctxEmbed.beginPath();
    ctxEmbed.arc(x, y, 4, 0, Math.PI * 2);
    ctxEmbed.fill();

    // Optional: draw label on hover (simplified: just draw all)
    // In real app, use mouse events
  });
}

// Train model
async function trainModel() {
  if (!interactions.length) {
    statusEl.textContent = 'No data loaded!';
    return;
  }

  const numUsers = userIdToIndex.size;
  const numCars = carIdToIndex.size;

  // Create brand map
  const brands = new Set();
  cars.forEach(car => brands.add(car.make));
  const brandToIndex = new Map();
  [...brands].forEach((b, i) => brandToIndex.set(b, i));
  const numBrands = brands.size;

  model = new TwoTowerModel(numUsers, numCars, numBrands, config.embeddingDim);
  model.compile(tf.train.adam(config.learningRate));

  lossHistory = [];
  trainBtn.disabled = true;
  testBtn.disabled = true;

  const carBrands = indexToCarId.map(id => {
    const car = cars.get(id);
    return car ? brandToIndex.get(car.make) : 0;
  });

  for (let epoch = 0; epoch < config.epochs; epoch++) {
    let epochLoss = 0;
    let batchCount = 0;

    // Shuffle interactions
    const shuffled = [...interactions].sort(() => Math.random() - 0.5);
    for (let i = 0; i < shuffled.length; i += config.batchSize) {
      const batch = shuffled.slice(i, i + config.batchSize);
      if (batch.length < 2) continue;

      const userIndices = batch.map(i => userIdToIndex.get(i.userId));
      const posCarIndices = batch.map(i => carIdToIndex.get(i.carId));
      const brandIndices = posCarIndices.map(idx => carBrands[idx]);

      // Negative sampling: random cars not in batch
      const negCarIndices = posCarIndices.map(() => 
        Math.floor(Math.random() * numCars)
      );

      const loss = await model.trainStep(
        tf.tensor1d(userIndices, 'int32'),
        tf.tensor1d(posCarIndices, 'int32'),
        tf.tensor1d(negCarIndices, 'int32'),
        tf.tensor1d(brandIndices, 'int32')
      );

      epochLoss += loss;
      batchCount++;
      tf.dispose([loss]);
    }

    const avgLoss = epochLoss / batchCount;
    drawLossChart(avgLoss);
    statusEl.textContent = `Epoch ${epoch + 1}/${config.epochs} | Loss: ${avgLoss.toFixed(4)}`;
    await tf.nextFrame(); // yield to UI
  }

  // Extract car embeddings for inference
  const allCarIndices = tf.range(0, numCars, 1, 'int32');
  const allBrandIndices = tf.tensor1d(carBrands, 'int32');
  carEmbeddingsMatrix = model.itemForward(allCarIndices, allBrandIndices);
  allCarIndices.dispose();
  allBrandIndices.dispose();

  drawEmbeddingProjection();

  statusEl.textContent = 'Training complete!';
  trainBtn.disabled = false;
  testBtn.disabled = false;
}

// Test model
async function testModel() {
  // Find user with >=10 ratings
  let testUser = null;
  for (let [userId, carsList] of userToCars.entries()) {
    if (carsList.length >= 10) {
      testUser = userId;
      break;
    }
  }

  if (!testUser) {
    statusEl.textContent = 'No user with ≥10 ratings found.';
    return;
  }

  const userIndex = userIdToIndex.get(testUser);
  const userEmb = model.getUserEmbedding(userIndex);
  const scores = model.getScoresForAllItems(userEmb, carEmbeddingsMatrix);
  const scoresData = await scores.data();
  userEmb.dispose();
  scores.dispose();

  // Get top-10 recommended (exclude already rated)
  const ratedSet = new Set(userToCars.get(testUser).map(r => r.carId));
  const scoredCars = indexToCarId
    .map((carId, idx) => ({ carId, score: scoresData[idx] }))
    .filter(item => !ratedSet.has(item.carId))
    .sort((a, b) => b.score - a.score)
    .slice(0, 10);

  // Get top-10 historical
  const topRated = userToCars.get(testUser).slice(0, 10);

  // Render side-by-side
  const leftTable = `
    <div class="result-table">
      <h3>Top-10 Rated by User</h3>
      <table>
        <thead><tr><th>Car</th><th>Rating</th></tr></thead>
        <tbody>
          ${topRated.map(r => {
            const car = cars.get(r.carId);
            return `<tr><td>${car?.make || ''} ${car?.model || ''} (${car?.year || ''})</td><td>${r.rating}</td></tr>`;
          }).join('')}
        </tbody>
      </table>
    </div>
  `;

  const rightTable = `
    <div class="result-table">
      <h3>Top-10 Recommended</h3>
      <table>
        <thead><tr><th>Car</th><th>Score</th></tr></thead>
        <tbody>
          ${scoredCars.map(item => {
            const car = cars.get(item.carId);
            return `<tr><td>${car?.make || ''} ${car?.model || ''} (${car?.year || ''})</td><td>${item.score.toFixed(3)}</td></tr>`;
          }).join('')}
        </tbody>
      </table>
    </div>
  `;

  resultsEl.innerHTML = leftTable + rightTable;
  statusEl.textContent = `Recommendations for user ${testUser}`;
}

// Event listeners
loadBtn.addEventListener('click', loadData);
trainBtn.addEventListener('click', trainModel);
testBtn.addEventListener('click', testModel);

// Initialize canvas sizes
window.addEventListener('load', () => {
  lossCanvas.width = lossCanvas.clientWidth;
  lossCanvas.height = lossCanvas.clientHeight;
  embedCanvas.width = embedCanvas.clientWidth;
  embedCanvas.height = embedCanvas.clientHeight;
});
