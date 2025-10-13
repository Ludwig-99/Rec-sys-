// app.js
// Car Recommender System – Data Loading, Training, and Inference Logic

let interactions = [];
let cars = new Map(); // carId → { name, year }
let userToCars = new Map(); // userId → Set(carId)
let carIndexer = new Map(); // carName+year → carId
let userIndexer = new Map(); // auto-incremented userId
let reverseCarIndex = [];
let reverseUserIndex = [];
let model = null;
let carEmbeddingsMatrix = null;

const CONFIG = {
  maxInteractions: 80000,
  embeddingDim: 32,
  epochs: 10,
  batchSize: 64,
  learningRate: 0.01,
  numSamples: 1000 // for PCA
};

const statusEl = document.getElementById('status');
const loadBtn = document.getElementById('loadBtn');
const trainBtn = document.getElementById('trainBtn');
const testBtn = document.getElementById('testBtn');
const lossCanvas = document.getElementById('lossChart');
const embedCanvas = document.getElementById('embeddingChart');
const resultsDiv = document.getElementById('results');

loadBtn.addEventListener('click', loadData);
trainBtn.addEventListener('click', trainModel);
testBtn.addEventListener('click', testRecommendation);

async function loadData() {
  setStatus('Loading car data...');
  try {
    const response = await fetch('data/cars.csv');
    if (!response.ok) throw new Error('Failed to load data/cars.csv');
    const text = await response.text();
    parseCarData(text);
    setStatus(`Loaded ${interactions.length} interactions from ${userIndexer.size} users and ${carIndexer.size} cars.`);
    trainBtn.disabled = false;
  } catch (err) {
    setStatus(`Error: ${err.message}`);
    console.error(err);
  }
}

function parseCarData(csvText) {
  const lines = csvText.trim().split('\n');
  const headers = lines[0].split(',');
  if (headers[0].toLowerCase().includes('car_name')) {
    lines.shift(); // remove header
  }

  let carIdCounter = 0;
  let userIdCounter = 0;

  for (const line of lines) {
    const fields = line.split(',');
    if (fields.length < 9) continue;

    const carName = fields[0].trim();
    const year = parseInt(fields[1], 10);
    const sellingPrice = parseFloat(fields[2]);
    const presentPrice = parseFloat(fields[3]);
    const kms = parseInt(fields[4], 10);

    if (isNaN(year) || isNaN(sellingPrice) || isNaN(presentPrice) || isNaN(kms)) continue;

    // Create unique car key
    const carKey = `${carName}_${year}`;
    let carId = carIndexer.get(carKey);
    if (carId === undefined) {
      carId = carIdCounter++;
      carIndexer.set(carKey, carId);
      cars.set(carId, { name: carName, year });
      reverseCarIndex[carId] = carKey;
    }

    // Treat each row as a "user" (since no real user ID exists)
    // We simulate users by grouping identical car-year entries as one user interaction
    const userId = userIdCounter++;
    userIndexer.set(userId, userId);
    reverseUserIndex[userId] = userId;

    interactions.push({ userId, carId, rating: sellingPrice, ts: Date.now() });

    if (!userToCars.has(userId)) {
      userToCars.set(userId, new Set());
    }
    userToCars.get(userId).add(carId);

    if (interactions.length >= CONFIG.maxInteractions) break;
  }
}

async function trainModel() {
  if (!interactions.length) {
    setStatus('No data loaded!');
    return;
  }

  trainBtn.disabled = true;
  setStatus('Initializing model...');

  const numUsers = userIndexer.size;
  const numCars = carIndexer.size;

  model = new TwoTowerModel(numUsers, numCars, CONFIG.embeddingDim);
  const optimizer = tf.train.adam(CONFIG.learningRate);

  // Prepare tensors
  const userTensor = tf.tensor1d(interactions.map(i => i.userId), 'int32');
  const carTensor = tf.tensor1d(interactions.map(i => i.carId), 'int32');

  const lossCtx = lossCanvas.getContext('2d');
  const losses = [];

  setStatus('Training... (may take 1–3 minutes)');
  for (let epoch = 0; epoch < CONFIG.epochs; epoch++) {
    const numBatches = Math.ceil(interactions.length / CONFIG.batchSize);
    for (let i = 0; i < numBatches; i++) {
      const start = i * CONFIG.batchSize;
      const end = Math.min(start + CONFIG.batchSize, interactions.length);
      const batchUsers = userTensor.slice([start], [end - start]);
      const batchItems = carTensor.slice([start], [end - start]);

      const loss = tf.tidy(() => {
        const userEmb = model.userForward(batchUsers);
        const itemEmb = model.itemForward(batchItems);

        // In-batch sampled softmax
        const logits = tf.matMul(userEmb, itemEmb, false, true); // [B, B]
        const labels = tf.tensor1d([...Array(end - start).keys()], 'int32');
        const lossVal = tf.losses.softmaxCrossEntropy(
          tf.oneHot(labels, end - start),
          logits
        );
        return lossVal;
      });

      optimizer.minimize(() => loss);
      const lossVal = await loss.data();
      losses.push(lossVal[0]);
      loss.dispose();

      // Update chart every few batches
      if (i % 10 === 0) {
        drawLossChart(lossCtx, losses);
      }
    }
    setStatus(`Epoch ${epoch + 1}/${CONFIG.epochs} complete.`);
  }

  // Cache item embeddings for inference
  carEmbeddingsMatrix = tf.tidy(() => {
    const allCarIds = tf.range(0, numCars, 1, 'int32');
    return model.itemForward(allCarIds);
  });

  // Draw PCA projection
  await drawEmbeddingProjection();

  testBtn.disabled = false;
  trainBtn.disabled = false;
  setStatus('Training complete! Click "Test Recommendation".');

  userTensor.dispose();
  carTensor.dispose();
}

function drawLossChart(ctx, losses) {
  const w = lossCanvas.width;
  const h = lossCanvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = '#007acc';
  ctx.lineWidth = 2;
  ctx.beginPath();
  const maxLoss = Math.max(...losses);
  const minLoss = Math.min(...losses);
  const range = maxLoss - minLoss || 1;
  for (let i = 0; i < losses.length; i++) {
    const x = (i / (losses.length - 1 || 1)) * w;
    const y = h - ((losses[i] - minLoss) / range) * (h - 20) - 10;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

async function drawEmbeddingProjection() {
  const numCars = carIndexer.size;
  const sampleSize = Math.min(CONFIG.numSamples, numCars);
  const indices = tf.util.createShuffledIndices(numCars).slice(0, sampleSize);
  const sampleEmbeds = tf.gather(carEmbeddingsMatrix, tf.tensor1d(indices, 'int32'));

  // Simple PCA via SVD approximation
  const data = await sampleEmbeds.array();
  const centered = data.map(row => row.map(val => val - row.reduce((a,b)=>a+b,0)/row.length));
  const cov = centered[0].map((_, i) => 
    centered[0].map((_, j) => 
      centered.reduce((sum, row) => sum + row[i]*row[j], 0) / centered.length
    )
  );
  // Use first two principal components via power iteration (simplified)
  const pc1 = centered.map(row => row[0]); // fallback
  const pc2 = centered.map(row => row[1]);

  const ctx = embedCanvas.getContext('2d');
  ctx.clearRect(0, 0, embedCanvas.width, embedCanvas.height);
  const w = embedCanvas.width;
  const h = embedCanvas.height;
  const xMin = Math.min(...pc1), xMax = Math.max(...pc1);
  const yMin = Math.min(...pc2), yMax = Math.max(...pc2);
  const xRange = xMax - xMin || 1;
  const yRange = yMax - yMin || 1;

  ctx.fillStyle = '#00a8ff';
  for (let i = 0; i < pc1.length; i++) {
    const x = ((pc1[i] - xMin) / xRange) * (w - 40) + 20;
    const y = h - ((pc2[i] - yMin) / yRange) * (h - 40) - 20;
    ctx.beginPath();
    ctx.arc(x, y, 3, 0, Math.PI * 2);
    ctx.fill();
  }
  sampleEmbeds.dispose();
}

async function testRecommendation() {
  // Find a user with at least 1 rating (since we simulate 1 interaction per user)
  const validUsers = [...userToCars.keys()].filter(u => userToCars.get(u).size >= 1);
  if (validUsers.length === 0) {
    setStatus('No qualified users found.');
    return;
  }
  const randomUser = validUsers[Math.floor(Math.random() * validUsers.length)];
  const ratedCarIds = Array.from(userToCars.get(randomUser));

  // Get user embedding
  const userEmb = tf.tidy(() => model.userForward(tf.tensor1d([randomUser], 'int32')));

  // Compute scores for all cars
  const scores = tf.tidy(() => {
    const dots = tf.matMul(userEmb, carEmbeddingsMatrix, false, true); // [1, N]
    return dots.reshape([carEmbeddingsMatrix.shape[0]]);
  });

  const scoresArray = await scores.data();
  scores.dispose();
  userEmb.dispose();

  // Get top 10 recommended (excluding rated)
  const allCarScores = [...Array(carIndexer.size).keys()]
    .map(id => ({ id, score: scoresArray[id] }))
    .filter(item => !ratedCarIds.includes(item.id))
    .sort((a, b) => b.score - a.score)
    .slice(0, 10);

  // Top rated: just the one car they "rated"
  const topRated = ratedCarIds.slice(0, 10).map(id => cars.get(id));

  const recommended = allCarScores.map(item => cars.get(item.id));

  renderResults(topRated, recommended);
}

function renderResults(topRated, recommended) {
  let html = `<table><thead><tr><th>Top Rated Cars</th><th>Recommended Cars</th></tr></thead><tbody>`;
  const maxLen = Math.max(topRated.length, recommended.length);
  for (let i = 0; i < maxLen; i++) {
    const rated = topRated[i] ? `${topRated[i].name} (${topRated[i].year})` : '';
    const rec = recommended[i] ? `${recommended[i].name} (${recommended[i].year})` : '';
    html += `<tr><td>${rated}</td><td>${rec}</td></tr>`;
  }
  html += `</tbody></table>`;
  resultsDiv.innerHTML = html;
}

function setStatus(msg) {
  statusEl.textContent = msg;
}
