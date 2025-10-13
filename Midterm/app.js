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
  maxInteractions: 2000,
  embeddingDim: 32,
  epochs: 20,
  batchSize: 64,
  learningRate: 0.01,
  sampleCarCount: 400
};

// Embedded car data from car data.csv
const CAR_DATA_CSV = `Car_Name,Year,Selling_Price,Present_Price,Kms_Driven,Fuel_Type,Seller_Type,Transmission,Owner
ritz,2014,3.35,5.59,27000,Petrol,Dealer,Manual,0
sx4,2013,4.75,9.54,43000,Diesel,Dealer,Manual,0
ciaz,2017,7.25,9.85,6900,Petrol,Dealer,Manual,0
wagon r,2011,2.85,4.15,5200,Petrol,Dealer,Manual,0
swift,2014,4.6,6.87,42450,Diesel,Dealer,Manual,0
vitara brezza,2018,9.25,9.83,2071,Diesel,Dealer,Manual,0
ciaz,2015,6.75,8.12,18796,Petrol,Dealer,Manual,0
s cross,2015,6.5,8.61,33429,Diesel,Dealer,Manual,0
ciaz,2016,8.75,8.89,20273,Diesel,Dealer,Manual,0
ciaz,2015,7.45,8.92,42367,Diesel,Dealer,Manual,0
alto 800,2017,2.85,3.6,2135,Petrol,Dealer,Manual,0
ciaz,2015,6.85,10.38,51000,Diesel,Dealer,Manual,0
ciaz,2015,7.5,9.94,15000,Petrol,Dealer,Automatic,0
ertiga,2015,6.1,7.71,26000,Petrol,Dealer,Manual,0
dzire,2009,2.25,7.21,77427,Petrol,Dealer,Manual,0
ertiga,2016,7.75,10.79,43000,Diesel,Dealer,Manual,0
ertiga,2015,7.25,10.79,41678,Diesel,Dealer,Manual,0
ertiga,2016,7.75,10.79,43000,Diesel,Dealer,Manual,0
wagon r,2015,3.25,5.09,35500,CNG,Dealer,Manual,0
sx4,2010,2.65,7.98,41442,Petrol,Dealer,Manual,0
alto k10,2016,2.85,3.95,25000,Petrol,Dealer,Manual,0
ignis,2017,4.9,5.71,2400,Petrol,Dealer,Manual,0
sx4,2011,4.4,8.01,50000,Petrol,Dealer,Automatic,0
alto k10,2014,2.5,3.46,45280,Petrol,Dealer,Manual,0
wagon r,2013,2.9,4.41,56879,Petrol,Dealer,Manual,0
swift,2011,3,4.99,20000,Petrol,Dealer,Manual,0
swift,2013,4.15,5.87,55138,Petrol,Dealer,Manual,0
swift,2017,6,6.49,16200,Petrol,Individual,Manual,0
alto k10,2010,1.95,3.95,44542,Petrol,Dealer,Manual,0
ciaz,2015,7.45,10.38,45000,Diesel,Dealer,Manual,0
ritz,2012,3.1,5.98,51439,Diesel,Dealer,Manual,0
ritz,2011,2.35,4.89,54200,Petrol,Dealer,Manual,0
swift,2014,4.95,7.49,39000,Diesel,Dealer,Manual,0
ertiga,2014,6,9.95,45000,Diesel,Dealer,Manual,0
dzire,2014,5.5,8.06,45000,Diesel,Dealer,Manual,0
sx4,2011,2.95,7.74,49998,CNG,Dealer,Manual,0
dzire,2015,4.65,7.2,48767,Petrol,Dealer,Manual,0
800,2003,0.35,2.28,127000,Petrol,Individual,Manual,0
alto k10,2016,3,3.76,10079,Petrol,Dealer,Manual,0
sx4,2003,2.25,7.98,62000,Petrol,Dealer,Manual,0
baleno,2016,5.85,7.87,24524,Petrol,Dealer,Automatic,0
alto k10,2014,2.55,3.98,46706,Petrol,Dealer,Manual,0
sx4,2008,1.95,7.15,58000,Petrol,Dealer,Manual,0
dzire,2014,5.5,8.06,45780,Diesel,Dealer,Manual,0
omni,2012,1.25,2.69,50000,Petrol,Dealer,Manual,0
ciaz,2014,7.5,12.04,15000,Petrol,Dealer,Automatic,0
ritz,2013,2.65,4.89,64532,Petrol,Dealer,Manual,0
wagon r,2006,1.05,4.15,65000,Petrol,Dealer,Manual,0
ertiga,2015,5.8,7.71,25870,Petrol,Dealer,Manual,0
ciaz,2017,7.75,9.29,37000,Petrol,Dealer,Automatic,0
fortuner,2012,14.9,30.61,104707,Diesel,Dealer,Automatic,0
fortuner,2015,23,30.61,40000,Diesel,Dealer,Automatic,0
innova,2017,18,19.77,15000,Diesel,Dealer,Automatic,0
fortuner,2013,16,30.61,135000,Diesel,Individual,Automatic,0
innova,2005,2.75,10.21,90000,Petrol,Individual,Manual,0
corolla altis,2009,3.6,15.04,70000,Petrol,Dealer,Automatic,0
etios cross,2015,4.5,7.27,40534,Petrol,Dealer,Manual,0
corolla altis,2010,4.75,18.54,50000,Petrol,Dealer,Manual,0
etios g,2014,4.1,6.8,39485,Petrol,Dealer,Manual,1
fortuner,2014,19.99,35.96,41000,Diesel,Dealer,Automatic,0
corolla altis,2013,6.95,18.61,40001,Petrol,Dealer,Manual,0
etios cross,2015,4.5,7.7,40588,Petrol,Dealer,Manual,0
fortuner,2014,18.75,35.96,78000,Diesel,Dealer,Automatic,0
fortuner,2015,23.5,35.96,47000,Diesel,Dealer,Automatic,0
fortuner,2017,33,36.23,6000,Diesel,Dealer,Automatic,0
etios liva,2014,4.75,6.95,45000,Diesel,Dealer,Manual,0
innova,2017,19.75,23.15,11000,Petrol,Dealer,Automatic,0
fortuner,2010,9.25,20.45,59000,Diesel,Dealer,Manual,0
corolla altis,2011,4.35,13.74,88000,Petrol,Dealer,Manual,0
corolla altis,2016,14.25,20.91,12000,Petrol,Dealer,Manual,0
etios liva,2014,3.95,6.76,71000,Diesel,Dealer,Manual,0
corolla altis,2011,4.5,12.48,45000,Diesel,Dealer,Manual,0
corolla altis,2013,7.45,18.61,56001,Petrol,Dealer,Manual,0
etios liva,2011,2.65,5.71,43000,Petrol,Dealer,Manual,0
etios cross,2014,4.9,8.93,83000,Diesel,Dealer,Manual,0
etios g,2015,3.95,6.8,36000,Petrol,Dealer,Manual,0
corolla altis,2013,5.5,14.68,72000,Petrol,Dealer,Manual,0
corolla,2004,1.5,12.35,135154,Petrol,Dealer,Automatic,0
corolla altis,2010,5.25,22.83,80000,Petrol,Dealer,Automatic,0
fortuner,2012,14.5,30.61,89000,Diesel,Dealer,Automatic,0
corolla altis,2016,14.73,14.89,23000,Diesel,Dealer,Manual,0
etios gd,2015,4.75,7.85,40000,Diesel,Dealer,Manual,0
innova,2017,23,25.39,15000,Diesel,Dealer,Automatic,0
innova,2015,12.5,13.46,38000,Diesel,Dealer,Manual,0
innova,2005,3.49,13.46,197176,Diesel,Dealer,Manual,0
camry,2006,2.5,23.73,142000,Petrol,Individual,Automatic,3
land cruiser,2010,35,92.6,78000,Diesel,Dealer,Manual,0
corolla altis,2012,5.9,13.74,56000,Petrol,Dealer,Manual,0
etios liva,2013,3.45,6.05,47000,Petrol,Dealer,Manual,0
etios g,2014,4.75,6.76,40000,Petrol,Dealer,Manual,0
corolla altis,2009,3.8,18.61,62000,Petrol,Dealer,Manual,0
innova,2014,11.25,16.09,58242,Diesel,Dealer,Manual,0
innova,2005,3.51,13.7,75000,Petrol,Dealer,Manual,0
fortuner,2015,23,30.61,40000,Diesel,Dealer,Automatic,0
corolla altis,2008,4,22.78,89000,Petrol,Dealer,Automatic,0
corolla altis,2012,5.85,18.61,72000,Petrol,Dealer,Manual,0
innova,2016,20.75,25.39,29000,Diesel,Dealer,Automatic,0
corolla altis,2017,17,18.64,8700,Petrol,Dealer,Manual,0
corolla altis,2013,7.05,18.61,45000,Petrol,Dealer,Manual,0
fortuner,2010,9.65,20.45,50024,Diesel,Dealer,Manual,0
brio,2016,5.3,5.9,5464,Petrol,Dealer,Manual,0
`;

const statusEl = document.getElementById('status');
const loadBtn = document.getElementById('loadBtn');
const trainBtn = document.getElementById('trainBtn');
const testBtn = document.getElementById('testBtn');
const lossCanvas = document.getElementById('lossChart');
const embedCanvas = document.getElementById('embeddingCanvas');
const resultsEl = document.getElementById('results');

const ctxLoss = lossCanvas.getContext('2d');
const ctxEmbed = embedCanvas.getContext('2d');

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

function generateDataFromCSV() {
  const rows = parseCSV(CAR_DATA_CSV);
  cars.clear();
  rows.forEach((row, idx) => {
    const carId = `car_${idx}`;
    cars.set(carId, {
      make: row.Car_Name.split(' ')[0] || 'Unknown',
      model: row.Car_Name || 'Unknown',
      year: row.Year || '',
      bodyType: 'Sedan',
      priceRange: parseFloat(row.Selling_Price) > 10 ? 'High' : parseFloat(row.Selling_Price) > 5 ? 'Mid' : 'Low'
    });
  });

  interactions = [];
  const numUsers = 100;
  const numCars = rows.length;
  const maxRatingsPerUser = 30;

  for (let u = 0; u < numUsers; u++) {
    const userId = `user_${u}`;
    const numRatings = 5 + Math.floor(Math.random() * maxRatingsPerUser);
    const rated = new Set();
    for (let r = 0; r < numRatings; r++) {
      const carIdx = Math.floor(Math.random() * numCars);
      if (rated.has(carIdx)) continue;
      rated.add(carIdx);
      const carId = `car_${carIdx}`;
      const rating = Math.min(5, Math.max(1, parseFloat(rows[carIdx].Selling_Price) / 2 + Math.random() * 2));
      interactions.push({
        userId,
        carId,
        rating,
        ts: Date.now() - Math.floor(Math.random() * 365 * 24 * 60 * 60 * 1000)
      });
    }
  }
}

async function loadData() {
  try {
    statusEl.textContent = 'Processing car data...';
    generateDataFromCSV();

    const uniqueUsers = [...new Set(interactions.map(i => i.userId))];
    const uniqueCars = [...cars.keys()];

    userIdToIndex.clear();
    carIdToIndex.clear();
    indexToUserId = uniqueUsers;
    indexToCarId = uniqueCars;

    uniqueUsers.forEach((id, idx) => userIdToIndex.set(id, idx));
    uniqueCars.forEach((id, idx) => carIdToIndex.set(id, idx));

    userToCars.clear();
    interactions.forEach(i => {
      if (!userToCars.has(i.userId)) userToCars.set(i.userId, []);
      userToCars.get(i.userId).push({ carId: i.carId, rating: i.rating, ts: i.ts });
    });

    for (let [userId, carsList] of userToCars.entries()) {
      carsList.sort((a, b) => b.rating - a.rating || b.ts - a.ts);
    }

    statusEl.textContent = `Generated ${interactions.length} interactions, ${uniqueUsers.length} users, ${uniqueCars.length} cars.`;
    trainBtn.disabled = false;
  } catch (err) {
    statusEl.textContent = `Error: ${err.message}`;
    console.error(err);
  }
}

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

function approximatePCA(embeddings, nComponents = 2) {
  const n = embeddings.shape[0];
  const d = embeddings.shape[1];
  const data = embeddings.dataSync();

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

  proj2d.forEach((p, i) => {
    const x = ((p[0] - xMin) / xRange) * w;
    const y = ((p[1] - yMin) / yRange) * h;
    ctxEmbed.fillStyle = '#1a5276';
    ctxEmbed.beginPath();
    ctxEmbed.arc(x, y, 3, 0, Math.PI * 2);
    ctxEmbed.fill();
  });
}

async function trainModel() {
  const numUsers = userIdToIndex.size;
  const numCars = carIdToIndex.size;

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

    const shuffled = [...interactions].sort(() => Math.random() - 0.5);
    for (let i = 0; i < shuffled.length; i += config.batchSize) {
      const batch = shuffled.slice(i, i + config.batchSize);
      if (batch.length < 2) continue;

      const userIndices = batch.map(i => userIdToIndex.get(i.userId));
      const posCarIndices = batch.map(i => carIdToIndex.get(i.carId));
      const brandIndices = posCarIndices.map(idx => carBrands[idx]);
      const negCarIndices = posCarIndices.map(() => Math.floor(Math.random() * numCars));

      const loss = await model.trainStep(
        tf.tensor1d(userIndices, 'int32'),
        tf.tensor1d(posCarIndices, 'int32'),
        tf.tensor1d(negCarIndices, 'int32'),
        tf.tensor1d(brandIndices, 'int32')
      );

      epochLoss += await loss.data()[0];
      batchCount++;
      tf.dispose(loss);
    }

    const avgLoss = epochLoss / batchCount;
    drawLossChart(avgLoss);
    statusEl.textContent = `Epoch ${epoch + 1}/${config.epochs} | Loss: ${avgLoss.toFixed(4)}`;
    await tf.nextFrame();
  }

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

async function testModel() {
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

  const ratedSet = new Set(userToCars.get(testUser).map(r => r.carId));
  const scoredCars = indexToCarId
    .map((carId, idx) => ({ carId, score: scoresData[idx] }))
    .filter(item => !ratedSet.has(item.carId))
    .sort((a, b) => b.score - a.score)
    .slice(0, 10);

  const topRated = userToCars.get(testUser).slice(0, 10);

  const leftTable = `
    <div class="result-table">
      <h3>Top-10 Rated by User</h3>
      <table>
        <thead><tr><th>Car</th><th>Rating</th></tr></thead>
        <tbody>
          ${topRated.map(r => {
            const car = cars.get(r.carId);
            return `<tr><td>${car?.make || ''} ${car?.model || ''} (${car?.year || ''})</td><td>${r.rating.toFixed(1)}</td></tr>`;
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

loadBtn.addEventListener('click', loadData);
trainBtn.addEventListener('click', trainModel);
testBtn.addEventListener('click', testModel);

window.addEventListener('load', () => {
  lossCanvas.width = lossCanvas.clientWidth;
  lossCanvas.height = lossCanvas.clientHeight;
  embedCanvas.width = embedCanvas.clientWidth;
  embedCanvas.height = embedCanvas.clientHeight;
});
