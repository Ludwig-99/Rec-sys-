// Конвертация: 1 USD = 83 INR
const INR_TO_USD = 83;
let carData = [];

// Загрузка CSV-файла
async function loadCarData() {
  const response = await fetch('CAR DETAILS FROM CAR DEKHO.csv');
  const text = await response.text();
  const lines = text.split('\n').filter(line => line.trim() !== '');

  // Определяем, есть ли заголовок
  const hasHeader = lines[0].toLowerCase().includes('name') || lines[0].includes('year');
  const startIndex = hasHeader ? 1 : 0;

  carData = lines.slice(startIndex).map(line => {
    // Разделяем по запятым, но учитываем кавычки (простая реализация)
    const fields = [];
    let current = '';
    let inQuotes = false;

    for (let char of line) {
      if (char === '"') {
        inQuotes = !inQuotes;
      } else if (char === ',' && !inQuotes) {
        fields.push(current.trim());
        current = '';
      } else {
        current += char;
      }
    }
    fields.push(current.trim());

    // Ожидаем 8 полей
    if (fields.length >= 8) {
      return [
        fields[0], // name
        parseInt(fields[1]) || 0, // year
        parseInt(fields[2]) || 0, // selling_price (INR)
        parseInt(fields[3]) || 0, // km_driven
        fields[4], // fuel
        fields[5], // seller_type
        fields[6], // transmission
        fields[7]  // owner
      ];
    }
    return null;
  }).filter(Boolean);

  console.log(`Loaded ${carData.length} cars.`);
}

// Инициализация
document.addEventListener('DOMContentLoaded', () => {
  const loadingEl = document.getElementById('loading');
  loadingEl.textContent = 'Loading car data...';
  loadCarData().then(() => {
    loadingEl.textContent = '';
  }).catch(err => {
    loadingEl.textContent = 'Failed to load car data.';
    console.error(err);
  });
});

function getRecommendations() {
  if (carData.length === 0) {
    document.getElementById("results").innerHTML = "<p>Please wait while car data is loading...</p>";
    return;
  }

  const maxPriceUSD = parseFloat(document.getElementById("maxPrice").value) || Infinity;
  const fuelType = document.getElementById("fuelType").value || null;
  const transmission = document.getElementById("transmission").value || null;
  const minYear = parseInt(document.getElementById("minYear").value) || 0;
  const maxKm = parseInt(document.getElementById("maxKm").value) || Infinity;
  const owner = document.getElementById("owner").value || null;

  const maxPriceINR = maxPriceUSD * INR_TO_USD;

  let filtered = carData.filter(car => {
    const [name, year, priceINR, km, fuel, seller, trans, own] = car;
    return (
      priceINR <= maxPriceINR &&
      year >= minYear &&
      km <= maxKm &&
      (fuelType === null || fuel === fuelType) &&
      (transmission === null || trans === transmission) &&
      (owner === null || own === owner)
    );
  });

  if (filtered.length === 0) {
    document.getElementById("results").innerHTML = "<p>No cars match your criteria.</p>";
    return;
  }

  // Сортировка: чем выше соотношение цена/пробег — тем лучше состояние
  filtered.sort((a, b) => {
    const scoreA = a[2] / (a[3] + 1);
    const scoreB = b[2] / (b[3] + 1);
    return scoreB - scoreA;
  });

  const top5 = filtered.slice(0, 5);
  const html = top5.map(car => {
    const [name, year, priceINR, km, fuel, seller, trans, own] = car;
    const priceUSD = Math.round(priceINR / INR_TO_USD).toLocaleString();
    return `
      <div class="car-card">
        <strong>${name} (${year})</strong><br>
        Price: $${priceUSD} | Mileage: ${km.toLocaleString()} km<br>
        Fuel: ${fuel} | Transmission: ${trans}<br>
        Owner: ${own}
      </div>
    `;
  }).join("");

  document.getElementById("results").innerHTML = `<h3>Top Recommendations:</h3>${html}`;
}
