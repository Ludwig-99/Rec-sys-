// Загрузка данных (в реальном проекте — через fetch или API)
const carData = [
  // Примеры строк из CAR DETAILS FROM CAR DEKHO.csv
  // Формат: [name, year, selling_price, km_driven, fuel_type, seller_type, transmission, owner]
  ["Hyundai i20 Asta 1.2", 2014, 475000, 23122, "Petrol", "Dealer", "Manual", "Second Owner"],
  ["Maruti Swift VDI", 2016, 630000, 55000, "Diesel", "Dealer", "Manual", "First Owner"],
  ["Honda City i DTEC VX", 2014, 600000, 90000, "Diesel", "Individual", "Manual", "Second Owner"],
  ["Toyota Fortuner 4x2 AT", 2017, 2600000, 47162, "Diesel", "Trustmark Dealer", "Automatic", "First Owner"],
  ["Tata Harrier XZ BSIV", 2019, 1700000, 10000, "Diesel", "Individual", "Manual", "First Owner"],
  ["Hyundai Creta 1.6 SX Option", 2017, 1025000, 9000, "Petrol", "Dealer", "Manual", "First Owner"],
  ["Maruti Alto 800 LXI", 2018, 310000, 5000, "Petrol", "Individual", "Manual", "First Owner"],
  ["Maruti Ertiga VDI", 2019, 925000, 50000, "Diesel", "Individual", "Manual", "First Owner"],
  ["Honda WR-V i-DTEC VX", 2019, 1240000, 13000, "Diesel", "Dealer", "Manual", "First Owner"],
  ["Renault KWID RXT", 2018, 360000, 26500, "Petrol", "Individual", "Manual", "First Owner"]
  // В реальном проекте здесь будет весь CSV (можно загрузить через PapaParse или встроить как JSON)
];

function getRecommendations() {
  const maxPrice = parseFloat(document.getElementById("maxPrice").value) || Infinity;
  const fuelType = document.getElementById("fuelType").value || null;
  const minYear = parseInt(document.getElementById("minYear").value) || 0;

  let filtered = carData.filter(car => {
    const [name, year, price, km, fuel] = car;
    return (
      price <= maxPrice &&
      year >= minYear &&
      (fuelType === null || fuel === fuelType)
    );
  });

  if (filtered.length === 0) {
    document.getElementById("results").innerHTML = "<p>No cars match your criteria.</p>";
    return;
  }

  // Сортировка: чем меньше пробег и выше цена (для данного бюджета) — тем лучше
  filtered.sort((a, b) => {
    const scoreA = a[2] / (a[3] + 1); // price / (km + 1)
    const scoreB = b[2] / (b[3] + 1);
    return scoreB - scoreA; // descending
  });

  const top5 = filtered.slice(0, 5);
  const html = top5.map(car => {
    const [name, year, price, km, fuel, seller, transmission, owner] = car;
    return `
      <div class="car-card">
        <strong>${name} (${year})</strong><br>
        Price: ₹${price.toLocaleString()} | Mileage: ${km.toLocaleString()} km<br>
        Fuel: ${fuel} | Transmission: ${transmission}<br>
        Owner: ${owner}
      </div>
    `;
  }).join("");

  document.getElementById("results").innerHTML = `<h3>Top Recommendations:</h3>${html}`;
}
