// two-tower.js
// Minimal Two-Tower Model for Car Recommendation in TensorFlow.js

class TwoTowerModel {
  constructor(numUsers, numCars, embDim) {
    this.numUsers = numUsers;
    this.numCars = numCars;
    this.embDim = embDim;

    // User Tower
    this.userTower = tf.sequential({
      layers: [
        tf.layers.embedding({ inputDim: numUsers, outputDim: embDim }),
        tf.layers.dense({ units: 64, activation: 'relu' }),
        tf.layers.dense({ units: 32, activation: 'relu' }),
        tf.layers.dense({ units: embDim, activation: 'linear' })
      ]
    });

    // Item (Car) Tower
    this.itemTower = tf.sequential({
      layers: [
        tf.layers.embedding({ inputDim: numCars, outputDim: embDim }),
        tf.layers.dense({ units: 64, activation: 'relu' }),
        tf.layers.dense({ units: 32, activation: 'relu' }),
        tf.layers.dense({ units: embDim, activation: 'linear' })
      ]
    });
  }

  userForward(userIdxTensor) {
    return this.userTower.apply(userIdxTensor);
  }

  itemForward(carIdxTensor) {
    return this.itemTower.apply(carIdxTensor);
  }

  // Not used directly but kept for interface consistency
  score(userEmb, itemEmb) {
    return tf.sum(userEmb.mul(itemEmb), axis: 1);
  }
}
