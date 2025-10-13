// two-tower.js
class TwoTowerModel {
  constructor(numUsers, numCars, numBrands, embDim, hiddenDim = 64) {
    this.numUsers = numUsers;
    this.numCars = numCars;
    this.numBrands = numBrands;
    this.embDim = embDim;
    this.hiddenDim = hiddenDim;

    this.userEmbedding = tf.variable(tf.randomNormal([numUsers, embDim], 0, 0.1));
    this.userDense1 = tf.layers.dense({ units: hiddenDim, activation: 'relu' });
    this.userDense2 = tf.layers.dense({ units: hiddenDim, activation: 'relu' });
    this.userDenseOut = tf.layers.dense({ units: embDim, activation: null });

    this.carEmbedding = tf.variable(tf.randomNormal([numCars, embDim], 0, 0.1));
    this.brandEmbedding = tf.variable(tf.randomNormal([numBrands, embDim], 0, 0.1));
    this.itemDense1 = tf.layers.dense({ units: hiddenDim, activation: 'relu' });
    this.itemDense2 = tf.layers.dense({ units: hiddenDim, activation: 'relu' });
    this.itemDenseOut = tf.layers.dense({ units: embDim, activation: null });
  }

  userForward(userIdx) {
    const emb = tf.gather(this.userEmbedding, userIdx);
    let x = this.userDense1.apply(emb);
    x = this.userDense2.apply(x);
    x = this.userDenseOut.apply(x);
    const norm = tf.linalg.normalize(x, 2, 1);
    return norm.normalized;
  }

  itemForward(carIdx, brandIdx) {
    const carEmb = tf.gather(this.carEmbedding, carIdx);
    const brandEmb = tf.gather(this.brandEmbedding, brandIdx);
    const concat = tf.concat([carEmb, brandEmb], 1);
    let x = this.itemDense1.apply(concat);
    x = this.itemDense2.apply(x);
    x = this.itemDenseOut.apply(x);
    const norm = tf.linalg.normalize(x, 2, 1);
    return norm.normalized;
  }

  score(userEmb, itemEmb) {
    return tf.sum(tf.mul(userEmb, itemEmb), 1, true);
  }

  compile(optimizer) {
    this.optimizer = optimizer;
  }

  async trainStep(userIdx, posCarIdx, negCarIdx, brandIdx) {
    return tf.tidy(() => {
      const batchBrandIdx = brandIdx;
      const negBrandIdx = tf.gather(brandIdx, tf.randomUniform([negCarIdx.shape[0]], 0, brandIdx.shape[0], 'int32'));

      const userEmb = this.userForward(userIdx);
      const posItemEmb = this.itemForward(posCarIdx, batchBrandIdx);
      const negItemEmb = this.itemForward(negCarIdx, negBrandIdx);

      const posScores = this.score(userEmb, posItemEmb);
      const negScores = this.score(userEmb, negItemEmb);

      const logits = tf.concat([posScores, negScores], 1);
      const labels = tf.zeros([userIdx.shape[0]], 'int32');
      const loss = tf.losses.sparseCategoricalCrossentropy(labels, logits);

      this.optimizer.minimize(() => loss, true, [
        this.userEmbedding,
        this.carEmbedding,
        this.brandEmbedding,
        ...this.userDense1.trainableWeights,
        ...this.userDense2.trainableWeights,
        ...this.userDenseOut.trainableWeights,
        ...this.itemDense1.trainableWeights,
        ...this.itemDense2.trainableWeights,
        ...this.itemDenseOut.trainableWeights
      ]);

      return loss;
    });
  }

  getUserEmbedding(userIdx) {
    return tf.tidy(() => {
      const u = tf.tensor1d([userIdx], 'int32');
      const emb = this.userForward(u);
      u.dispose();
      return emb;
    });
  }

  getScoresForAllItems(userEmb, allItemEmbs) {
    return tf.tidy(() => {
      const u = tf.expandDims(userEmb, 0);
      const scores = tf.matMul(u, allItemEmbs, false, true);
      return tf.squeeze(scores, [0]);
    });
  }
}
