// two-tower.js
class TwoTowerModel {
  constructor(numUsers, numCars, numBrands, embDim, hiddenDim = 64) {
    this.numUsers = numUsers;
    this.numCars = numCars;
    this.numBrands = numBrands;
    this.embDim = embDim;
    this.hiddenDim = hiddenDim;

    // User tower
    this.userEmbedding = tf.variable(
      tf.randomNormal([numUsers, embDim], 0, 0.1)
    );
    this.userDense1 = tf.layers.dense({ units: hiddenDim, activation: 'relu' });
    this.userDense2 = tf.layers.dense({ units: hiddenDim, activation: 'relu' });
    this.userDenseOut = tf.layers.dense({ units: embDim, activation: null });

    // Item tower
    this.carEmbedding = tf.variable(
      tf.randomNormal([numCars, embDim], 0, 0.1)
    );
    this.brandEmbedding = tf.variable(
      tf.randomNormal([numBrands, embDim], 0, 0.1)
    );
    this.itemDense1 = tf.layers.dense({ units: hiddenDim, activation: 'relu' });
    this.itemDense2 = tf.layers.dense({ units: hiddenDim, activation: 'relu' });
    this.itemDenseOut = tf.layers.dense({ units: embDim, activation: null });
  }

  userForward(userIdx) {
    const emb = tf.gather(this.userEmbedding, userIdx);
    let x = this.userDense1.apply(emb);
    x = this.userDense2.apply(x);
    x = this.userDenseOut.apply(x);
    return tf.nn.l2Normalize(x, 1);
  }

  itemForward(carIdx, brandIdx) {
    const carEmb = tf.gather(this.carEmbedding, carIdx);
    const brandEmb = tf.gather(this.brandEmbedding, brandIdx);
    const concat = tf.concat([carEmb, brandEmb], 1);
    let x = this.itemDense1.apply(concat);
    x = this.itemDense2.apply(x);
    x = this.itemDenseOut.apply(x);
    return tf.nn.l2Normalize(x, 1);
  }

  score(userEmb, itemEmb) {
    return tf.sum(tf.mul(userEmb, itemEmb), 1, true); // (B,1)
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

      // In-batch softmax loss
      const logits = tf.concat([posScores, negScores], 1); // (B, 1 + B)
      const labels = tf.zeros([userIdx.shape[0]], 'int32');
      const loss = tf.losses.sparseCategoricalCrossentropy(labels, logits);

      return loss;
    });
  }

  async getUserEmbedding(userIdx) {
    return tf.tidy(() => {
      const u = tf.tensor1d([userIdx], 'int32');
      const emb = this.userForward(u);
      u.dispose();
      return emb;
    });
  }

  getScoresForAllItems(userEmb, allItemEmbs) {
    return tf.tidy(() => {
      const u = tf.expandDims(userEmb, 0); // (1, D)
      const scores = tf.matMul(u, allItemEmbs, false, true); // (1, N)
      return tf.squeeze(scores, [0]); // (N,)
    });
  }
}
