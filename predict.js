/**
 * Get the car data reduced to just the variables we are interested
 * and cleaned of missing data.
 */
let data;
let pred;
function dist(ax, ay, bx, by) {
  return Math.sqrt(Math.pow((ax - bx), 2) + Math.pow((ay - by), 2));
}
async function getData() {
    const DataReq = await fetch('data/sj.json');
    const Data = await DataReq.json();  
    const cleaned = Data.map(pose => ({
      conf: pose.confidence,
      left_eye: dist(pose.noseX, pose.noseY, pose.left_eyeX, pose.left_eyeY),
      right_eye: dist(pose.noseX, pose.noseY, pose.right_eyeX, pose.right_eyeY),
      left_ear: dist(pose.noseX, pose.noseY, pose.left_earX, pose.left_earY),
      right_ear: dist(pose.noseX, pose.noseY, pose.right_earX, pose.right_earY),
      left_shoulder: dist(pose.noseX, pose.noseY, pose.left_shouldX, pose.left_shouldY),
      right_shoulder: dist(pose.noseX, pose.noseY, pose.right_shouldX, pose.right_shouldY),
      left_elbow: dist(pose.noseX, pose.noseY, pose.left_elbowX, pose.left_elbowY),
      right_elbow: dist(pose.noseX, pose.noseY, pose.right_elbowX, pose.right_elbowY),
      left_wrist: dist(pose.noseX, pose.noseY, pose.left_wristX, pose.left_wristY),
      right_wrist: dist(pose.noseX, pose.noseY, pose.right_wristX, pose.right_wristY),
      focused: pose.focused,
    }))
    .filter(pose => (pose.conf != null));
    
    return cleaned;
  }

async function getTestData() {
  const DataReq = await fetch('data/cjtest.json');
  const Data = await DataReq.json();  
  const cleaned = Data.map(pose => ({
    conf: pose.confidence,
    left_eye: dist(pose.noseX, pose.noseY, pose.left_eyeX, pose.left_eyeY),
    right_eye: dist(pose.noseX, pose.noseY, pose.right_eyeX, pose.right_eyeY),
    left_ear: dist(pose.noseX, pose.noseY, pose.left_earX, pose.left_earY),
    right_ear: dist(pose.noseX, pose.noseY, pose.right_earX, pose.right_earY),
    left_shoulder: dist(pose.noseX, pose.noseY, pose.left_shouldX, pose.left_shouldY),
    right_shoulder: dist(pose.noseX, pose.noseY, pose.right_shouldX, pose.right_shouldY),
    left_elbow: dist(pose.noseX, pose.noseY, pose.left_elbowX, pose.left_elbowY),
    right_elbow: dist(pose.noseX, pose.noseY, pose.right_elbowX, pose.right_elbowY),
    left_wrist: dist(pose.noseX, pose.noseY, pose.left_wristX, pose.left_wristY),
    right_wrist: dist(pose.noseX, pose.noseY, pose.right_wristX, pose.right_wristY),
    focused: pose.focused,
  }))
  .filter(pose => (pose.conf != null));
  
  return cleaned;
}

async function run() {
  // Load and plot the original input data that we are going to train on.
  data = await getData();
  test = await getTestData();
  console.log('data loaded');
  /*
  const values = data.map(d => ({
    x: d.horsepower,
    y: d.mpg,
  }));

  tfvis.render.scatterplot(
    {name: 'Horsepower v MPG'},
    {values}, 
    {
      xLabel: 'Horsepower',
      yLabel: 'MPG',
      height: 300
    }
  );
  */
  
  // Create the model
  const model = createModel();  
  tfvis.show.modelSummary({name: 'Model Summary'}, model);

  // Convert the data to a form we can use for training.
  const tensorData = convertToTensor(data);
  const {inputs, labels} = tensorData;

  const testTensorData = convertToTensor(test);
  const {testin, testla} = testTensorData;
  console.log(testTensorData);

  //const testTensorData = convertToTensor(test);
  // Train the model  
  await trainModel(model, inputs, labels);
  console.log('Done Training');
  await model.save('downloads://my-model');
  testModel(model, testin, testTensorData);
}

document.addEventListener('DOMContentLoaded', run);

function createModel() {
    // Create a sequential model
    const model = tf.sequential(); 
    
    // Add a single hidden layer
    model.add(tf.layers.dense({inputShape: [11,], units: 16, useBias: true, activation: 'relu6'}));
    
    model.add(tf.layers.dense({units: 16, useBias: true, activation: 'relu6'}));

    // Add an output layer
    model.add(tf.layers.dense({units: 1, useBias: true, activation: 'sigmoid'}));
  
    return model;
  }

/**
 * Convert the input data to a tensors that we can use for machine 
 * learning. We will also do the important best practices of _shuffling_
 * the data and _normalizing_ the data
 * MPG on the y-axis.
 */
function convertToTensor(data) {
    // Wrapping these calculations in a tidy will dispose any 
    // intermediate tensors.
    
    return tf.tidy(() => {
      // Step 1. Shuffle the data    
      tf.util.shuffle(data);
      // Step 2. Convert data to Tensor
      const inputs = data.map(d => Object.values(d).slice(0,11));
      const labels = data.map(d => Object.values(d).slice(11,12));
  
      const inputTensor = tf.tensor2d(inputs, [inputs.length, 11]);
      const labelTensor = tf.tensor2d(labels, [labels.length, 1]);
  
      //Step 3. Normalize the data to the range 0 - 1 using min-max scaling
      const inputMax = inputTensor.max();
      const inputMin = inputTensor.min();  
      const labelMax = labelTensor.max();
      const labelMin = labelTensor.min();
  
      const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
      const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));
  
      return {
        inputs: normalizedInputs,
        labels: normalizedLabels,
        //inputs: inputTensor,
        //labels: labelTensor,
        // Return the min/max bounds so we can use them later.
        //inputMax,
        //inputMin,
        //labelMax,
        //labelMin,
      }
    });  
  }
async function trainModel(model, inputs, labels) {
    // Prepare the model for training.  
    model.compile({
        optimizer: tf.train.adam(),
        loss: 'binaryCrossentropy',
        //loss: 'tf.losses.meanSquaredError',
        metrics: ['mse'],
    });

    const batchSize = 20;
    const epochs = 20;

    return await model.fit(inputs, labels, {
        batchSize,
        epochs,
        shuffle: true,
        callbacks: tfvis.show.fitCallbacks(
        { name: 'Training Performance' },
        ['loss', 'mse'], 
        { height: 200, callbacks: ['onEpochEnd'] }
        )
    });
}

function testModel(model, inputData, normalizationData) {
    //const {inputMax, inputMin, labelMin, labelMax} = normalizationData;  
    
    // Generate predictions for a uniform range of numbers between 0 and 1;
    // We un-normalize the data by doing the inverse of the min-max scaling 
    // that we did earlier.
    /*
    const [xs, preds] = tf.tidy(() => {
      
      const xs = tf.linspace(0, 1, 100);      
      const preds = model.predict(xs.reshape([100, 1]));      
      
      const unNormXs = xs
        .mul(inputMax.sub(inputMin))
        .add(inputMin);
      
      const unNormPreds = preds
        .mul(labelMax.sub(labelMin))
        .add(labelMin);
      
      // Un-normalize the data
      return [unNormXs.dataSync(), unNormPreds.dataSync()];
    });
    */
    //pred = model.predict(inputData);
    pred = model.predict(tf.randomNormal([20, 11]));
    console.log(pred)
   
    /*
    const predictedPoints = Array.from(xs).map((val, i) => {
      return {x: val, y: preds[i]}
    });
    
    const originalPoints = inputData.map(d => ({
      x: d.horsepower, y: d.mpg,
    }));
    
    tfvis.render.scatterplot(
      {name: 'Model Predictions vs Original Data'}, 
      {values: [originalPoints, predictedPoints], series: ['original', 'predicted']}, 
      {
        xLabel: 'Horsepower',
        yLabel: 'MPG',
        height: 300
      }
    );
    */
  }