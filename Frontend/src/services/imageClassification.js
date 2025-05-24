import * as tf from '@tensorflow/tfjs';

// Load the model
let model;
export async function loadModel() {
  if (!model) {
    model = await tf.loadGraphModel('https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v2_100_224/classification/4/default/1');
  }
  return model;
}

// Preprocess image
export async function preprocessImage(imageElement) {
  return tf.tidy(() => {
    // Convert the image to a tensor
    const tensor = tf.browser.fromPixels(imageElement)
      .resizeNearestNeighbor([224, 224])
      .toFloat()
      .expandDims();

    // Normalize the image
    const offset = tf.scalar(127.5);
    return tensor.sub(offset).div(offset);
  });
}

// Classify image
export async function classifyImage(imageElement) {
  try {
    const model = await loadModel();
    const processedImage = await preprocessImage(imageElement);
    
    // Get predictions
    const predictions = await model.predict(processedImage);
    const values = await predictions.data();
    
    // Get top 5 predictions
    const top5 = Array.from(values)
      .map((value, index) => ({ value, index }))
      .sort((a, b) => b.value - a.value)
      .slice(0, 5);
    
    return top5;
  } catch (error) {
    console.error('Error classifying image:', error);
    throw error;
  }
}

// Simple road/not road classification
export async function isRoad(imageElement) {
  try {
    const predictions = await classifyImage(imageElement);
    // This is a simplified check - in a real application, you'd want to use a model
    // specifically trained for road detection
    const roadRelatedClasses = ['road', 'highway', 'street', 'pavement', 'sidewalk'];
    const isRoad = predictions.some(pred => 
      roadRelatedClasses.some(roadClass => 
        pred.index.toString().includes(roadClass)
      )
    );
    return isRoad;
  } catch (error) {
    console.error('Error checking if image is road:', error);
    throw error;
  }
} 