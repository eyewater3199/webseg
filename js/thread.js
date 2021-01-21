importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@2.7.0/dist/tf.min.js');

const FLOAT_255 = tf.scalar(255.0);
const MODEL_SIZE = 128;
const DATA_THRESHOLD = 0.7;
const HUMAN_RANGE_AVG_THRESHOLD = 0.1;

let model;  //  Tensorflowjs model

async function segmentate(imageData) {
    const expdim = tf.tidy(() => { 
        const resize = imageData.div(FLOAT_255).resizeBilinear([MODEL_SIZE, MODEL_SIZE]);
		return resize.expandDims(0);		
    });

    // Predict the model output
    const predict = await model.predict(expdim);

    if (predict) {        
        const predictData = predict.dataSync();
		const size = predict.size;
		const post_predictData = postProcessing(predictData, size);

        const predictShape = predict.shape;
        postMessage({array: post_predictData, shape: predictShape});
    }

    expdim.dispose();
    predict.dispose();
}

async function setup() {
	try {
        model = await tf.loadLayersModel("../db/kt_seg.json");   
        postMessage({ modelIsReady: true});
    } catch (err) {
        console.error("Can't load model: ", err);
        postMessage({ modelIsReady: false});
    }
}

function postProcessing(data, size) {
	var avg = 0;
	var total_value = 0;
	let postData = data;
	for(let i=0; i<size; i++) {
		if(data[i] > DATA_THRESHOLD) {
			total_value++;
		}
	}
	avg = total_value/size;

	if(avg < HUMAN_RANGE_AVG_THRESHOLD) {
		for(let j=0; j<size; j++) {
			postData[j] = 0;
		}
	}
	return postData;
}


setup();

onmessage = event => {
    if (model) {       
        const imgAsTensor = tf.tidy(() => {
            return tf.tensor(event.data.array, event.data.shape);
        });
        //console.time("predict");
        segmentate(imgAsTensor);
		imgAsTensor.dispose();
        //console.timeEnd("predict");
    }
}
