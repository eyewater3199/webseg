///////////////////////////////////////// UI //////////////////////////////////////
const RENDER_MODE = 1;
const FLOAT_255 = tf.scalar(255);
const OUTPUT_WIDTH = 300;
const OUTPUT_HEIGHT = 300;
const MODEL_SIZE = 128;
const DATA_THRESHOLD = 0.7;
const HUMAN_RANGE_AVG_THRESHOLD = 0.1;

let webcam;  // Webcam iterator
let model;  //  Tensorflowjs model
let bg;       // Background image
let canvas = document.getElementById('mycanvas');
const ctx = canvas.getContext("2d");
let isPredicting = false;
let imageData = null;

//Gaussian kernel of size (7,7)
const kernel = tf.tensor4d([0.00092991, 0.00223073, 0.00416755, 0.00606375, 0.00687113, 0.00606375,
    0.00416755, 0.00223073, 0.00535124, 0.00999743, 0.01454618, 0.01648298,
    0.01454618, 0.00999743, 0.00416755, 0.00999743, 0.01867766, 0.02717584,
    0.03079426, 0.02717584, 0.01867766, 0.00606375, 0.01454618, 0.02717584,
    0.03954061, 0.04480539, 0.03954061, 0.02717584, 0.00687113, 0.01648298,
    0.03079426, 0.04480539, 0.05077116, 0.04480539, 0.03079426, 0.00606375,
    0.01454618, 0.02717584, 0.03954061, 0.04480539, 0.03954061, 0.02717584,
    0.00416755, 0.00999743, 0.01867766, 0.02717584, 0.03079426, 0.02717584,
    0.01867766
   ], [7, 7, 1, 1]);

document.getElementById('start').addEventListener('click', async () => {
	isPredicting = true;
	document.getElementById('start').disabled = true;
	document.getElementById('stop').disabled = false;

	runSegmentation();
});

document.getElementById('stop').addEventListener('click', () => {
	isPredicting = false;
	document.getElementById('start').disabled = false;
	document.getElementById('stop').disabled = true;
});

function blending(mask, image) {
    return tf.tidy(() => {             
        return bg.add(mask.mul(image.sub(bg)));
	});
}    

/* Smooth the mask edges */
function smoothstep(x) {
	return tf.tidy(() => {
		 // Define the left and right edges 
		 const edge0 = tf.scalar(0.3);
		 const edge1 = tf.scalar(0.5);

		 // Scale, bias and saturate x to 0..1 range
		 const z = tf.clipByValue(x.sub(edge0).div(edge1.sub(edge0)), 0.0, 1.0);
		 
		 //Evaluate polynomial  z * z * (3 - 2 * x)
		 return tf.square(z).mul(tf.scalar(3).sub(z.mul(tf.scalar(2))));
	});
}

// Perform mask feathering (Gaussian-blurring + Egde-smoothing)
function refineMask(mask) {
	return tf.tidy(() => {
		  // Threshold the output to obtain mask 
         const resize = mask.resizeBilinear([OUTPUT_WIDTH, OUTPUT_HEIGHT]);            
		 // Reshape input
		 const reshape = resize.reshape([1, OUTPUT_WIDTH, OUTPUT_HEIGHT, 1]);

		
		 // Convolve the mask with kernel   
		 const blurred = tf.conv2d(reshape, kernel, strides = [1, 1], padding = 'same');		 
		 const fb = blurred.squeeze(0); 
		 
	     const norm = fb.sub(fb.min()).div(fb.max().sub(fb.min()))
         const thresh = tf.scalar(0.7);

		 var binarized;
		 if(RENDER_MODE == 1) {
			 if(fb.min() && fb.max()) {
				binarized = fb.greater(thresh);
			 }else{
				binarized = norm.greater(thresh);
			 }
			 return smoothstep(binarized);   
		 }else if(RENDER_MODE == 2) {
			 return fb;
		 }
         
	});
}

function process(mask, img) {
	return tf.tidy(() => {             
        return blending(refineMask(mask), img);
	});
}

function segmentate(imageData) {
    return tf.tidy(() => { 
        const resize = imageData.resizeBilinear([MODEL_SIZE, MODEL_SIZE]);
        const expdims = resize.expandDims(0);
		return model.predict(expdims);		
    });
}


async function runSegmentation() {
	 while (isPredicting) {
         // Capture the frame from the webcam.
         console.time("time");
         const input_frame = await webcam.capture();     
         const input_frame_div = input_frame.div(FLOAT_255);
        
         const mask = segmentate(input_frame_div);
		 const maskAsTensor = tf.tensor(postProcessing(mask.dataSync(), mask.size), mask.shape);

         // Post-process the output and blend images
         const output = process(maskAsTensor, input_frame_div);
         const output_frame = tensorToPixelData(output);
         ctx.putImageData(output_frame, 0, 0);            

		 mask.dispose();
         maskAsTensor.dispose();
         output.dispose();
         input_frame.dispose();
         input_frame_div.dispose();

         //console.timeEnd("draw");
         console.timeEnd("time");
         console.log('num tensor : ' + tf.memory().numTensors);

		 // Wait for next frame
		 await tf.nextFrame();
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

function tensorToPixelData(tensor) {
    if (imageData == null) {
        imageData = new ImageData(tensor.shape[0], tensor.shape[1]);
    } else {
        const pixels = tensor.dataSync();
        const len = tensor.size;
        for(let i=0; i<len/3; i++) {
            imageData.data[i*4+0] = (pixels[i*3+0]) * 255;
            imageData.data[i*4+1] = (pixels[i*3+1]) * 255;
            imageData.data[i*4+2] = (pixels[i*3+2]) * 255;
            imageData.data[i*4+3] = 255;
        }
    }
    tensor.dispose();
	return imageData;
}

async function loadImage(src) {
    const tf_image = await tf.browser.fromPixels(src);
    return tf.tidy(() => {
      return tf_image.div(FLOAT_255).resizeBilinear([OUTPUT_WIDTH, OUTPUT_HEIGHT]);    
    });
}

async function init() {
	 try {
        webcam = await tf.data.webcam(document.getElementById('webcam'));     
	    model = await tf.loadLayersModel("./db/kt_seg.json"); 
		bg = await loadImage(document.getElementById('bg')); 
		document.getElementById('start').disabled = false;
	 } catch (e) {
	  console.log(e);
	  //alert("No webcam found"); webcam 오류가 아닌 경우도 있으므로 팝업 오류는 제외하고 콘솔에서 확인
	 }
	
	 //const pred = model.predict(tf.zeros([1, MODEL_SIZE, MODEL_SIZE, 3]).toFloat());
	 //var readable_output = pred.dataSync();
	 //pred.dispose();
}

/* Initialize the application */
init();
