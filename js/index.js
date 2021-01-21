///////////////////////////////////////// UI //////////////////////////////////////
const RENDER_MODE = 1;
const FLOAT_255 = tf.scalar(255);
const OUTPUT_WIDTH = 300;
const OUTPUT_HEIGHT = 300;

let model;  //  Tensorflowjs model
let webcam;  // Webcam iterator
let bg;       // Background image
let input_frame;

let canvas = document.getElementById('mycanvas');
const ctx = canvas.getContext("2d");
let isPredicting = false;
let webWorker = null;
let isWaiting = false;
let workerModelIsReady = false;
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
        /*
		 const result = tf.tidy(() => {  
			const img_crop = tf.mul(image, mask);						
			const bgd_crop = bg.mul(tf.scalar(1.0).sub(mask));		
		    const add_result = tf.add(img_crop, bgd_crop);
			return add_result;
		 });

         return result;
         */
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
        return blending(refineMask(mask), img).div(FLOAT_255);
	});
}

async function runSegmentation() {
	 while (isPredicting && workerModelIsReady) {
         if (window.Worker) {           
            if (!isWaiting) {
                // Capture the frame from the webcam.
                console.time("time");
				input_frame = await webcam.capture();                
				
                // predict
                const frameData = await input_frame.dataSync();
				const frameShape = input_frame.shape;
                webWorker.postMessage({array: frameData, shape: frameShape});
                isWaiting = true;
            }  
		}

		//console.log('num tensor : ' + tf.memory().numTensors);
         
		 // Wait for next frame
		 await tf.nextFrame();
	 }
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
      return tf_image.resizeBilinear([OUTPUT_WIDTH, OUTPUT_HEIGHT]);    
    });
	//tf_image.dispose();
}

async function setupModel() {
    if (window.Worker) {
        webWorker = new Worker('https://cdn.jsdelivr.net/gh/eyewater3199/webseg@main/js/thread.js');
        // render
        webWorker.onmessage = event => {
            if (workerModelIsReady && isWaiting) {
				//console.time("draw");
                // Draw output on the canvas				
                const maskAsTensor = tf.tidy(() => {
                    return tf.tensor(event.data.array, event.data.shape);
                });
                
                // Post-process the output and blend images
                const output = process(maskAsTensor, input_frame);
                const output_frame = tensorToPixelData(output);
                ctx.putImageData(output_frame, 0, 0);            

				maskAsTensor.dispose();
				output.dispose();
				input_frame.dispose();

				isWaiting = false;
                //console.timeEnd("draw");
				console.timeEnd("time");
            }

            if (event.data.modelIsReady) {
                workerModelIsReady = true;                	     
            }
        };
    } else {
        try {
            model = await tf.loadLayersModel("../db/kt_seg.json"); 
        } catch (err) {
            console.error("Can't load model: ", err)
        }
    }    
}

async function init() {
    try {
        webcam = await tf.data.webcam(document.getElementById('webcam'));     
        bg = await loadImage(document.getElementById('bg'));                  
        document.getElementById('start').disabled = false;
        
        setupModel();
		
	 } catch (err) {
	  console.error(err);
	  //alert("No webcam found"); webcam 오류가 아닌 경우도 있으므로 팝업 오류는 제외하고 콘솔에서 확인
     }                
	 //const pred = model.predict(tf.zeros([1, MODEL_SIZE, MODEL_SIZE, 3]).toFloat());
	 //var readable_output = pred.dataSync();
	 //pred.dispose();
}

/* Initialize the application */
init();
