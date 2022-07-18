from flask import Flask, render_template, request
import time, string, random, cv2, os
import numpy as np

PEOPLE_FOLDER = os.path.join('static', 'uploads/')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER

@app.route('/')
def hello():
    return render_template('hello.html')

@app.route('/upload/do', methods=['POST'])
def upload_do():
    if request.method == 'POST':
        f = request.files['file']
        ts = str(time.time())
        letters = string.ascii_lowercase
        therandom = ''.join(random.choice(letters) for i in range(10))
        filenameNoExt = ts +'.'+ therandom
        extension = '.jpg'
        filename = filenameNoExt+extension
        full_filelocation = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        full_filelocation_origin = full_filelocation
        f.save(full_filelocation)

        image = cv2.imread(full_filelocation)

        # convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # reshape the image to a 2D array of pixels and 3 color values (RGB)
        pixel_values = image.reshape((-1, 3))

        # convert to float
        pixel_values = np.float32(pixel_values)

        # define stopping criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)

        # number of clusters (K)
        k = 5
        _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # convert back to 8 bit values
        centers = np.uint8(centers)

        # flatten the labels array
        labels = labels.flatten()

        # convert all pixels to the color of the centroids
        segmented_image = centers[labels.flatten()]

        # reshape back to the original image dimension
        segmented_image = segmented_image.reshape(image.shape)

        # disable only the cluster number 2 (turn the pixel into black)
        masked_image = np.copy(image)

        # convert to the shape of a vector of pixel values
        masked_image = masked_image.reshape((-1, 3))

        # color (i.e cluster) to disable
        cluster = 2
        masked_image[labels == cluster] = [0, 0, 0]

        # convert back to original shape
        masked_image = masked_image.reshape(image.shape)

        filename = filenameNoExt+'_generated'+extension
        full_filelocation = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        cv2.imwrite(full_filelocation,masked_image)
    return render_template('result.html', img=full_filelocation, img_origin=full_filelocation_origin)