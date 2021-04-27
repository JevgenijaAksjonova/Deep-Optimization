import numpy as np
import odl
import os

MAYO_FOLDER = ""

# generate data with normal 5% noise
def generate_transform_mayo(images,operator):
    
    data = []
    for image in images:
        transformed = operator(image.squeeze()).asarray()
        noisy = (transformed + 
                 0.05 * np.random.randn(*operator.range.shape) * np.mean(np.abs(transformed)))
        data.append(noisy)
    return np.array(data)[..., None]
            
# generates pairs (image, data)
# mode - ["train", "validate", "test"]
def generate_data(operator, mode, batch_size, val_ratio=0.):
    shape = (batch_size, 512, 512, 1)
    folder = MAYO_FOLDER
    test = 'L286'
    data = []
    if mode == 'test':
        for (dirpath, dirnames, filenames) in os.walk(folder):
            data.extend([os.path.join(folder, fi) for fi in filenames if fi.startswith(test)])
        data = np.sort(data)
    else:
        # mode is "train" or "validate"
        for (dirpath, dirnames, filenames) in os.walk(folder):
            data.extend([os.path.join(folder, fi) for fi in filenames if not fi.startswith(test)])
        n_val = int(val_ratio * len(data))
        # the same images are uniformly selected for validation
        data = np.sort(data)
        step = (len(data) + n_val - 1) // n_val
        if mode == 'validate':
            data = [data[i] for i in range(len(data)) if i % step == 0]
        else:
            data = [data[i] for i in range(len(data)) if i % step != 0]
    n_images = len(data)
    n_batches = int(n_images / batch_size)
    while True:
        if mode == 'train':
            np.random.shuffle(data)
        for i in range(n_batches):
            filenames = data[i * batch_size:(i + 1) * batch_size]
            images = []
            for fn in filenames:
                image = np.load(fn)
                image = image / 1000.0
                #image = np.rot90(image, -1)
                images.append(image)
            x = np.stack(images, axis = 0).reshape(shape)
            y = generate_transform_mayo(images, operator)
            yield((x, y))
        # training images are sampled infinitely
        if mode != 'train':
            break
