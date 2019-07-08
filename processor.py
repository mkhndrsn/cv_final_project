from data_loaders import center_samples
from networks import get_fractal_network
import numpy as np
import cv2

models = {}


def get_fractal_model(image_shape):
    model = get_fractal_network(input_shape=image_shape, final_activation='sigmoid', training=False)
    model.load_weights('weights/recursive10_2018-04-22T2342.h5')
    return model


def get_non_overlapping_boxes(initial_boxes):
    def is_overlapping(box1, box2):
        if box1[0] < box2[2] and box1[2] > box2[0] and box1[1] < box2[3] and box1[3] > box2[1]:
            min_size = min((box1[2] - box1[0]) * (box1[3] - box1[1]), (box2[2] - box2[0]) * (box2[3] - box2[1]))
            overlap = (min(box1[2], box2[2]) - max(box1[0], box2[0])) * (min(box1[3], box2[3]) - max(box1[1], box2[1]))
            return overlap * 1.0 / min_size >= 0.5
        return False

    initial_boxes = sorted(initial_boxes, key=lambda x: x[1], reverse=True)

    boxes = []
    for i in range(len(initial_boxes)):
        cls, value, box = initial_boxes[i]
        add_box = True
        for c2, b2 in boxes:
            if is_overlapping(box, b2):
                add_box = False
                break
        if add_box:
            boxes.append((cls, box))
    return boxes


def process_image(bgr_image, save_model=False):
    orig_image = cv2.cvtColor(cv2.fastNlMeansDenoisingColored(bgr_image, None), cv2.COLOR_BGR2RGB)
    boxes = []
    for scale in [2.0, 1.0, 0.5]:
        image = np.asarray([cv2.resize(orig_image, None, fx=scale, fy=scale).astype('float')])
        # image = np.asarray([cv2.resize(orig_image, None, fx=scale, fy=scale).astype('float')[:,:,:]])
        if save_model:
            global models
            model = models.get(scale)
            if model is None:
                model = models.setdefault(scale, get_fractal_model(image.shape[1:]))
        else:
            model =  get_fractal_model(image.shape[1:])
        center_samples(image)
        output = model.predict(image, batch_size=128)
        for c in range(output.shape[0]):
            results = output[c]
            step_size_i = (image.shape[1] - 32) * 1.0 / results.shape[0]
            step_size_j = (image.shape[2] - 32) * 1.0 / results.shape[1]
            index = 0
            # results = np.where(np.max(results, axis=-1) > 0.997, np.argmax(results, axis=-1), 10)
            for i in range(results.shape[0]):
                for j in range(results.shape[1]):
                    for k in range(results.shape[2]):
                        if results[i][j][k] > 0.935 + 0.03 * scale:
                            y = int(i * step_size_i / scale)
                            x = int(j * step_size_j / scale)
                            size = int(32 / scale)
                            boxes.append((k, results[i][j][k] - 0.03 * scale, [x, y, x + size, y + size]))
    boxes = get_non_overlapping_boxes(boxes)
    for b in boxes:
        cv2.rectangle(bgr_image, (b[1][0], b[1][1]), (b[1][2], b[1][3]), (0, 0, 255), 2)
        cv2.putText(bgr_image, str(b[0]), (b[1][0], b[1][1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 3)
        cv2.putText(bgr_image, str(b[0]), (b[1][0] + 1, b[1][1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1)

    return bgr_image