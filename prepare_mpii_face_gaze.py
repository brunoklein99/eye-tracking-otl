import cv2
import numpy as np
import settings
from os import listdir
from os.path import join, isdir, split, splitext
import scipy.io

threshold = 10_000


def get_face_coordinates(image):
    height, width, _ = image.shape
    image = np.sum(image, axis=2)
    w_sum = np.sum(image, axis=0)
    h_sum = np.sum(image, axis=1)
    w_low = w_high = h_low = h_high = 0
    for i in range(len(w_sum)):
        if w_sum[i] > threshold:
            w_low = i
            break
    for i in reversed(range(len(w_sum))):
        if w_sum[i] > threshold:
            w_high = i
            break
    for i in range(len(h_sum)):
        if h_sum[i] > threshold:
            h_low = i
            break
    for i in reversed(range(len(h_sum))):
        if h_sum[i] > threshold:
            h_high = i
            break
    return w_low, w_high, h_low, h_high


if __name__ == '__main__':
    with open(join(settings.PREPARED_DIR, 'labels.csv'), 'w') as fd_labels:
        fd_labels.write('imagename,maskname,x,y\n')
        for participant_dir_name in listdir(settings.ORIGINAL_DIR):
            participant_dir_fullname = join(settings.ORIGINAL_DIR, participant_dir_name)
            if not isdir(participant_dir_fullname):
                continue
            screen_size = scipy.io.loadmat(join(participant_dir_fullname, 'Calibration/screenSize.mat'))
            labels_filename = join(participant_dir_fullname, '{}.txt'.format(participant_dir_name))
            with open(labels_filename) as f:
                for line in f:
                    tokens = line.split(' ')

                    image_rel_path = tokens[0]
                    image_fullname = join(participant_dir_fullname, image_rel_path)
                    day_name, image_filename = split(image_rel_path)
                    image_name, extension = splitext(image_filename)

                    image_new_name = join(settings.PREPARED_DIR,
                                          '{}_{}_{}'.format(participant_dir_name, day_name, image_filename))

                    print('started {}'.format(image_new_name))

                    p_x = int(tokens[1])
                    p_x /= screen_size['width_pixel'][0][0]
                    p_y = int(tokens[2])
                    p_y /= screen_size['height_pixel'][0][0]

                    img_original = cv2.imread(image_fullname)
                    w_low, w_high, h_low, h_high = get_face_coordinates(img_original)
                    img = img_original[h_low:h_high, w_low:w_high]
                    img = cv2.resize(img, dsize=(settings.IMAGE_SIZE, settings.IMAGE_SIZE))

                    image_new_name_mask = join(settings.PREPARED_DIR,
                                               '{}_{}_{}_mask{}'.format(participant_dir_name, day_name, image_name,
                                                                        extension))

                    cv2.imwrite(image_new_name, img)

                    img = np.sum(img_original, axis=2, keepdims=True)
                    img[img > 0] = 255
                    img = img.astype(dtype=np.uint8)
                    img = cv2.resize(img, dsize=(20, 20))
                    cv2.imwrite(image_new_name_mask, img)

                    fd_labels.write('{},{},{},{}\n'.format(image_new_name, image_new_name_mask, p_x, p_y))

                    print('done {}'.format(image_new_name))
