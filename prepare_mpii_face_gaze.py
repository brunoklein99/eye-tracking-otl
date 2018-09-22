import cv2
import numpy as np
import settings
from os import listdir
from os.path import join, isdir, split, splitext
import scipy.io

original_dir = 'data/mpii_face_gaze_original'
prepared_dir = 'data/mpii_face_gaze_prepared'

if __name__ == '__main__':
    with open(join(prepared_dir, 'labels.csv'), 'w') as fd_labels:
        for participant_dir_name in listdir(original_dir):
            participant_dir_fullname = join(original_dir, participant_dir_name)
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

                    p_x = int(tokens[1])
                    p_x /= screen_size['width_pixel'][0][0]
                    p_y = int(tokens[2])
                    p_y /= screen_size['height_pixel'][0][0]

                    img = cv2.imread(image_fullname)
                    img = cv2.resize(img, dsize=(settings.IMAGE_SIZE, settings.IMAGE_SIZE))

                    image_new_name = join(prepared_dir, '{}_{}_{}'.format(participant_dir_name, day_name, image_filename))
                    image_new_name_mask = join(prepared_dir, '{}_{}_{}_mask{}'.format(participant_dir_name, day_name, image_name, extension))

                    cv2.imwrite(image_new_name, img)

                    img = np.sum(img, axis=2, keepdims=True)
                    img[img > 0] = 255
                    img = img.astype(dtype=np.uint8)
                    img = cv2.resize(img, dsize=(20, 20))

                    cv2.imwrite(image_new_name_mask, img)

                    fd_labels.write('{};{};{};{};\n'.format(image_new_name, image_new_name_mask, p_x, p_y))

                    print('done {}'.format(image_new_name))
