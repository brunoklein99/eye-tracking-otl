from collections import deque

import cv2
import numpy as np
from os import listdir
from os.path import join, isdir, splitext, isfile

from scipy.io import loadmat

from face_extractor import extract_face


def get_participant_screen_size(participant_dir_fullname):
    screen_size_file = join(participant_dir_fullname, 'Calibration', 'screenSize.mat')
    screen_size = loadmat(screen_size_file)
    screen_w = int(np.squeeze(screen_size['width_pixel']))
    screen_h = int(np.squeeze(screen_size['height_pixel']))
    return screen_w, screen_h


if __name__ == '__main__':
    mpii_dir = 'data/mpii'
    mpii_dir_prepared = mpii_dir + '_prepared'
    mpii_metadata_fullname = join(mpii_dir_prepared, 'metadata.csv')
    error_count = 0
    with open(mpii_metadata_fullname, 'w') as f:
        f.write('imagename,x,y\n')
        for participant_dir in listdir(mpii_dir):
            participant_dir_fullname = join(mpii_dir, participant_dir)
            if not isdir(participant_dir_fullname):
                continue
            screen_w, screen_h = get_participant_screen_size(participant_dir_fullname)
            labels_fullname = join(participant_dir_fullname, '{}.txt'.format(participant_dir))
            with open(labels_fullname) as flabel:
                for line in flabel:
                    # noinspection PyRedeclaration
                    image_filename, *tail = deque(line.split())
                    image_fullname = join(participant_dir_fullname, image_filename)

                    # noinspection PyRedeclaration
                    day, imagename = image_filename.split('/')
                    imagename, _ = splitext(imagename)

                    imagename_new = '{}_{}_{}.jpg'.format(participant_dir, day, imagename)
                    imagename_new = join(mpii_dir_prepared, imagename_new)

                    if isfile(imagename_new):
                        print('skipping', imagename_new)
                        continue

                    target_x, *tail = tail
                    target_y, *tail = tail
                    target_x = int(target_x)
                    target_y = int(target_y)
                    target_x /= screen_w
                    target_y /= screen_h

                    print('starting participant {} day {} img {}'.format(participant_dir, day, imagename))

                    try:
                        # noinspection PyRedeclaration
                        img = cv2.imread(image_fullname)
                        img = extract_face(img)
                        img = cv2.resize(img, dsize=(448, 448))

                        cv2.imwrite(imagename_new, img)

                        f.write('{},{},{}\n'.format(imagename_new, target_x, target_y))
                        f.flush()
                    except AssertionError as e:
                        print('assert error')
                        error_count += 1
                    except Exception as e:
                        print(str(e))
                        error_count += 1
    print('error count', error_count)
