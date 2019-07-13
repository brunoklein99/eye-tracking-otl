import re
from argparse import ArgumentParser

import cv2
from os import listdir
from os.path import join, splitext

from face_extractor import extract_face


def listdir_ordered(dir):
    filenames = listdir(dir)
    filenames_split = [x.split('-') for x in filenames]
    code, _ = zip(*filenames_split)
    code = [int(x) for x in code]
    filenames = zip(code, filenames)
    filenames = sorted(filenames)
    code, filenames = zip(*filenames)
    return filenames


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-n', type=int, dest='index', help='index of environment dataset')
    args = parser.parse_args()

    custom_dir = 'data/custom{}'.format(args.index)
    custom_dir_prepared = custom_dir + '_prepared'
    with open(join(custom_dir_prepared, 'metadata.csv'), 'w') as f:
        f.write('imagename,x,y\n')
        for image_filename in listdir_ordered(custom_dir):
            print('starting {}'.format(image_filename))
            image_fullname = join(custom_dir, image_filename)
            imagename, _ = splitext(image_filename)
            code, resolution = imagename.split('-')
            x, y = resolution.split('x')
            x = int(x) / 1920
            y = int(y) / 1080
            img = cv2.imread(image_fullname)

            try:
                img = extract_face(img)
                img = cv2.resize(img, dsize=(448, 448))

                image_fullname = join(custom_dir_prepared, image_filename)

                cv2.imwrite(image_fullname, img)

                f.write('{},{},{}\n'.format(image_fullname, x, y))
                f.flush()
            except AssertionError as e:
                print('assertion error')
            except Exception as e:
                print(str(e))
