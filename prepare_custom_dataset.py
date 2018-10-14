import re
import cv2
from os import listdir
from os.path import join, splitext

from face_extractor import extract_face

if __name__ == '__main__':
    custom_dir = 'data/custom'
    custom_dir_prepared = custom_dir + '_prepared'
    regex = re.compile(pattern='([0-9]+)x([0-9]+)')
    with open(join(custom_dir_prepared, 'metadata.csv'), 'w') as f:
        f.write('imagename,x,y\n')
        for image_filename in listdir(custom_dir):

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
