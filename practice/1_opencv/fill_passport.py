import sys
import os.path
from turtle import width
import cv2 as cv
import argparse

def make_cat_passport_image(input_image_path, cat_image_file_path):

    # Read images
    template = cv.imread(input_image_path)
    cat_image = cv.imread(cat_image_file_path)

    # Resize image
    resized_cat_image = cv.resize(cat_image, (172, 140), interpolation = cv.INTER_AREA)

    # Place cat image inside passport template
    width, height, _ = resized_cat_image.shape
    x_corner, y_corner = 47, 30
    template [x_corner:x_corner+width, y_corner:y_corner+height] = resized_cat_image

    # Place information in the passport
    template = cv.putText(template, 'Space', (90, 220), cv.FONT_HERSHEY_SIMPLEX, 
                   0.4, (0, 0, 0), 1, cv.LINE_AA)
    template = cv.putText(template, '05.18.2012', (120, 273), cv.FONT_HERSHEY_SIMPLEX, 
                   0.3, (0, 0, 0), 1, cv.LINE_AA)

    # Display result image
    cv.imshow("window_name", template) 
    cv.waitKey(0) 
    cv.destroyAllWindows()
    
    # Save result image to file
    cv.imwrite('final_passport.jpg', template)

    print("Finished successfully!")
    return


def build_argparser():
    parser = argparse.ArgumentParser(
        description='Speech denoising demo', add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                      help='Show this help message and exit.')
    args.add_argument('-i', '--input', type=str, required=True,
                      help='Required. Path to input image')
    return parser


def main():
    
    args = build_argparser().parse_args()

    cat_image_file_path = "out.jpg"
    if not os.path.exists(cat_image_file_path):
        print('Before launching this program, please launch cat_passport.py')
        exit(1)

    make_cat_passport_image(args.input, cat_image_file_path)

    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
