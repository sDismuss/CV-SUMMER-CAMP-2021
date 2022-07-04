from email.mime import image
import re
import sys
import argparse
import cv2 as cv


def make_cat_passport_image(input_image_path, haar_model_path):

    # Read image
    image = cv.imread(input_image_path)

    # Convert image to grayscale
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Normalize image intensity
    norm_image = cv.equalizeHist(gray_image)

    # Resize image
    resized_image = cv.resize(norm_image, (640, 480), interpolation = cv.INTER_AREA)
    image = cv.resize(image, (640, 480), interpolation = cv.INTER_AREA)

    # Detect cat faces using Haar Cascade
    detector = cv.CascadeClassifier(haar_model_path)
    rects = detector.detectMultiScale(resized_image, scaleFactor=1.1, minNeighbors=5, minSize=(75, 75))
    #print(rects)

    # Draw bounding box
    for (i, (x, y, w, h)) in enumerate(rects): 
        cv.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2) 
        cv.putText(image, "Cat #{}".format(i + 1), (x, y - 10), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    # Display result image
    # cv.imshow("window_name", image) 
    # cv.waitKey(0) 
    # cv.destroyAllWindows()

    # Crop image
    x, y, w, h = rects[0] 
    image = image[y:y+h, x:x+w]

    # Save result image to file
    cv.imwrite('out.jpg', image)

    print("Finished successfully!")
    return


def build_argparser():
    parser = argparse.ArgumentParser(
        description='Speech denoising demo', add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                      help='Show this help message and exit.')
    args.add_argument('-m', '--model', type=str, required=True,
                      help='Required. Path to .XML file with pre-trained model.')
    args.add_argument('-i', '--input', type=str, required=True,
                      help='Required. Path to input image')
    return parser


def main():
    
    args = build_argparser().parse_args()
    make_cat_passport_image(args.input, args.model)

    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
