from skimage.filters import gaussian_filter
import cv2

def blurImage(filename, sigma):
    im = cv2.imread(filename)
    return gaussian_filter(im, sigma=sigma, multichannel=True)

if __name__ == "__main__":
    filename = './unsplashFilter.jpg'
    cv2.imwrite('./karan.jpg', blurImage(filename, sigma=7))
