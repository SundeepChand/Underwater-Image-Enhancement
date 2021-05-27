import numpy as np
import cv2


def resize(img, perc=0.5):
    """Utility function to resize the image"""
    width = int(img.shape[1] * perc)
    height = int(img.shape[0] * perc)
    resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    return resized


def dark_channel(img, sz=15):
    """Calculates the dark channel in step 1"""
    b, g, r = cv2.split(img)
    dc = r - cv2.max(b, g)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
    dark = cv2.erode(dc, kernel)
    return dark


def airlight(img, dc):
    """Calculates the air-light in step 4"""
    [h, w] = img.shape[:2]
    imgsz = h * w
    imgvec = img.reshape(imgsz, 3)
    dcvec = dc.reshape(imgsz)

    indices = dcvec.argsort()
    return imgvec[indices[0]]


def transmission_estimate(dc):
    """Calculates the transmission map as mentioned in step 2"""
    te = dc + (1 - np.max(dc))
    return te


def guided_filter(img, p, r, eps):
    """Helper function to refine the transmission"""
    mean_I = cv2.boxFilter(img, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(img * p, cv2.CV_64F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(img * img, cv2.CV_64F, (r, r))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

    q = mean_a * img + mean_b
    return q


def transmission_refine(img, et):
    """Refines the transmission map as mentioned in step 3"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray) / 255
    r = 60
    eps = 0.0001
    t = guided_filter(gray, et, r, eps)
    return t


def recover(img, t, A):
    """Finally recovers the image using the image, the transmission map and air-light"""
    res = np.empty(img.shape, img.dtype)
    for index in range(0, 3):
        res[:, :, index] = (img[:, :, index] - A[index]) / t + A[index]

    return res


def normalize_image(img):
    """Utility function to normalize the intensities of the pixels in the image"""
    img = img - img.min()
    img = img / img.max() * 255
    img = np.uint8(img)
    return img


def dummy(x):
    pass


if __name__ == '__main__':
    filename = 'table.png'
    scale = 1

    src = cv2.imread(f'image/ocean_input/{filename}')
    src = resize(src, scale)

    img = src.astype('float64') / 255

    cv2.namedWindow('controls')
    cv2.createTrackbar('ksize', 'controls', 1, 600, dummy)

    while True:
        ksize = int(cv2.getTrackbarPos('ksize', 'controls')) + 1

        dc = dark_channel(img, ksize)
        te = transmission_estimate(dc)
        tr = transmission_refine(src, te)
        A = airlight(img, tr)
        result = recover(img, tr, A)
        result = normalize_image(result)

        cv2.imshow('Transmission estimate', te)
        cv2.imshow('Transmission refined', tr)
        cv2.imshow('Input', img)
        cv2.imshow('Result', result)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cv2.imwrite(f'image/ocean_output/{filename}', result)
