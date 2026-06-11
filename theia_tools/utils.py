import cv2
from pycocotools import mask as maskUtils

def read_image(image_path, segmentation, keep_bg=True):
    cvimage = cv2.imread(image_path)
    cvimage = cv2.cvtColor(cvimage, cv2.COLOR_BGR2RGB)

    rle = maskUtils.frPyObjects(segmentation, segmentation['size'][0], segmentation['size'][1])
    mask = maskUtils.decode(rle)

    if not keep_bg:
        cvimage = cv2.bitwise_and(cvimage, cvimage, mask=mask)
    x, y, w, h = cv2.boundingRect(mask)
    cvimage_cropped = cvimage[y:y+h, x:x+w]
    return cvimage_cropped

def read_image_w_mask(image_path, mask):
    cvimage = cv2.imread(image_path)
    cvimage = cv2.cvtColor(cvimage, cv2.COLOR_BGR2RGB)
    cvimage = cv2.bitwise_and(cvimage, cvimage, mask=mask)
    x, y, w, h = cv2.boundingRect(mask)
    cvimage_cropped = cvimage[y:y+h, x:x+w]
    return cvimage_cropped

def read_image_bbox(image_path, bbox_relative, keep_bg=True):
    cvimage = cv2.imread(image_path)
    cvimage = cv2.cvtColor(cvimage, cv2.COLOR_BGR2RGB)

    x, y, w, h = bbox_relative
    x, w = int(x * cvimage.shape[1]), int(w * cvimage.shape[1])
    y, h = int(y * cvimage.shape[0]), int(h * cvimage.shape[0])


    cvimage_cropped = cvimage[y:y+h, x:x+w]
    return cvimage_cropped

def read_image_default(image_path):
    cvimage = cv2.imread(image_path)
    cvimage = cv2.cvtColor(cvimage, cv2.COLOR_BGR2RGB)
    return cvimage
