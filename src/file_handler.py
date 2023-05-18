import cv2 as cv

supported_img_files = ["jpg", "png"]

def process_img_type(img, ext):
    if ext == "jpg":
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        return img
    elif ext == "png":
        img = img[:,:,:3]
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        return img

def load_img_file():
    img_path = input("Specify Img File Path: ")
    img = cv.imread(img_path, cv.IMREAD_UNCHANGED)
    img_extension = img_path.split(".")[-1]
    img = process_img_type(img, img_extension)
    return img
