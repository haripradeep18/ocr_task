import cProfile, os, re
import cv2
import pytesseract
import numpy as np
#import imutils
import argparse
from PIL import Image
import time
start_time = time.time()


dir_path_orig = os.path.dirname(os.path.realpath(__file__))
dir_path = dir_path_orig.replace("\\", "/")
#pytesseract.pytesseract.tesseract_cmd = dir_path+r'/softwares/Tesseract-OCR/tesseract.exe'

def join_images(list_of_images):
    images = [Image.open(x) for x in list_of_images]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    images = images[::-1]
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]

    new_im.save('out.jpg')
    joined_text = pytesseract.image_to_string(new_im, config='--psm 10 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    return (joined_text)

def read_each_image(image_name):
    # Load image, grayscale, Otsu's threshold
    image = cv2.imread(image_name)

    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    #image = imutils.resize(image, width=500)

    # Remove border
    kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1,50))
    temp1 = 255 - cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel_vertical)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50,1))
    temp2 = 255 - cv2.morphologyEx(image, cv2.MORPH_CLOSE, horizontal_kernel)
    temp3 = cv2.add(temp1, temp2)

    result = cv2.add(temp3, image)
    # Convert to grayscale and Otsu's threshold
    gray = cv2.cvtColor(temp3, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Find contours and filter using contour area
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    #sort contours
    cnts = sorted(cnts, key=lambda ctr: cv2.boundingRect(ctr)[1])
    cnts= cnts[::-1]
    list_of_images = []
    for i, c in enumerate(cnts):
        temp3 = cv2.medianBlur(temp3, 3)
        temp3[np.where(temp3 > [120])] = [255]

        x,y,w,h = cv2.boundingRect(c)
        try:
            cropped = temp3[y-9 :y +  h+9 , x-9 : x + w+9]
            s = dir_path+'/sub_images/_r_crop_' + str(i) + '.jpg' 
            cv2.imwrite(s , cropped)
        except:
            cropped = temp3[y :y + h , x : x + w]
            s = dir_path+'/sub_images/_r_crop_' + str(i) + '.jpg' 
            cv2.imwrite(s , cropped)

#         cv2.rectangle(temp3, (x-9, y-9), (x + w+9, y + h+9), (255,0, 255), 2)

        list_of_images.append(s)

    each_word = join_images(list_of_images)
    return each_word


def captch_ex(file_name, is_gruopby):
    img = cv2.imread(file_name)

    img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh,mask = cv2.threshold(img2gray,150,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU) 

    image_final = cv2.bitwise_and(img2gray, img2gray, mask=mask)
    ret, new_img = cv2.threshold(image_final, 180, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # cv.THRESH_BINARY_INV
   
    '''
            line  8 to 12  : Remove noisy portion 
    '''
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,
                                                         3))  # to manipulate the orientation of dilution , large x means horizonatally dilating  more, large y means vertically dilating more
    dilated = cv2.dilate(new_img, kernel, iterations=9)  # dilate , more the iteration more the dilation

    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # findContours returns 3 variables for getting contours

    words_list = []
    index = 1
    if not os.path.exists(dir_path+'/sub_images'): os.makedirs(dir_path+'/sub_images')
    if is_gruopby:
        
        for contour in contours:
            # get rectangle bounding contour
            [x, y, w, h] = cv2.boundingRect(contour)

            # Get plot positives that are text
            if w < 35 and h < 35:
                continue
            #you can crop image and send to OCR  , false detected will return no text :)
            cropped = new_img[y :y +  h , x : x + w]
            cv2.rectangle(new_img, (x, y), (x + w, y + h), (255, 0, 255), 2)
            s = dir_path+'/sub_images/new_img' + str(index) + '.jpg' 
            cv2.imwrite(s , cropped)
            image_text = pytesseract.image_to_string(cropped)
            index = index + 1

            return image_text
    else:   
        for contour in contours:
            # get rectangle bounding contour
            [x, y, w, h] = cv2.boundingRect(contour)

            # Get plot positives that are text
            if w < 55 and h > 35:
                # draw rectangle around contour on original image
    #             cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)

                #you can crop image and send to OCR  , false detected will return no text :)
                cropped = new_img[y :y +  h , x : x + w]

                s = dir_path+'/sub_images/first_crop_' + str(index) + '.jpg' 
                cv2.imwrite(s , cropped)
                word = read_each_image(s)
                words_list.append(word)
                index = index + 1
            
    return "---------------\n".join(words_list)


def read_text_from_image(file_name, is_truck=False, is_gruopby=False):
    if is_truck:
        image_text = captch_ex(file_name, is_gruopby)
    else:
        img = cv2.imread(file_name)
        #img = imutils.resize(img, width=500)

        img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh,mask = cv2.threshold(img2gray,150,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU) 

        image_final = cv2.bitwise_and(img2gray, img2gray, mask=mask)
        ret, new_img = cv2.threshold(image_final, 180, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # cv.THRESH_BINARY_INV

        image_text = pytesseract.image_to_string(new_img)
    print(image_text)

def str2bool(v):
	return v.lower() in ("yes", "true", "t", "1")
  
  
 
if __name__ == '__main__':
    pr = cProfile.Profile()
    pr.enable()
    ap = argparse.ArgumentParser()
    ap.add_argument("-image", "--image", type=str, help="path to input image")
    ap.add_argument("-truck", "--is_truck_image", type=str	, help="is input image contain truck image")
    ap.add_argument("-groupby", "--is_gruopby", type=str, help="is input image need to gruop by text content")
    args = vars(ap.parse_args())
    file_name = args["is_truck_image"]
    if not file_name: file_name = dir_path+"/images/161115046977_D_3.jpeg"
    print(file_name)
    is_truck_image = str2bool(args["is_truck_image"] or "")
    if not is_truck_image: is_truck_image=False
    is_gruopby=str2bool(args["is_gruopby"] or "")
    if not is_gruopby: is_gruopby=False
    read_text_from_image(file_name, is_truck_image, is_gruopby)
    pr.disable()
    #pr.print_stats()
    print("--- %s seconds ---" % (time.time() - start_time))
