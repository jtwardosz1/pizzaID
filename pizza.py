#This is my final code. I have it set up to test pepperoni first. It goest through all the methods to test for different pizzas and each method has output images
# to show the isolated topping colors so be aware each pizza will give out quite a few output images unless commented out in main. The output in the terminal will show what pizza 
# was detected. Thanks! 
import cv2 as cv
import math
import numpy as np
from matplotlib import pyplot
 
#read image into pizza. To test different pizzas, choose from pizza images sent along with this file (pep.jpg, buffalo.jpg, cheese.jpg, mac.jpg, pomo.jpg)
# pizza = cv.imread('pep.jpg')
# pizza = cv.imread('buffalo.jpg')
# pizza = cv.imread('cheese.jpg')
# pizza = cv.imread('mac.jpg')
pizza = cv.imread('pomo.jpg')

#create rgb image to show original image
original = cv.cvtColor(pizza, cv.COLOR_BGR2RGB)

pyplot.figure()
pyplot.title('Original image')
pyplot.imshow(original)
pyplot.show()

#create gray image to use specifically for circle detection for pepperoni pizza 
grayPizza = cv.cvtColor(pizza, cv.COLOR_BGR2RGB)
grayPizza = cv.cvtColor(grayPizza, cv.COLOR_RGB2GRAY)

#convert pizza image to HSV
pizza = cv.cvtColor(pizza, cv.COLOR_BGR2HSV)


 
def macAndCheese(pizza):
    
    mac = pizza.copy()
    
    #isolating yellow hsv values and creating mask
    lower_yellow = np.array([15, 160, 150])
    upper_yellow = np.array([25, 230, 255])
    macMask = cv.inRange(mac, lower_yellow, upper_yellow)
    
    #create value for mac using sum of mask
    isMac = np.sum(macMask)
    
    mac[np.where(macMask==0)] = 0

    mac_output_img = cv.cvtColor(mac, cv.COLOR_HSV2RGB)

    if isMac > 750000000:
        print("This is a Mac and Cheese Pizza")
        
    pyplot.figure()
    pyplot.title('Mac Output image')
    pyplot.imshow(mac_output_img)
    pyplot.show()
    

def pepperoni(pizza, grayPizza):
    
    pep = pizza.copy()
    
    #this code chunk was taken from Derricw on Stackoverflow to create a red mask for hsv
    #lower red mask
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    lower_mask = cv.inRange(pep, lower_red, upper_red)

    #upper red mask
    lower_red = np.array([170, 50, 50])
    upper_red = np.array([180, 255, 255])
    upper_mask = cv.inRange(pep, lower_red, upper_red)

    #combine masks to cover all red hsv values
    mask = lower_mask + upper_mask

    #create value to check for pepperoni by finding sum of mask
    isPepperoni = np.sum(mask)

    #turn image to just show red
    pep[np.where(mask==0)] = 0

    output_img = cv.cvtColor(pep, cv.COLOR_HSV2RGB)

    #kernels for morphology on pepperonis
    kernel = np.ones((15, 15), np.uint8)

    kernel2 = np.ones((35, 35), np.uint8)

    output_img = cv.morphologyEx(output_img, cv.MORPH_OPEN, kernel)

    output_img = cv.morphologyEx(output_img, cv.MORPH_CLOSE, kernel2)

    pyplot.figure()
    pyplot.title('Pep Output image')
    pyplot.imshow(output_img)
    
    pyplot.show()

    output_img = cv.cvtColor(output_img, cv.COLOR_RGB2GRAY)

    #scale image down for easier circle detection
    scale_percent = 25

    width1 = int(output_img.shape[1] * scale_percent / 100)
    height1 = int(output_img.shape[0] * scale_percent / 100)
    dim = (width1, height1)

    output_img = cv.resize(output_img, dim, interpolation = cv.INTER_AREA)

    #blur image for easier circle detection
    output_img = cv.blur(output_img, (3,3))

                
    #hough circles function used to detect circles - Used similar code from Using hough circles from pyimagesearch.com article by Adrian Rosebrock.         
    circles = cv.HoughCircles(output_img, cv.HOUGH_GRADIENT, 1.2, 55, param1 = 208,
                param2 = 10.5, minRadius = 28, maxRadius = 33)

    if circles is not None:
        
        circles = np.round(circles[0, :]).astype("int")
        
        for (x, y, r) in circles:
            
            cv.circle(output_img, (x,y), r, (255, 0, 0), 1)
            
        output = output_img

        # print(len(circles))
        
        # pyplot.figure()
        
        pyplot.title('circle')
        pyplot.imshow(output)
    

    #check if image has property amount of pepperoni color and number of pepperonis
    if isPepperoni > 720000000 and len(circles) > 40:
        
        print("This is Pepperoni Pizza!")
        
    pyplot.show()
    


    
def buffalo(pizza):
    
    buff = pizza.copy()
    buff1 = pizza.copy()
     
    lower_bleu = np.array([5, 1, 135])
    upper_bleu = np.array([25, 84, 225])
    bleuMask = cv.inRange(buff, lower_bleu, upper_bleu)
    
    isBleu = np.sum(bleuMask)
    
    
    buff[np.where(bleuMask==0)] = 0

    bleu_output_img = cv.cvtColor(buff, cv.COLOR_HSV2RGB)
    
    pyplot.figure()
    pyplot.title('Bleu cheese Output image')
    pyplot.imshow(bleu_output_img)
    pyplot.show()
    
    #this code chunk was taken from Derricw on Stackoverflow to create a red mask for hsv
    #hot sauce mask
    #lower mask
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    lower_mask = cv.inRange(buff1, lower_red, upper_red)

    #upper mask
    lower_red = np.array([170, 50, 50])
    upper_red = np.array([180, 255, 255])
    upper_mask = cv.inRange(buff1, lower_red, upper_red)

    mask = lower_mask + upper_mask

    isHot = np.sum(mask)
    
    buff1[np.where(mask==0)] = 0
    
    #create image of bleau cheese detected and hot sauce detected
    buff_output_img = buff + buff1
    
    buff_output_img = cv.cvtColor(buff_output_img, cv.COLOR_HSV2RGB)
    
    # pyplot.figure()
    # pyplot.title('Output image')
    # # pyplot.imshow(output_img2)
    # pyplot.show()
    
    # print("bleu: ")
    # print(isBleu)
    # print("hot sauce: ")
    # print(isHot)
    
    if isHot > 130000000 and isBleu > 780000000:
        print("This is a Buffalo Chicken Pizza")
        
    pyplot.figure()
    pyplot.title('Buffalo Output image')
    pyplot.imshow(buff_output_img)
    pyplot.show()
    
def pomodoro(pizza):
    
    pomo = pizza.copy()
    pomo1 = pizza.copy()
    pomo2 = pizza.copy()
     
    #SPINACH
    lower_spinach = np.array([30, 10, 10])
    upper_spinach = np.array([50, 150, 200])
    spinMask = cv.inRange(pomo, lower_spinach, upper_spinach)
    
    isSpinach = np.sum(spinMask)
    
    pomo[np.where(spinMask==0)] = 0

    spin_output_img = cv.cvtColor(pomo, cv.COLOR_HSV2RGB)
    
    pyplot.figure()
    pyplot.title('Spinach Output image')
    pyplot.imshow(spin_output_img)
    pyplot.show()
    
    #this code chunk was taken from Derricw on Stackoverflow to create a red mask for hsv
    #SAUCE
    #lower mask
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    lower_mask = cv.inRange(pomo1, lower_red, upper_red)

    #upper mask
    lower_red = np.array([170, 50, 50])
    upper_red = np.array([180, 255, 255])
    upper_mask = cv.inRange(pomo1, lower_red, upper_red)

    mask = lower_mask + upper_mask

    isCaccia = np.sum(mask)
    
    pomo1[np.where(mask==0)] = 0
    
    sauce_output_img = cv.cvtColor(pomo1, cv.COLOR_HSV2RGB)
    
    pyplot.figure()
    pyplot.title('Cacciatore Output image')
    pyplot.imshow(sauce_output_img)
    pyplot.show()
    
    caccia_output_img = pomo + pomo1
    
    caccia_output_img = cv.cvtColor(caccia_output_img, cv.COLOR_HSV2RGB)
    
    pyplot.figure()
    pyplot.title('Spin and Caccia Output image')
    pyplot.imshow(caccia_output_img)
    pyplot.show()
    
    #FETA
    lower_feta = np.array([0, 0, 200])
    upper_feta = np.array([60, 60, 255])
    fetaMask = cv.inRange(pomo2, lower_feta, upper_feta)
    
    isFeta = np.sum(fetaMask)
    
    pomo2[np.where(fetaMask==0)] = 0

    feta_output_img = cv.cvtColor(pomo2, cv.COLOR_HSV2RGB)
    
    pyplot.figure()
    pyplot.title('feta Output image')
    pyplot.imshow(feta_output_img)
    pyplot.show()
    
    # print(isSpinach)
    # print(isCaccia)
    # print(isFeta)
    
    if isSpinach > 190000000 and isCaccia > 290000000 and isCaccia < 380000000 and isFeta > 175000000:
        print('This is a Pomodoro pizza!')
        
def cheese(pizza):
    
    cheese = pizza.copy()
     
    #cheese
    lower_cheese = np.array([3, 65, 110])
    upper_cheese = np.array([23, 200, 255])
    cheeseMask = cv.inRange(cheese, lower_cheese, upper_cheese)
    
    isChz = np.sum(cheeseMask)
    
    cheese[np.where(cheeseMask==0)] = 0

    cheese_output_img = cv.cvtColor(cheese, cv.COLOR_HSV2RGB)
    
    # print(isChz)
    
    pyplot.figure()
    pyplot.title('Cheese Output image')
    pyplot.imshow(cheese_output_img)
    pyplot.show()
    
    if isChz > 1700000000:
        print("This is a Cheese Pizza!")
    
if __name__ == '__main__':
    
    print("This is a pizza test... \n\n")
    pepperoni(pizza, grayPizza)
    macAndCheese(pizza)
    buffalo(pizza)
    pomodoro(pizza)
    cheese(pizza)