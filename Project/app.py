from re import X
from numpy import testing
from numpy.core.fromnumeric import sort
from numpy.lib.type_check import imag
import pygame,sys
from pygame import font
from pygame.font import Font
from pygame.locals import *
import numpy as np
from keras.models import Model, load_model
import cv2



boundry =5


sizeX=640
sizeY=480

white = (255,255,255)
black = (0,0,0)
red = (255,0,0)

image_save =False

model=load_model("bestmodel.h5")
PREDICT=True

labels={0:"Zero", 1:"One", 2:"Two", 3:"Three", 4:"Four",
        5:"Five", 6:"Six", 7:"Seven", 8:"Eight", 9:"Nine"}

#Initialize our pygame
pygame.init()
display_surface=pygame.display.set_mode((sizeX,sizeY))
pygame.display.set_caption("Digit Board")

FONT=pygame.font.Font(None,18)


iswriting=False

number_xcord = []
number_ycord = []

img_count=1

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        if event.type == MOUSEMOTION and iswriting:
            xcord,ycord = event.pos
            pygame.draw.circle(display_surface,white,(xcord,ycord),4,0)

            number_xcord.append(xcord)
            number_ycord.append(ycord)

        if event.type ==MOUSEBUTTONDOWN:
            iswriting = True

        if event.type ==MOUSEBUTTONUP:
            iswriting = False
            number_xcord = sorted(number_xcord)
            number_ycord = sorted(number_ycord)

            rect_min_x, rect_max_x = max(number_xcord[0]-boundry,0),min(sizeX, number_xcord[-1]+boundry)
            rect_min_y, rect_max_y = max(number_ycord[0]-boundry,0),min( number_ycord[-1]+boundry, sizeY)

            number_xcord = []
            number_ycord = []

            img_array= np.array(pygame.PixelArray(display_surface))[rect_min_x:rect_max_x,rect_min_y:rect_max_y].T.astype(np.float32)

            if image_save:
                cv2.imwrite("image.png")
                img_count += 1

            if PREDICT:
                image = cv2.resize(img_array,(28,28))
                image = np.pad(image, (10,10), "constant", constant_values = 0)
                image = cv2.resize(image, (28,28))/255

                label = str(labels[np.argmax(model.predict(image.reshape(1,28,28,1)))])

                text_surface = FONT.render(label,True,red,white)
                text_rect_object = text_surface.get_rect()
                text_rect_object.left , text_rect_object.bottom = rect_min_x,rect_max_y

                display_surface.blit(text_surface,text_rect_object)

            if event.type == KEYDOWN:
                if event.unicode == "n":
                    display_surface.fill(black)
        
        pygame.display.update()

