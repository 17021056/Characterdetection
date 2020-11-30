import sys
import numpy as np
import cv2
import os

MIN_CONTOUR_AREA = 100

RESIZED_IMAGE_WIDTH = 30
RESIZED_IMAGE_HEIGHT = 40
def main():
  imgTrainingNumbers = cv2.imread("temple4.png")

  if imgTrainingNumbers is None:  
      print ("error: image not read from file \n\n") 
      os.system("pause") 
      return

  imgGray = cv2.cvtColor(imgTrainingNumbers, cv2.COLOR_BGR2GRAY)
  imgBlurred = cv2.GaussianBlur(imgGray, (5,5), 0)

  imgThresh = cv2.adaptiveThreshold(imgBlurred,
                                    255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV,
                                    11,
                                    2)

  cv2.imshow("imgThresh", imgThresh)

  imgThreshCopy = imgThresh.copy()

  imgContours, npaContours, npaHierarchy = cv2.findContours(imgThreshCopy,
                                              cv2.RETR_EXTERNAL, 
                                              cv2.CHAIN_APPROX_SIMPLE)


  npaFlattenedImages = np.loadtxt("flattened_imagesNEWG.txt")

  npaClassifications = np.loadtxt("classificationsNEWG.txt")

  intValidChars = [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9'),
                  ord('A'), ord('B'), ord('C'), ord('D'), ord('E'), ord('F'), ord('G'), ord('H'), ord('I'), ord('J'),
                  ord('K'), ord('L'), ord('M'), ord('N'), ord('O'), ord('P'), ord('Q'), ord('R'), ord('S'), ord('T'),
                  ord('U'), ord('V'), ord('W'), ord('X'), ord('Y'), ord('Z'),ord('a'),ord('b'),ord('c'),ord('d'),
                  ord('e'),ord('f'),ord('g'),ord('h'),ord('i'),ord('j'),ord('k'),ord('l'),ord('m'),ord('n'),ord('o'),
                  ord('p'),ord('q'),ord('r'),ord('s'),ord('t'),ord('u'),ord('v'),ord('w'),ord('x'),ord('y'),ord('z') ]


  for npaContour in npaContours:
      if cv2.contourArea(npaContour) > MIN_CONTOUR_AREA:
          [intX, intY, intW, intH] = cv2.boundingRect(npaContour)


          cv2.rectangle(imgTrainingNumbers,
                        (intX, intY), 
                        (intX+intW,intY+intH),
                        (178,103,63),
                        2)

          imgROI = imgThresh[intY:intY+intH, intX:intX+intW] 
          imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))

          cv2.imshow("imgROI", imgROI)               
          cv2.imshow("imgROIResized", imgROIResized)
          cv2.imshow("training_numbers.png", imgTrainingNumbers)
          
          cv2.waitKey(0) 

          npaFlattenedImage = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))

          for i in range(len(npaClassifications)):
            if((npaFlattenedImages[i] != np.array(npaFlattenedImage)).all()):
                print("I dont know your text")
            elif((npaFlattenedImages[i] == np.array(npaFlattenedImage)).all()):
                print(chr(int(npaClassifications[i])))

  print ("\n\ndetect complete !!\n")

  cv2.destroyAllWindows()
  return

if __name__ == "__main__":
  main()
