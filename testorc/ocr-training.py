import sys
import numpy as np
import cv2
import os

MIN_CONTOUR_AREA = 100

RESIZED_IMAGE_WIDTH = 30
RESIZED_IMAGE_HEIGHT = 40
def main():
  imgTrainingNumbers = cv2.imread("temple5.png")
  # đọc ảnh vào

  if imgTrainingNumbers is None:  
      print ("error: image not read from file \n\n") 
      os.system("pause") 
      return
  # nếu ko thấy ảnh thì xuất lỗi và dừng chương trình
  imgGray = cv2.cvtColor(imgTrainingNumbers, cv2.COLOR_BGR2GRAY)
  # chuyển màu của ảnh thành COLOR_BGR2GRAY
  imgBlurred = cv2.GaussianBlur(imgGray, (5,5), 0)
  # làm mờ ảnh
  imgThresh = cv2.adaptiveThreshold(imgBlurred,
                                    255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV,
                                    11,
                                    2)
    # đảo màu
    # nếu dùng ảnh imgGray thì imgThresh sẽ ko mịn nên phải blur ảnh trước
  cv2.imshow("imgThresh",imgThresh)

  imgThreshCopy = imgThresh.copy()

  imgContours, npaContours, npaHierarchy = cv2.findContours(imgThreshCopy,
                                              cv2.RETR_EXTERNAL, 
                                              cv2.CHAIN_APPROX_SIMPLE)
  # phân tích hình dạng , phát hiện , nhận dạng đối tượng : npacontour là danh sách các contour có trong ảnh nhị phân , mỗi contour được lưu dưới dạng vector các điểm ,hierarchy : danh sách các vector , chứa mối quan hệ giữa các countour
  npaFlattenedImages =  np.empty((0, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
  intClassifications = []
  # phân loại
  intValidChars = [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9'),
                  ord('A'), ord('B'), ord('C'), ord('D'), ord('E'), ord('F'), ord('G'), ord('H'), ord('I'), ord('J'),
                  ord('K'), ord('L'), ord('M'), ord('N'), ord('O'), ord('P'), ord('Q'), ord('R'), ord('S'), ord('T'),
                  ord('U'), ord('V'), ord('W'), ord('X'), ord('Y'), ord('Z'),ord('a'),ord('b'),ord('c'),ord('d'),
                  ord('e'),ord('f'),ord('g'),ord('h'),ord('i'),ord('j'),ord('k'),ord('l'),ord('m'),ord('n'),ord('o'),
                  ord('p'),ord('q'),ord('r'),ord('s'),ord('t'),ord('u'),ord('v'),ord('w'),ord('x'),ord('y'),ord('z') ]


  for npaContour in npaContours:
      if cv2.contourArea(npaContour) > MIN_CONTOUR_AREA:
          [intX, intY, intW, intH] = cv2.boundingRect(npaContour)
          # Nếu đường bao quanh của ký tự được xác định > MIN_CONTOUR_AREA ta xác định kích thước khung
          print(cv2.boundingRect(npaContour))
          cv2.rectangle(imgTrainingNumbers,
                        (intX, intY), 
                        (intX+intW,intY+intH),
                        (0, 0, 255),
                        2)
          # print(cv2.contourArea(npaContour))
          # print(cv2.boundingRect(npaContour))
          imgROI = imgThresh[intY:intY+intH, intX:intX+intW]
          # cắt ảnh
          imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))
          # resize ảnh theo mẫu
          cv2.imshow("imgROI", imgROI)               
          cv2.imshow("imgROIResized", imgROIResized)
          cv2.imshow("training_numbers.png", imgTrainingNumbers)
          cv2.imwrite("training_numbers.png", imgTrainingNumbers)

          intChar = cv2.waitKey(0) 
          # nhận ký tự từ phím
          if intChar == 27:
              sys.exit()
          elif intChar in intValidChars:
               # nếu inchar có trong intValidChars thì thêm vào mảng intClassifications
              intClassifications.append(intChar)
              print(intChar)

              npaFlattenedImage = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
              npaFlattenedImages = np.append(npaFlattenedImages, npaFlattenedImage, 0) 
              print(npaFlattenedImages)
  fltClassifications = np.array(intClassifications, np.float32)
  npaClassifications = fltClassifications.reshape((fltClassifications.size, 1))
  # đổi thành dạng float32 rồi reshape thành mảng mỗi dòng 1 phần tử
  print ("\n\ntraining complete !!\n")

  np.savetxt("classificationsNEWG.txt", npaClassifications)
  np.savetxt("flattened_imagesNEWG.txt", npaFlattenedImages)
  cv2.destroyAllWindows()

  return

if __name__ == "__main__":
  main()
