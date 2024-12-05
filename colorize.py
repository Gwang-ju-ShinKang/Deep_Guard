import cv2
import numpy as np

###
proto = 'data/colorization_deploy_v2.prototxt'
weights = 'data/colorization_release_v2.caffemodel'

net = cv2.dnn.readNetFromCaffe(proto, weights)

pts_in_hull = np.load('data/pts_in_hull.npy')
pts_in_hull = pts_in_hull.transpose().reshape(2, 313, 1, 1).astype(np.float32)
net.getLayer(net.getLayerId('class8_ab')).blobs = [pts_in_hull]

net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full((1, 313), 2.606, np.float32)]
###
img = cv2.imread('data/grayscale_image.jpg')

h, w, c = img.shape

img_input = img.copy() # 복사

# 전처리 과정
# 기존 정수형(uint8)을 float32(32비트 소수점) 형태로 바꿔라
img_input = img_input.astype('float32') / 255. 
img_lab = cv2.cvtColor(img_input, cv2.COLOR_BGR2Lab) # Lab 컬러시스템으로 변경
img_l = img_lab[:, :, 0:1] # L, a, b 채널 중 L 채널만 추출

# resize, mean, 차원변형
blob = cv2.dnn.blobFromImage(img_l, size=(224, 224), mean=[50, 50, 50])
###
net.setInput(blob)
output = net.forward()
###
# 인간이 이해할 수 있는 형태로 다시 차원 변형
output = output.squeeze().transpose((1, 2, 0)) 

# 원본 사이즈로 다시 바꿔줌
output_resized = cv2.resize(output, (w, h)) 

# L 채널 합치기 # 채널 방향으로 합치기 (axis=0세로1가로2채널)
output_lab = np.concatenate([img_l, output_resized], axis=2)

output_bgr = cv2.cvtColor(output_lab, cv2.COLOR_Lab2BGR)
output_bgr = output_bgr * 255
output_bgr = np.clip(output_bgr, 0, 255) # 0-255 잘라내기
output_bgr = output_bgr.astype('uint8') # 정수형으로 변경
###
cv2.imshow('img', img_input)
cv2.imshow('result', output_bgr)
cv2.waitKey(0)