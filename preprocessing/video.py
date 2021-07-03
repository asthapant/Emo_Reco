eye_cascade = cv2.CascadeClassifier('/content/haarcascade_eye.xml')
face_cascade = cv2.CascadeClassifier('/content/haarcascade_frontalface_default.xml')

def rotate(image):
  l1=[]
  eyes = eye_cascade.detectMultiScale(image)
  for (ex,ey,ew,eh) in eyes:
      l1.append((ex+ew/2,ey+eh/2))

  if(len(l1)==2):
    dist_x = l1[1][0]-l1[0][0]
    dist_y = l1[1][1] - l1[0][1]
    if dist_x<0:
      dist_y = -dist_y
    dist_x = np.abs(dist_x)
    angle = np.arctan(dist_y/(dist_x+1e-8)) * 180/3.14
    M = cv2.getRotationMatrix2D((240, 240), angle, 1.0)
    image = cv2.warpAffine(image, M,(480,480))
  return image

def crop(image,x_factor=2.1,y_factor=3.2):
  l1=[]
  eyes = eye_cascade.detectMultiScale(image)
  for (ex,ey,ew,eh) in eyes:
      l1.append((ex+ew/2,ey+eh/2))
  if(len(l1)==2):
    dist = np.sqrt((l1[0][0]-l1[1][0])**2+(l1[0][1]-l1[1][1])**2)
    center_x = image.shape[1]//2
    center_y = image.shape[0]//2
    shift_x = int(dist*x_factor)//2
    shift_y = int(dist*y_factor)//2
    start_x = center_x - shift_x
    start_x = max(start_x,0)
    end_x = center_x+shift_x
    end_x = min(end_x,image.shape[1])
    start_y = center_y - shift_y
    start_y = max(start_y,0)
    end_y = center_y + shift_y
    end_y = min(end_y,image.shape[0])
    image = image[start_y:end_y,start_x:end_x]
  return image

