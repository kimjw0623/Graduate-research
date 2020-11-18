
#import matplotlib.pyplot as plt
import cv2
import numpy as np
import glob
"""
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((5*8,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:5].T.reshape(-1,2)*30

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('*.jpg')


for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (8,5),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (8,5), corners2,ret)
        cv2.imshow('img',img)
        cv2.waitKey(500)
    else:
        print('ff')

cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

img = cv2.imread('1.jpg')
h,  w = img.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
#undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
#crop the image
#x,y,w,h = roi
#dst = dst[y:y+h, x:x+w]
cv2.imwrite('new1.jpg', dst)

print(mtx)
"""
# show image..
img = cv2.imread("1.jpg", cv2.IMREAD_COLOR)
height, width, channel = img.shape
print(height, width , channel)

flash_point = []

# return pixel if Lmouseclick
def findpixel(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        print(x, y)
        flash_point.append((x,y))
cv2.namedWindow('flash')
cv2.setMouseCallback("flash", findpixel)

cv2.imshow("flash", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(flash_point)

f_x = 1113.04#745.159
f_y = 1114.05#745.743
c_x = 727.69#485.509
c_y = 566.13#377.439


flash_point = np.array(flash_point)#[(492, 531), (510, 527), (526, 523), (543, 521), (558, 518)])
camera_point = np.ones((5,3))

camera_point[:, 0] = (flash_point[:, 0] - c_x) / f_x
camera_point[:, 1] = (flash_point[:, 0] - c_y) / f_y
for i in range(len(camera_point)):
    camera_point[i,:] = camera_point[i,:] / np.linalg.norm(camera_point[i,:])

print(camera_point)

#least square method
#coeff = coeff of [p_x,p_y,p_z,d_y,d_z]
coeff = np.zeros((15, 5))
const = np.zeros((15, 1))

for i in range(0, len(camera_point)):
    f_ix = camera_point[i, 0]
    f_iy = camera_point[i, 1]
    f_iz = camera_point[i, 2]
    coeff[i*3,:] = np.array([0, -f_iz, f_iy, -i * f_iz, i * f_iy])
    const[i*3,:] = np.array([0])
    coeff[i*3 + 1,:] = np.array([f_iz, 0, - f_ix, 0, -i*f_ix])
    const[i*3 + 1,:] = np.array([-i * f_iz])
    coeff[i*3 + 2,:] = np.array([-f_iy, f_ix, 0, i*f_ix, 0])
    const[i*3 + 2,:] = np.array([i * f_iy])
    

#print(coeff)
#print(const)

ans = np.dot(np.linalg.pinv(coeff),const)
#print(ans)

k = np.sqrt(7.5**2 / (1 + ans[3, 0]** 2 + ans[4, 0]** 2))
print(k)
print(ans * k)
ans = ans*k

ball_pos = np.zeros((5,3))
d = np.array([k, ans[3, 0], ans[4, 0]])
p0 = np.array([ans[0,0],ans[1,0],ans[2,0]])

for i in range(5):
    ball_pos[i,:] = p0 + i * d
    
print('ball_pos :')
print(ball_pos)
print('d :')
print(d)

p0_y = ans[0, 0]
p0_z = ans[0, 0]


##########finding intersection with sphere#########

#choose highlight point 
img = cv2.imread("1.jpg", cv2.IMREAD_COLOR)
height, width, channel = img.shape
print(height, width , channel)

light_point = []

# return pixel if Lmouseclick
def findlightpixel(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        print(x, y)
        light_point.append((x,y))
cv2.namedWindow('light')
cv2.setMouseCallback("light", findlightpixel)

cv2.imshow("light", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(light_point)

light_point = np.array(light_point)#[(492, 531), (510, 527), (526, 523), (543, 521), (558, 518)])
camera_lpoint = np.ones((5,3))

camera_lpoint[:, 0] = (light_point[:, 0] - c_x) / f_x
camera_lpoint[:, 1] = (light_point[:, 0] - c_y) / f_y

for i in range(len(camera_lpoint)):
    camera_lpoint[i,:] = camera_lpoint[i,:] / np.linalg.norm(camera_lpoint[i,:])

print('light pos :')
print(camera_lpoint)
#find intersection with sphere (use np roots)

inter_point = np.zeros((5,3))

for i in range(len(camera_lpoint)):
    f_ix = camera_lpoint[i,0]
    f_iy = camera_lpoint[i,1]
    f_iz = camera_lpoint[i, 2]
    f_i = np.array([f_ix,f_iy,f_iz])#camera_lpoint[i,:]
    ball_ix = ball_pos[i, 0]
    ball_iy = ball_pos[i, 1]
    ball_iz = ball_pos[i, 2]
    ball_i = np.array([ball_ix,ball_iy,ball_iz])
    cons = [np.sum(f_i ** 2), -2 * (np.dot(f_i, ball_i)), np.sum(ball_i ** 2) - 1]
    print(cons)
    t = np.roots(cons)
    print(t)
    #inter_point[i,:] = t[0] * camera_lpoint[i,:]

    inter_point[i,:] = t[1] * camera_lpoint[i,:]

print(inter_point)

#find normal vector of intersection
normal = inter_point - ball_pos
for i in range(len(normal)):
    normal[i,:] = normal[i,:] / np.linalg.norm(normal[i,:])

print(normal)

#find reflection vector
reflect = np.zeros((5, 3))
for i in range(len(reflect)):
    reflect[i,:] = inter_point[i,:] - 2 * normal[i,:] * (np.sum(inter_point[i,:] * normal[i,:]))
    print((inter_point[i,:] * normal[i,:]))
print(reflect)

#least square method with intersection and reflection vector -> find light source position
#light coeff = (l_x, l_y, l_z, t_1, t_2, ... t_5) light: (l_x,l_y,l_z) param: t_i
light_coeff = np.zeros((15, 8))
light_const = inter_point.flatten()
for i in range(5):
    #light_coeff[i*3] = np.array([1,0,0,reflect[i,0],0,0,0,0])
    light_coeff[i * 3, 0] = 1
    light_coeff[i * 3, i + 3] = - reflect[i, 0]
    light_coeff[i * 3 + 1, 1] = 1
    light_coeff[i * 3 + 1, i + 3] = - reflect[i, 1]
    light_coeff[i * 3 + 2, 2] = 1
    light_coeff[i * 3 + 2, i + 3] = -reflect[i, 2]

print('light_coeff :')
print(light_coeff)
print('light_const :')
print(light_const)

light_pos = np.dot(np.linalg.pinv(light_coeff), light_const)
print('light_pos :')
print(light_pos)
    