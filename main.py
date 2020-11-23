
#import matplotlib.pyplot as plt
import cv2
import numpy as np
import glob

#known
f_x = 1132.804#745.159
f_y = 1132.785#745.743
c_x = 710.39#485.509
c_y = 510.06#377.439
r = 0.95
distance = 5
flash_img = "10.jpg"
light_img = "11.jpg"
automatic = True

# show image..
img = cv2.imread(flash_img, cv2.IMREAD_COLOR)
height, width, channel = img.shape
print(height, width , channel)

flash_point = []

def point_highlight(image):
    #  constants
    BINARY_THRESHOLD = 50
    CONNECTIVITY = 2
    DRAW_CIRCLE_RADIUS = 2

    #  convert to gray
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #  extract edges
    binary_image = cv2.Laplacian(gray_image, cv2.CV_8UC1)

    #  fill in the holes between edges with dilation
    dilated_image = cv2.dilate(binary_image, np.ones((4, 4)))

    #  threshold the black/ non-black areas
    _, thresh = cv2.threshold(dilated_image, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)

    #  find connected components
    components = cv2.connectedComponentsWithStats(thresh, CONNECTIVITY, cv2.CV_32S)

    #  draw circles around center of components
    #see connectedComponentsWithStats function for attributes of components variable
    centers = components[3]
    centers = centers[centers[:, 1] > 600]
    print(centers)
    cv2.imshow("result", thresh)
    cv2.waitKey(0)
    #print(centers[centers[:,1]>600])
    cv2.imwrite(light_img+"new.jpg", thresh)
    return sorted(centers,key=lambda x: x[0]+x[1])

# return pixel if Lmouseclick
def findpixel(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        print(x, y)
        flash_point.append((x, y))
if automatic:
    flash_point = point_highlight(img)
else:
    cv2.namedWindow('flash')
    cv2.setMouseCallback("flash", findpixel)

    cv2.imshow("flash", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#
print(flash_point)



flash_point = np.array(flash_point)#[(492, 531), (510, 527), (526, 523), (543, 521), (558, 518)])
camera_point = np.ones((5,3))

camera_point[:, 0] = (flash_point[:, 0] - c_x) / f_x
camera_point[:, 1] = (flash_point[:, 1] - c_y) / f_y
for i in range(len(camera_point)):
    camera_point[i,:] = camera_point[i,:] / np.linalg.norm(camera_point[i,:])
print('camera_point:')
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

k = np.sqrt(distance**2 / (1 + ans[3, 0]** 2 + ans[4, 0]** 2))
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
img = cv2.imread(light_img, cv2.IMREAD_COLOR)
height, width, channel = img.shape
print(height, width , channel)

light_point = []

# return pixel if Lmouseclick
def findlightpixel(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        print(x, y)
        light_point.append((x,y))


if automatic:
    light_point = point_highlight(img)
else:
    cv2.namedWindow('light')
    cv2.setMouseCallback("light", findlightpixel)

    cv2.imshow("light", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
print(light_point)

light_point = np.array(light_point)#[(492, 531), (510, 527), (526, 523), (543, 521), (558, 518)])
camera_lpoint = np.ones((5,3))

camera_lpoint[:, 0] = (light_point[:, 0] - c_x) / f_x
camera_lpoint[:, 1] = (light_point[:, 1] - c_y) / f_y

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
    cons = [np.sum(f_i ** 2), -2 * (np.dot(f_i, ball_i)), np.sum(ball_i ** 2) -(r**2)]
    print(cons)
    t = np.roots(cons)
    print(t)
    #inter_point[i,:] = t[0] * camera_lpoint[i,:]

    inter_point[i,:] = np.min(t) * camera_lpoint[i,:]
print('inter_point:')
print(inter_point)

#find normal vector of intersection
normal = inter_point - ball_pos
print('normal:')
print(normal)
for i in range(len(normal)):
    normal[i,:] = normal[i,:] / np.linalg.norm(normal[i,:])
print('normal:')
print(normal)

#find reflection vector
reflect = np.zeros((5, 3))
for i in range(len(reflect)):
    reflect[i,:] = inter_point[i,:] - 2 * normal[i,:] * (np.dot(inter_point[i,:],normal[i,:]))
    print((inter_point[i,:] * normal[i,:]))

print('reflect:')
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
    