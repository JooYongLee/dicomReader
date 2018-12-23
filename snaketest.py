"""
ref : https://www.crisluengo.net/archives/217
authorized lee jooyong
2018.12.23
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2

DEBUG_SHOW = True
alpha = 0.001
beta = 0.4
gamma = 10
iterations = 50


def _get_circle_img():
    W = 200
    H = 200
    img = np.ones([H,W], dtype=np.uint8) * 150
    ctr_x = 140
    ctr_y = 100
    radius = 50

    radius_2 = radius * radius

    mx,my = np.meshgrid(np.arange(W),np.arange(H))
    dist = np.power((my - ctr_y),2) +np.power((mx - ctr_x),2)

    img[dist<=radius_2] = 50
    return img
def _add_noise_to_img(img, m=0,std=10):
    randimg = np.random.normal(m, std, img.shape)
    img = img + randimg.astype(np.uint8)
    return img

def _get_initial_snake(op='line',drawop=False):
    if op == 'circle':
        ctr_x = 250
        ctr_y = 215
        radius = 180
        theta = np.linspace(0,2*np.pi,100)
        x = ctr_x + radius*np.cos(theta)
        y = ctr_y + radius*np.sin(theta)
    elif op == 'arc':
        theta = np.linspace(np.pi/2,np.pi*1.5,10)
        x = 100+60*np.cos(theta)
        y = 100+60*np.sin(theta)
    elif op == 'line':
        x = np.linspace(100,200,100)
        y = 800 - 3*x

    if drawop:
        plt.plot(x,y,'g')
    return x,y
def grad_mag(fimg):
    img = fimg.copy().astype(np.float64)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.abs(sobelx) + np.abs(sobely)
    sobel_mag = sobel_mag / 2
    sobel_inv = sobel_mag
    return sobel_inv, sobelx, sobely

def get_quiver(fx, fy, mu,iter = 50):
    u = fx.copy()
    v = fy.copy()

    fx2 = np.power(fx,2)
    fy2 = np.power(fy,2)
    sqrMag = fx2 + fy2
    dt = 1
    """
    Solve the differential equation iteratively
    Diffusion function being implemented is from the paper Snakes, Shapes and Gradient Vector FLow
    by Chenyang Xu and Jerry L Prince
    u = u + mu * del2(u) -(u - fx) * (fx2 + fy2);
    v = v + mu * del2(v) - (v - fy) * (fx2 + fy2); 
    """
    for cnt in range(iter):
        gradX = cv2.Laplacian(u, cv2.CV_64F, ksize=1, borderType=cv2.BORDER_REFLECT )
        gradY = cv2.Laplacian(v, cv2.CV_64F, ksize=1, borderType=cv2.BORDER_REFLECT)

        u_term2 = (u - fx ) * sqrMag
        v_term2 = (v - fy) * sqrMag

        u = u + dt * (mu * gradX - u_term2)
        v = v + dt * (mu * gradY - v_term2)

    return u,v
def normalize_mag(x,y):
    mag = np.power(x,2) + np.power(y,2)
    mag = np.sqrt(mag)
    mag[mag==0.] = 1.
    x = x / mag
    y = y / mag
    # x[mag==0.] = 0
    # y[mag==0.] = 0
    return x, y



def get_external_force_matrix(N):
    # N = init_x.shape[0]
    a = gamma*(2*alpha+6*beta)+1
    b = gamma*(-alpha-4*beta)
    c = gamma*beta

    P = np.diag(np.ones(N)*a)

    P = P + np.diag(np.ones(N-1)*b, 1) + np.diag(np.ones(1)*b, -N+1)
    P = P + np.diag(np.ones(N-1)*b,-1) + np.diag(np.ones(1)*b,  N-1)
    P = P + np.diag(np.ones(N-2)*c, 2) + np.diag(np.ones(2)*c, -N+2)
    P = P + np.diag(np.ones(N-2)*c,-2) + np.diag(np.ones(2)*c,  N-2)

    # P = P + np.diag(np.ones(N-1)*b, 1)
    # P = P + np.diag(np.ones(N-1)*b,-1)
    # P = P + np.diag(np.ones(N-2)*c, 2)
    # P = P + np.diag(np.ones(N-2)*c,-2)

    mP = np.matrix(P)
    P = np.array(mP.I)
    return P
def get_vector_from_img(imgs,pts):
    """
    :param volumes: 2d image data
    :param pts: extract points... N X 3 [x,y]
    :return: using interpolation, return value from float points(pts)
    """
    from scipy.interpolate import RegularGridInterpolator
    x = np.linspace(0,imgs.shape[1],imgs.shape[1])
    y = np.linspace(0, imgs.shape[0],imgs.shape[0])
    interpolating_function = RegularGridInterpolator((y, x), imgs)
    return interpolating_function(pts)
def _get_test_sample_img():
    testfile = 'dog.jpg'
    import os
    if os.path.isfile(testfile):
        return cv2.imread(testfile, cv2.IMREAD_GRAYSCALE)
    else:
        img = _get_circle_img()
        return _add_noise_to_img(img)

def test_snake_actie_contour():




    img = _get_test_sample_img()

    init_x, init_y = _get_initial_snake()

    P = get_external_force_matrix(init_x.shape[0])

    x = init_x
    y = init_y

    dnoise_img = cv2.GaussianBlur(img,(3,3),0)
    edges = cv2.Canny(dnoise_img, 50,100)

    edges = 255 - edges
    distancemap = cv2.distanceTransform(edges,  cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    distancemap = distancemap / distancemap.max()
    distancemap = 1 - distancemap

    _, gradx, grady = grad_mag(distancemap)

    gradx, grady = normalize_mag(gradx,grady)
    # compute diffusion
    u, v = get_quiver(gradx,grady,0.1)

    gridx,gridy = np.meshgrid(np.arange(u.shape[1]),np.arange(u.shape[0]))

    if DEBUG_SHOW:
        DT = 10
        fig = plt.figure()
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax1.imshow(img,cmap='gray')
        ax1.plot(x, y, 'b')
        ax1.quiver(gridx[::DT,::DT], gridy[::DT,::DT], u[::DT,::DT], v[::DT,::DT],pivot='mid', units='inches')

        ax2.imshow(distancemap)



    for ii in range(iterations):
       coords = np.stack([y,x],axis=1)

       ex_u = get_vector_from_img(u,np.array(coords))
       ex_v = get_vector_from_img(v, np.array(coords))

       x = np.matmul(P, x + gamma * ex_u)
       y = np.matmul(P, y + gamma * ex_v)

       # if DEBUG_SHOW :
       if ii % 5 == 0:
           # ax3.imshow(img,cmap='gray')
           # ax3.plot(x,y,'b')
           plt.imshow(img,cmap='gray')
           plt.plot(x,y,'b')
           plt.show()

    # sortinds =np.argsort(x)
    # x = x[sortinds]
    # y = y[sortinds]
    ax3.plot(x,y,'r')

    ax3.plot(init_x,init_y,'g')
    plt.show()


if __name__=="__main__":
    test_snake_actie_contour()




