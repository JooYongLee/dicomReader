import numpy as np
import matplotlib.pyplot as plt
import cv2

def get_points():
    img = cv2.imread("Untitled.png", cv2.IMREAD_GRAYSCALE)
    # cv2.imshow("", img)
    # cv2.waitKey()
    print(img.shape, img.dtype)

    _, contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    drawing = np.zeros_like(img, dtype=np.uint8)

    con_contours = np.squeeze(np.concatenate(contours, axis=0))
    print(con_contours.shape)
    return con_contours


def get_laplacian(N):
    eye = np.diag(np.ones(N))
    L = np.diag(np.ones(N))
    # L[1:, :-1]= -1/2
    L[np.arange(1, N), np.arange(0, N-1)] = -1/2
    L[np.arange(0, N-1), np.arange(1, N)] = -1/2

    return L

def laplacian_shape_reconstruction():
    X = get_points()
    N = X.shape[0]
    L = get_laplacian(N)

    w, v = np.linalg.eig(L)

    ind = np.argsort(w)
    sort_v = v[:, ind]
    sort_w = w[ind]
    X_ = np.dot(sort_v.T, X)

    k = [3, 5, 10, 20, 30, N//2]
    fig = plt.figure()
    for i, ks in enumerate(k):
        coeff = X_.copy()
        coeff[ks:] = 0
        y = np.dot(sort_v, coeff)

        ax = fig.add_subplot("{}{}{}".format(2, 3, i + 1))
        ax.plot(y[:, 0], y[:, 1])
        ax.set_title("k={}".format(ks))
    plt.show()

def laplacian_filtering_smoothing():
    X = get_points()
    N = X.shape[0]
    L = get_laplacian(N)

    w, v = np.linalg.eig(L)

    ind = np.argsort(w)
    sort_v = v[:, ind]
    sort_w = w[ind]
    X_ = np.dot(sort_v.T, X)

    m = [1, 10, 50, 100]

    fig = plt.figure()
    for i in range(4):
        ax = fig.add_subplot("{}{}{}".format(2, 3, i+1))

        mS = sort_v * np.expand_dims((1-sort_w/2)**m[i], axis=0)
        xi = np.dot(mS, X_)
        ax.plot(xi[:, 0], xi[:, 1])
        ax.set_title("power : {}".format(m[i]))

    plt.show()


def show_eigenvalue_spectral_coefficents():
    X = get_points()
    N = X.shape[0]
    L = get_laplacian(N)

    w, v = np.linalg.eig(L)

    ind = np.argsort(w)
    sort_v = v[:, ind]
    sort_w = w[ind]
    X_ = np.dot(sort_v.T, X)
    plt.plot(sort_w, X_)
    plt.show()



if __name__=="__main__":
    show_eigenvalue_spectral_coefficents()
    laplacian_filtering_smoothing()
    laplacian_shape_reconstruction()
