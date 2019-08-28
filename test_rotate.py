import numpy as np
import vtk
class myTransform(vtk.vtkTransform):
    def __init__(self):
        super(myTransform, self).__init__()

    def convert_np_mat(self):
        mat = self.GetMatrix()
        np_mat = np.zeros([4, 4], dtype=np.float64)
        for i in range(4):
            for j in range(4):
                np_mat[i, j] = mat.GetElement(i, j)
        return np_mat

    def transfrom_numpy(self, np_pts):
        np_mat = self.convert_np_mat()
        ex_pts = np.ones([np_pts.shape[0], 4])
        ex_pts[:, :3] = np_pts
        out = np.dot(np_mat, ex_pts.T).T[:, :3]
        return out

    def transform_only_rotate(self, np_pts):
        np_mat = self.convert_np_mat()
        np_mat[:, 3] = np.array([0, 0, 0, 1])
        ex_pts = np.ones([np_pts.shape[0], 4])
        ex_pts[:, :3] = np_pts
        out = np.dot(np_mat, ex_pts.T).T[:, :3]
        return out

def create_rot_mat(theta, u):
    eye = np.diag(np.ones(3))
    ux = np.cross(eye, u)
    uxu = np.dot(u.reshape([3, 1]), u.reshape([1, 3]))
    rad = np.deg2rad(theta)
    R = np.cos(rad)*eye + np.sin(rad) * ux + (1-np.cos(rad)) * uxu
    return R


def rotate_x(theta, u, x):
    rad = np.deg2rad(theta)
    ux = np.cross(u, x)
    out = u * (np.dot(u, x)) + np.cos(rad) * ( np.cross(ux, u)) +\
        np.sin(rad) * ux
    return out

x = np.array([1, 2, 3])
nn = np.array([3, 5,-1])
n = nn / np.linalg.norm(nn)
t = myTransform()
t.RotateWXYZ(30, *tuple(n))
tmat = t.convert_np_mat()
R = create_rot_mat(30, n)

y = rotate_x(30, n, x)
y1 = np.dot(R, x)
print(y)
print(y1, y-y1)
