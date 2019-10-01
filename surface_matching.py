import numpy as np
import matplotlib.pyplot as plt
import cv2


def get_sample(max_theta, num_saple, noise_sigma):
    theta = np.linspace(0, max_theta, num_saple)
    t = np.linspace(0, 10, num_saple)
    r = np.exp(-((t - 5) ** 2) / (5 ** 2))
    rmean = np.mean(r)
    noise = np.random.normal(0, rmean * noise_sigma , [num_saple, 2])

    x = r * np.cos(theta)
    y = r * np.sin(theta)
    pts = np.stack([x, y], axis=1) + noise

    return pts

def apply_transform(t, pts):
    pts_ex = np.ones([pts.shape[0], 3])
    pts_ex[:, :2] = pts
    return np.dot(t, pts_ex.T).T[:, :2]

def create_transform(deg, origin):
    """
    :param vec: 2x2
    :param origin: 2
    :return:
    """
    rad = np.deg2rad(deg)
    c = np.cos(rad)
    s = np.sin(rad)
    rot = np.array([[c, -s], [s, c]])
    # norm_vec = vec / np.linalg.norm(vec, axis=1).reshape([-1, 1])
    t = np.diag(np.ones([3]))
    t[:2, :2] = rot
    t[:2, 2] = -np.dot(rot.T, origin)
    return t

def sampling(source, transform, model_unity, models=None):
    copy_src = source.copy()
    inv_t = np.linalg.inv(transform)
    transform_source = apply_transform(inv_t, source)
    transform_source_unity = transform_source / np.linalg.norm(transform_source, axis=1).reshape([-1, 1])

    dot = np.dot(transform_source_unity, model_unity.T)
    max_arg = np.argmax(dot, axis=0)
    max_value = np.max(dot, axis=0)
    thresh = 0.85
    constraint = max_value > thresh

    out = source[max_arg[constraint]]
    plt.plot(source.T[0], source.T[1], 'o')
    plt.plot(transform_source.T[0], transform_source.T[1], 'x')
    plt.plot(models.T[0], models.T[1], 'g*')

    for pt1, pt2 in zip(transform_source[max_arg[constraint]], models[constraint]):
        plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]])
    plt.show()

    return out, constraint



def main():
    model = get_sample(np.pi*2, 100, noise_sigma=0)

    eud = np.linalg.norm(model, axis=1)

    noise = np.random.normal(0, eud.mean() * 0.03 , model.shape)

    target_temp = model + noise

    transform = create_transform(np.random.choice(180, 1)[0], np.array([0.4, 0.2]))
    target = apply_transform(transform, target_temp)


    resample_model = np.concatenate([model[-20:], model[:20]], axis=0)
    # target_resample = np.concatenate([-20:])
    model_unity = resample_model / np.linalg.norm(resample_model, axis=1).reshape([-1, 1])


    t2 = create_transform(20, np.array([0.3, 0.1]))
    matching_tar, where = sampling(target, t2, model_unity, resample_model)
    model_pts = apply_transform(t2, resample_model)

    plt.plot(model_pts.T[0], model_pts.T[1], 'ro')

    for pt1, pt2 in zip(matching_tar, model_pts[where]):
        plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]])







    # rot = cv2.getRotationMatrix2D((0, 0), 35, 1)
    # translate = np.array([0.5, 0.2])
    # transfrom_target = apply_transform(rot, target) + translate
    # print(rot)


    # plt.plot(target.T[0], target.T[1], 'x')
    plt.plot(target.T[0], target.T[1], 'x')
    plt.plot(resample_model.T[0], resample_model.T[1], 'o')
    # plt.plot(model.T[0], model.T[1], '*')
    plt.show()

if __name__=="__main__":
    main()
