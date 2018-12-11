# import matplotlib.pyplot as plt
import numpy as np
import  matplotlib.pyplot as plt

def get_random_set(N,noise=True):
    x = np.linspace(0,100,N)
    y = np.square((x-50))*0.01 + 10
    if noise:
        y = y + np.random.normal(5,5,[y.shape[0]])
    return x,y
def plot_test():
    x,y =  get_random_set(100)
    figs = plt.figure()
    axes = figs.add_subplot(111)
    print(x)


    axes.plot(x,y)
    figs.show()
    plt.show()
def get_random_point(N):
    x = np.random.uniform(0,100)
    y = np.random.uniform(0,100)
    return x,y
def random_distance():
    x, y = get_random_set(100,noise=False)
    px, py = get_random_point(1)

    figs = plt.figure()
    axes1 = figs.add_subplot(121)
    axes2 = figs.add_subplot(122)

    ind  = np.random.choice(x.shape[0])
    random_x = x[ind]
    random_y = y[ind]
    random_point = np.array([px,py]).reshape([-1,2])
    # axes.plot(x,y)
    axes1.plot(x, y,'r')
    axes1.plot(px,py,'go')
    axes1.plot(random_x, random_y, 'go')

    du = np.diff(x)
    dv = np.diff(y)

    dx = (x[:-1] + x[1:])/2
    dy = (y[:-1] + y[1:])/2


    direct_vector = np.stack([dx,dy],axis=1) - random_point
    normal_vector = np.stack([du,dv],axis=1)


    norms_direct = np.expand_dims(np.sum(np.square(direct_vector), axis=1), axis=1)
    norms_normal = np.expand_dims(np.sum(np.square(normal_vector), axis=1), axis=1)
    direct_vector = direct_vector / np.sqrt(norms_direct)
    normal_vector = normal_vector / np.sqrt(norms_normal)
    print(np.sum(np.square(direct_vector),axis=1))


    print(direct_vector.shape)
    print(normal_vector.shape)
    # np.matmul(direct_vector)
    vector_sum = np.abs(np.sum(direct_vector * normal_vector,axis=1))
    # vector_sum = np.sum(direct_vector * normal_vector, axis=1)



    minimum_vector_ind = np.argmin(vector_sum)
    # minimum_vector_ind =min_dist_ind
    print(minimum_vector_ind)

    on_x = dx[minimum_vector_ind]
    on_y = dy[minimum_vector_ind]
    axes2.plot(vector_sum)

    axes1.plot(on_x,on_y,'bx')
    line_x = np.array([on_x,px])
    line_y = np.array([on_y,py])

    axes1.plot(line_x, line_y, 'b-.')
    axes1.axis('equal')



    #
    # plt.axis('equal')
    figs.show()
    plt.show()


    print(10)

    # import matplotlib.pyplot as plt
def vector_visualize():
    n = 8
    X, Y = np.mgrid[0:n, 0:n]
    T = np.arctan2(Y - n / 2., X - n / 2.)
    R = 10 + np.sqrt((Y - n / 2.0) ** 2 + (X - n / 2.0) ** 2)
    U, V = R * np.cos(T), R * np.sin(T)

    plt.axes([0.025, 0.025, 0.95, 0.95])
    plt.quiver(X, Y, U, V, R, alpha=.5)
    plt.quiver(X, Y, U, V, edgecolor='k', facecolor='None', linewidth=.5)

    plt.xlim(-1, n)
    plt.xticks(())
    plt.ylim(-1, n)
    plt.yticks(())

    plt.show()

    # shortest_dist =

# vector_visualize()
iteration = 0
while(iteration<10):
    random_distance()
    iteration +=1
# plot_test()
