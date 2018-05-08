import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from functools import partial

def sum_of_squares(v):
    return np.sum(v**2)


def derivative(f, x, h):
    return (f(x + h) - f(x)) / h


def partial_derivative(f, x, i=0, h=0.001):
    xc = np.array(x)
    xc[i] += h
    return (f(xc) - f(x)) / h


def f(x):
    return x * x


def dfdx(x):
    return 2 * x


def fxn(x):
    print(x)
    return x[0]**2 + x[1]**2


def vFxn(x, y):
    return x**2 + y**2


def plot():
    x = np.arange(-10, 11)
    y = f(x)
    dydx = dfdx(x)
    dydx_e = derivative(f, x, 0.5)

    # Create a plot of the f(x)
    data_set = pd.DataFrame(data={'x': x, 'y': y, 'dydx': dydx, 'dydx_e': dydx_e})
    melted_data_set = pd.melt(data_set, id_vars=['x', 'y'], var_name="dy")
    print(data_set)
    print(melted_data_set)

    # Graph one - of x vs y
    sns.lmplot(x='x', y='y', data=data_set, fit_reg=False)
    plt.title('x against y')

    # Graph two of x vs dydx
    #sns.lmplot(x='x', y='value', data=melted_data_set, fit_reg=False, hue='dy')
    #plt.title('x against dydx')

    #plt.show()


def gradient_descent():
    # Parameters
    h = 0.001  # step size
    t = 0.001  # tolerance

    # Pick a random starting point
    xi = np.random.randint(-10, 10)
    # Calculate the gradient at this point
    dfdxi = derivative(f, xi, h)

    while dfdxi > t:
        # Move down the equation by a tiny step
        xn = xi - h*dfdxi

        # Calculate the new gradient
        dfdxn = derivative(f, xn, h)

        print("New x: {0}, New gradient: {1}".format(str(xn), str(dfdxn)))

        # Repeat until dfdxn is very close to 0 i.e. the minimum
        xi = xn
        dfdxi = dfdxn


def gradient_descent_md():  # Multi dimensional
    h = 0.001
    t = 0.001

    dx = partial(partial_derivative, fxn, i=0)
    dy = partial(partial_derivative, fxn, i=1)

    x = np.round(np.random.random(2)*5)
    dx0 = dx(x)
    dx1 = dy(x)

    while dx0 > t and dx1 > 0:
        if dx0 > t:
            x[0] -= h*dx0
            dx0 = dx(x)

        if dx1 > t:
            x[1] -= h*dx1
            dx1 = dy(x)

        print('New x,y: {0}, new gradients: {1} {2}'.format(str(np.round(x, 4)), np.round(dx0, 4), np.round(dx1, 4)))


def get_plot_data():
    coords = np.dstack(np.meshgrid(np.arange(-10, 11), np.arange(-10, 11))).reshape(-1, 2)
    print(coords)
    x, y = np.split(coords, 2, axis=1)
    x = np.reshape(x, -1)
    y = np.reshape(y, -1)
    z = vFxn(x, y)
    i = np.arange(x.size)
    print(x.shape)
    print(y.shape)
    print(z.shape)

    data = pd.DataFrame({'x': x, 'y': y, 'z': z}, index=i)
    print(data)
    sns.set()
    data = data.pivot(index="x", columns="y", values="z")
    print(data)
    sns.heatmap(data, annot=False)
    plt.show()


def test_things():
    x = [0, 1, 2]
    y = [0, 1, 2]
    z = np.dstack(np.meshgrid(x, y)).reshape(-1, 2)
    print(z)


#gradient_descent()
#gradient_descent_md()
get_plot_data()
#test_things()
