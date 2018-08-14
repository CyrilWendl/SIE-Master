"""Functions to generate test data"""
import numpy as np


def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    :param origin: Origin around which to rotate
    :param point: data point
    :param angle: angle, should be given in radians
    :return: rotated x, y point coordinates
    """
    ox, oy = origin
    px, py = point

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def create_data(n_clusters, dimensions, covariance, npoints, minrange=1, maxrange=100, labelled=True,
                random_flip=False, nonlinearities=False):
    """
    Create Gaussian distributed point clusters, in n dimensions
    :param n_clusters: the number of clusters to create
    :param dimensions: the number of dimensions in which to create clusters points
    :param covariance: the covariance of a default cluster
    :param npoints: the number of points per cluster
    :param minrange: the minimum random cluster mean
    :param maxrange: the maximum random cluster mean
    :param labelled: whether to return the cluster labels or not
    :param random_flip: whether to randomly reverse the cluster covariance or not
    :param nonlinearities: whether to randomly transform clusters to nonlinear distributions
    """

    dataset = []
    labels = []
    for idx_c, c in enumerate(range(n_clusters)):
        make_nonlinear = False
        if nonlinearities:
            # set one covariance dimension to 1 to make a circular shape
            make_nonlinear = np.random.randint(0, 2)

        # cluster mean
        mean_c = []
        data_range = maxrange - minrange  # add some margin of data_range / 10
        for d in range(dimensions):
            mean_c.append(np.random.randint(int(minrange + data_range / 10), int(maxrange - data_range / 10)))

        # reshape covariance
        cov_c = np.identity(dimensions) * covariance

        # cluster covariance
        if random_flip:
            # randomly flip covariances (e.g., [10, 20] to [20, 10])
            if np.random.randint(0, 2):
                # get a random dimension, and increase / decrease it by a rand int between +- covariance
                cov_dim_change = np.random.randint(0, dimensions)
                rand_incr = np.random.randint(-covariance, covariance)
                cov_c[cov_dim_change, cov_dim_change] += cov_c[cov_dim_change, cov_dim_change] + rand_incr

        # if nonlinear, elongate covariance
        if make_nonlinear:
            dim = np.random.randint(0, dimensions)
            cov_c[dim, dim] = 1  # set one dimension to 1
            cov_c[dim == 0, dim == 0] = cov_c[dim == 0, dim == 0] * 4  # elongate the other dimensions more

        # generate cluster points
        pts = np.random.multivariate_normal(mean_c, cov_c, npoints).T

        # if nonlinear, curve points
        if make_nonlinear:
            distort_x = np.random.randint(0, 2)
            distort_y = np.random.randint(0, 2)
            if distort_x:  # random if done or not
                y_min = np.min(pts[1], axis=0)
                y_max = np.max(pts[1], axis=0)

                dy = (pts[1] - y_min) / (y_max - y_min)
                if np.random.randint(0, 2):  # random if done or not
                    pts[0] += gaussian(dy, .5, .25) * dy * 50
                else:
                    pts[0] -= gaussian(dy, .5, .25) * dy * 50
            if distort_x == 0 or distort_y:  # random if done or noT
                x_min = np.min(pts[0], axis=0)
                x_max = np.max(pts[0], axis=0)

                dx = (pts[0] - x_min) / (x_max - x_min)
                if np.random.randint(0, 2):  # random if done or not
                    pts[1] += gaussian(dx, .5, .25) * dx * 50
                else:
                    pts[1] -= gaussian(dx, .5, .25) * dx * 50

        # last, check we want to add the labels or not
        labels.append(np.ones(len(pts[0])) * (idx_c + 1))
        dataset.append(pts)

    labels = np.asarray(labels).flatten()

    dataset = np.concatenate(dataset, axis=1)  # merge all clusters to one big matrix [(n_pts*ncluster) * n_dims]

    if labelled:  # add label on top, return clusters
        dataset = np.concatenate((dataset, [labels]))

    return dataset.T


def create_spirals(n_points_arm=1000, radius_min=np.pi / 50, radius_max=7 * np.pi / 16, divergence=0.05, n_arms=4):
    """
    Create data according to a spiral distribution
    :param n_points_arm: number of points per arm
    :param radius_min: minimum radius from to which to create data
    :param radius_max: maximum radius up to which to create data
    :param divergence: scattering factor of points along the sprial arms
    :param n_arms: number of arms to create
    :return:
    """
    theta = np.linspace(radius_min, radius_max, n_points_arm)
    a = np.random.uniform(1 - divergence, 1 + divergence, n_points_arm)

    x = a * np.sqrt(theta) * np.cos(theta)
    y = a * np.sqrt(theta) * np.sin(theta)

    for i in range(1, 5):
        x_, y_ = rotate([0, 0], [x, y], i * 2 * np.pi / n_arms)
        x = np.concatenate((x, x_))
        y = np.concatenate((y, y_))

    # save as new dataset
    dataset = np.asarray([x, y]).T
    min_range = np.min(x)
    max_range = np.max(x)
    return dataset, min_range, max_range


def create_s_shape(n_points_arm=300, radius_min=np.pi / 4, radius_max=6 * np.pi / 4, divergence=0.05):
    """
    Create data according to a S shape
    :param n_points_arm: number of points per arm
    :param radius_min: minimum radius from to which to create data
    :param radius_max: maximum radius up to which to create data
    :param divergence: scattering factor of points along the spiral arms
    :return:
    """
    theta = np.linspace(radius_min, radius_max, n_points_arm)
    a = np.random.uniform(1 - divergence, 1 + divergence, n_points_arm)

    x = a * np.sqrt(theta) * np.sin(theta)
    y = a * np.sqrt(theta) * np.cos(theta) * x

    # save as new dataset
    dataset = np.asarray([x, y]).T
    min_range = np.min([np.min(x), np.min(y)])
    max_range = np.max([np.max(x), np.max(y)])
    return dataset, min_range, max_range


def data_to_clusters(dataset):
    """Helper function to get clusters from estimated labels"""
    clusters = []
    for val in np.unique(dataset[:, 2]):
        clusters.append(dataset[dataset[:, 2] == val])
    return clusters
