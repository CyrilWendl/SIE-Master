from density_forest.random_forest import get_grid_labels
from density_forest.create_data import data_to_clusters
from density_forest.helpers import get_values_preorder, draw_subsamples
from helpers.helpers import get_padded_im, get_n_patches
from matplotlib.pyplot import cm
from matplotlib.patches import Ellipse, Rectangle
import matplotlib.pylab as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from helpers.helpers import convert_patches_to_image, gt_label_to_color
from matplotlib import patches


def plot_data(data, ax, title=None, n_clusters=None, save=False, lines_x=None, lines_y=None,
              labels=True, minrange=1, maxrange=100, margin=2, grid_eval=None, show_data=True, means=None,
              covs=None):
    """
    Generic function to plot randomly generated labelled or unlabelled data.
    :param data: the data to plot
    :param ax: axis where to draw the plot
    :param title: the title of the plot
    :param n_clusters: number of clusters in data
    :param lines_x: x splitting lines to plot
    :param lines_y: y splitting lines to plot
    :param save: [True | False]save plot to a pdf file
    :param labels: [True | False] indicator whether data contains labels
    :param minrange: data parameters for setting axis limits
    :param maxrange: data parameters for setting axis limits
    :param margin: data parameters for setting axis limits
    :param grid_eval: whether to show evaluation on regular grid
    :param show_data: whether to show evaluation on regular grid
    :param means: parameters to show covariance ellipse with unlabelled data
    :param covs: parameters to show covariance ellipse with unlabelled data
    """
    if show_data:
        if labels:
            color = iter(cm.rainbow(np.linspace(0, 1, n_clusters)))
            for i, c in enumerate(data):
                color_cluster = next(color)
                ax.scatter(c[:, 0], c[:, 1], s=40, color=color_cluster)

                x = c[:, 0]
                y = c[:, 1]
                n = [int(c) for c in c[:, 2]]

                for j, txt in enumerate(n):
                    ax.annotate(txt, (x[j], y[j]))
        else:
            ax.plot(data[:, 0], data[:, 1], '.')
    if title is not None:
        ax.set_title(title)

    # draw split lines after partitioning
    ax.grid()
    if lines_x is not None and lines_y is not None:
        for y_line in range(len(lines_y)):
            ax.axhline(y=lines_y[y_line], c="red")
        for x_line in range(len(lines_x)):
            ax.axvline(x=lines_x[x_line], c="red")

    # draw colored meshgrid
    if grid_eval is not None:
        x_min, x_max = [minrange, maxrange]
        y_min, y_max = [minrange, maxrange]
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))

        grid = grid_eval

        # Put the result into a color plot
        grid = grid.reshape(xx.shape)
        ax.pcolormesh(xx, yy, grid, alpha=0.2, cmap='rainbow')
        # ax.set_clim(y.min(), y.max())

    ax.set_xlim([minrange - margin, maxrange + margin])
    ax.set_ylim([minrange - margin, maxrange + margin])

    #  covariance ellipses
    def eigsorted(cov):
        _vals, _vecs = np.linalg.eigh(cov)
        order = _vals.argsort()[::-1]
        return _vals[order], _vecs[:, order]

    nstd = 2
    if covs is not None:
        for i in range(len(covs)):
            vals, vecs = eigsorted(covs[i])
            theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
            w, h = 2 * nstd * np.sqrt(vals)
            ell = Ellipse(xy=means[i],
                          width=w, height=h,
                          angle=theta, color='red')
            ell.set_facecolor('none')
            ax.add_artist(ell)

    if save:
        plt.savefig('/Users/cyrilwendl/Documents/EPFL/Projet SIE/SIE-Project/random_data.pdf', bbox_inches='tight')


def visualize_decision_boundaries(dataset, rootnode, minrange, maxrange, rf=False, save=False, savename=None):
    """visualize decision boundaries for a given decision tree"""
    # plot data

    clusters = data_to_clusters(dataset)
    dataset_grid_eval = get_grid_labels(rootnode, minrange, maxrange, rf=rf)

    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches((12, 8))

    fig.set_size_inches((15, 6))
    plot_data(clusters, axes[0], "Training Data and Splits", n_clusters=len(clusters), minrange=minrange,
              maxrange=maxrange, margin=0, grid_eval=dataset_grid_eval, show_data=True)

    plot_data(clusters, axes[1], "Splits", n_clusters=len(clusters), minrange=minrange,
              maxrange=maxrange, margin=0, grid_eval=dataset_grid_eval, show_data=False)

    if save:
        plt.savefig(savename, bbox_inches='tight', pad_inches=0)

    plt.show()


def plot_pts_3d(x_pts, y_labels, classes_to_keep, colors,
                names=None, class_to_remove=None, subsample_pct=1, s_name=None):
    """
    Plot 3D data with class label and with op
    :param x_pts: 3D data to plot
    :param y_labels: the corresponding y labels to the PCA data, needs to be in same classes as classes_to_keep
    :param class_to_remove: class number removed in classification
    :param classes_to_keep: array of classes to keep
    :param names: class names
    :param colors: colors corresponding to classes
    :param subsample_pct: percentage of data to show on plot (optional)
    :param s_name: name where to save file (optional)
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection=Axes3D.name)

    # points corresponding to seen classes
    for i, class_keep in enumerate(classes_to_keep):
        data_plt = draw_subsamples(x_pts[y_labels == class_keep], subsample_pct=subsample_pct, replace=False)
        ax.scatter(data_plt[:, 0], data_plt[:, 1], zs=data_plt[:, 2],
                   c=np.asarray(colors)[class_keep], s=15, depthshade=True, marker='o', alpha=.3)

    # points corresponding to unseen class
    if class_to_remove is not None:
        data_plt = draw_subsamples(x_pts[y_labels == class_to_remove], subsample_pct=subsample_pct, replace=False)
        ax.scatter(data_plt[:, 0], data_plt[:, 1], zs=data_plt[:, 2],
                   c=np.asarray(colors)[class_to_remove], s=25, marker='x', depthshade=True, alpha=.9)

    # add legend
    if names is not None:
        names_keep = np.asarray(names)[classes_to_keep]
        names_keep = names_keep.tolist()
        names_legend = names_keep.copy()
        if class_to_remove is not None:
            names_legend.append('unseen class (' + names[class_to_remove] + ')')
        ax.legend(names_legend, framealpha=1)
    if s_name is not None:
        plt.savefig(s_name, bbox_inches='tight', pad_inches=0)


def plot_pts_2d(x_pts, y_labels, ax, classes_to_keep, colors,
                names=None, class_to_remove=None, s_name=None, subsample_pct=1):
    """
    Plot 2D data with class label
    :param x_pts: 2D data to plot
    :param y_labels: the corresponding y labels to the PCA data, needs to be in same classes as classes_to_keep
    :param ax: axis on which to plot data (can be combined with other plot calls, such as to show ellipses)
    :param class_to_remove: class number removed in classification
    :param classes_to_keep: array of classes to keep
    :param names: class names for legend (optional)
    :param subsample_pct: percentage of all data to show
    :param colors: colors corresponding to classes
    :param s_name: name where to save figure
    """
    # points corresponding to seen classes
    for i, class_keep in enumerate(classes_to_keep):
        data_plt = draw_subsamples(x_pts[y_labels == class_keep], subsample_pct, replace=False)
        if class_to_remove is not None:
            ax.scatter(data_plt[:, 0], data_plt[:, 1], edgecolors=np.asarray(colors)[class_keep], facecolors='None',
                       s=10, marker='o',alpha=.8)
        else:
            ax.scatter(data_plt[:, 0], data_plt[:, 1], c=np.asarray(colors)[class_keep], s=30, marker='o', alpha=.9)


    # points corresponding to unseen class
    if class_to_remove is not None:
        data_plt = draw_subsamples(x_pts[y_labels == class_to_remove], subsample_pct, replace=False)
        ax.scatter(data_plt[:, 0], data_plt[:, 1], c=np.asarray(colors)[class_to_remove], marker='x', s=60)

    # add legend
    if names is not None:
        names_keep = np.asarray(names)[classes_to_keep]
        names_keep = names_keep.tolist()
        names_legend = names_keep.copy()
        if class_to_remove is not None:
            names_legend.append('unseen class (' + names[class_to_remove] + ')')
        ax.legend(names_legend, framealpha=1)

    if s_name is not None:
        plt.savefig(s_name, bbox_inches='tight', pad_inches=0)




def plot_splits(dataset, root, minrange, maxrange, margin):
    """
    plot splits
    :param dataset: dataset
    :param root: root node
    :param minrange: minimum of range in x and y
    :param maxrange: maximum of range in x and y
    :param margin: margin at both sides
    """
    cut_vals, cut_dims = get_values_preorder(root, [], [])
    cut_vals = np.asarray(cut_vals).astype(float)
    cut_dims = np.asarray(cut_dims).astype(int)

    # show splits
    x_split = cut_vals[cut_dims == 0]
    y_split = cut_vals[cut_dims == 1]

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    plot_data(dataset, ax, title="Training data after splitting", labels=False, lines_x=x_split, lines_y=y_split,
              minrange=minrange, maxrange=maxrange, margin=margin)

    plt.axis('off')
    plt.show()


def plot_ellipses(ax, means=None, covs=None):
    """
    Overlay covariance ellipses on a 2D plot
    """
    #  covariance
    def eigsorted(cov):
        vals_, vecs_ = np.linalg.eigh(cov)
        order = vals_.argsort()[::-1]
        return vals_[order], vecs_[:, order]

    nstd = 2
    for i in range(len(covs)):
        vals, vecs = eigsorted(covs[i])
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        w, h = 2 * nstd * np.sqrt(vals)
        ell = Ellipse(xy=means[i],
                      width=w, height=h,
                      angle=theta, color='red')
        ell.set_facecolor('none')
        ax.add_artist(ell)


def export_figure_matplotlib(arr, f_name=None, dpi=200, resize_fact=1, plt_show=False, rect=None, cmap=cm.RdYlGn):
    """
    Export array as figure in original resolution
    :param arr: array of image to save in original resolution
    :param f_name: name of file where to save figure
    :param resize_fact: resize facter wrt shape of arr, in (0, np.infty)
    :param dpi: dpi of your screen
    :param plt_show: show plot or not
    :param rect: rectangle to overlay over image
    :param cmap: colormap using which to export figure
    """
    # plot
    fig = plt.figure(frameon=False)
    fig.set_size_inches(arr.shape[1] / dpi, arr.shape[0] / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(arr, cmap)

    # add rectangle overlay
    if rect is not None:
        ax.add_patch(rect)

    # save figure

    if f_name is not None:
        plt.savefig(f_name, dpi=(dpi * resize_fact))

    if plt_show:
        plt.show()
    else:
        plt.close()
    return fig


def export_particularity(dataset, img_idx, x, y, width, height, colors, confs_patches=None, names=None, dir_out=None,
                         idx=0, dpi=255):
    """
    export detail view of interesting image: image with blue rectangle, cropped region, cropped GT, cropped conf images
    """
    # Image
    class_to_remove = dataset.class_to_remove
    im = dataset.imgs[img_idx][..., :3]
    max_x = np.mod(im.shape[0], 32)
    max_y = np.mod(im.shape[1], 32)
    im = im[:-max_x, :-max_y]
    rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='b', facecolor='b', alpha=.5)
    f_name = dir_out + 'im_' + str(img_idx) + '_obj_' + str(idx) + '_im.jpg'
    _ = export_figure_matplotlib(im, dpi=dpi, rect=rect, f_name=f_name)
    plt.close()

    # Cropped Image
    im_crop = im[y:(y + height), x:(x + width)]
    f_name = dir_out + 'im_' + str(img_idx) + '_obj_' + str(idx) + '_im_crop.jpg'
    _ = export_figure_matplotlib(im_crop, dpi=dpi, f_name=f_name)
    plt.close()

    # GT
    rect = patches.Rectangle((x, y), width, height, linewidth=3, edgecolor='b', facecolor='None')
    im_gt = gt_label_to_color(dataset.gt[img_idx], colors) * 255
    max_x = np.mod(im_gt.shape[0], 32)
    max_y = np.mod(im_gt.shape[1], 32)
    im_gt = im_gt[:-max_x, :-max_y]
    f_name = dir_out + 'im_' + str(img_idx) + '_obj_' + str(idx) + '_gt.jpg'
    _ = export_figure_matplotlib(im_gt, dpi=dpi, rect=rect, f_name=f_name)
    plt.close()

    if confs_patches is not None:
        for idx_conf, conf_patches in enumerate(confs_patches):
            conf_imgs = convert_patches_to_image(dataset.imgs, conf_patches[..., np.newaxis], 64, 64)
            rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='b', facecolor='None')
            f_name = dir_out + 'im_' + str(img_idx) + '_obj_' + str(idx) + '_' + names[idx_conf] + '_wo_cl_' + str(
                class_to_remove) + '.jpg'
            export_figure_matplotlib(conf_imgs[img_idx], dpi=dpi, rect=rect, f_name=f_name)
            plt.close()


def export_pad(img, patch_size=64, stride=40, dpi=255, s_name=None, cmap=cm.rainbow):
    """
    for a given image, stride and patch size, export the padded image with overlapping patches and central strides
    :param img: original image
    :param patch_size: size of patches
    :param stride: stride between patches
    :param dpi: dpi of screen
    :param s_name: path and name where to save figure
    :param cmap: colormap for patches
    """
    # overlap visualization
    im_padded = get_padded_im(img, patch_size, stride)

    fig = plt.figure(frameon=False)
    fig.set_size_inches(im_padded.shape[1] / dpi, im_padded.shape[0] / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(im_padded[..., :3], alpha=.2)

    pad_h = int((im_padded.shape[0] - img.shape[0]) / 2)
    pad_w = int((im_padded.shape[1] - img.shape[1]) / 2)

    pad = int((patch_size - stride) / 2)

    rows, cols = get_n_patches(im_padded, stride)
    rows, cols = rows - 1, cols - 1
    color = iter(cmap(np.linspace(0, 1, rows)))
    # overlay pad total
    rect_fill = Rectangle((pad_w, pad_h), im_padded.shape[1] - (2 * pad_w), im_padded.shape[0] - (2 * pad_h),
                          edgecolor='blue', fill=False, linewidth=2)
    ax.add_patch(rect_fill)
    # overlay pad for
    rect_fill = Rectangle((pad, pad), im_padded.shape[1] - (2 * pad), im_padded.shape[0] - (2 * pad),
                          edgecolor='red', fill=False, linewidth=1, linestyle='--')

    ax.add_patch(rect_fill)
    # vertical patches
    for x in range(1, rows):
        color_win = next(color)
        # patch
        rect = Rectangle((0, x * stride), patch_size, patch_size, linewidth=1, color=color_win, fill=False)
        ax.add_patch(rect)
        # central part
        rect = Rectangle((pad, pad + (x * stride)), stride, stride, linewidth=1, facecolor=color_win, edgecolor='black',
                         fill=True)
        ax.add_patch(rect)

    color = iter(cmap(np.linspace(0, 1, cols)))
    # horizontal patches
    for x in range(cols):
        color_win = next(color)
        # patch
        rect = Rectangle((x * stride, 0), patch_size, patch_size, linewidth=1, color=color_win, fill=False)
        ax.add_patch(rect)
        # central part
        rect = Rectangle((pad + (x * stride), pad), stride, stride, linewidth=1, facecolor=color_win, edgecolor='black',
                         fill=True)
        ax.add_patch(rect)

    # Add the patch to the Axes
    ax.add_patch(rect_fill)
    if s_name is not None:
        fig.savefig(s_name, dpi=dpi)
