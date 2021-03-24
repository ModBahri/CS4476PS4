"""
Draw polygon regions of interest (ROIs) in matplotlib images,
similar to Matlab's roipoly function.

Created by Joerg Doepfert 2014 based on code posted by Daniel
Kornhauser.

Modified by Jiasen Lu for PS4.
"""
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


class roipoly:
    def __init__(self, fig=None, ax=None, color="b"):
        if fig is None:
            fig = plt.gcf()

        if ax is None:
            ax = plt.gca()

        self.previous_point = []
        self.all_x_points = []
        self.all_y_points = []
        self.start_point = []
        self.end_point = []
        self.line = None
        self.color = color
        self.fig = fig
        self.ax = ax

        self.__ID1 = self.fig.canvas.mpl_connect(
            "motion_notify_event", self.__motion_notify_callback
        )
        self.__ID2 = self.fig.canvas.mpl_connect(
            "button_press_event", self.__button_press_callback
        )

        if sys.flags.interactive:
            plt.show(block=False)
        else:
            plt.show()

    def get_indices(self, im, pos):
        # given the input, return the index of positions
        grid = self.get_mask(im)
        pos = np.round(pos)

        indices = []
        for i in range(pos.shape[0]):
            m = np.int(pos[i, 0])
            n = np.int(pos[i, 1])
            if grid[n, m]:
                indices.append(i)
        return indices

    def get_mask(self, currentImage):
        ny, nx, _ = np.shape(currentImage)
        poly_verts = [(self.all_x_points[0], self.all_y_points[0])]
        for i in range(len(self.all_x_points) - 1, -1, -1):
            poly_verts.append((self.all_x_points[i], self.all_y_points[i]))

        # Create vertex coordinates for each grid cell...
        # (<0,0> is at the top left of the grid in this system)
        x, y = np.meshgrid(np.arange(nx), np.arange(ny))
        x, y = x.flatten(), y.flatten()
        points = np.vstack((x, y)).T
        path = matplotlib.path.Path(poly_verts)
        grid = path.contains_points(points).reshape((ny, nx))
        return grid

    def display_roi(self, **kwargs):
        line = plt.Line2D(
            self.all_x_points + [self.all_x_points[0]],
            self.all_y_points + [self.all_y_points[0]],
            color=self.color,
            **kwargs
        )
        ax = plt.gca()
        ax.add_line(line)
        plt.draw()

    def display_mean(self, currentImage, **kwargs):
        mask = self.get_mask(currentImage)
        meanval = np.mean(np.extract(mask, currentImage))
        stdval = np.std(np.extract(mask, currentImage))
        string = "%.3f +- %.3f" % (meanval, stdval)
        plt.text(
            self.all_x_points[0],
            self.all_y_points[0],
            string,
            color=self.color,
            bbox=dict(facecolor="w", alpha=0.6),
            **kwargs
        )

    def __motion_notify_callback(self, event):
        if event.inaxes:
            x, y = event.xdata, event.ydata
            if (
                event.button is None or event.button == 1
            ) and self.line is not None:  # Move line around
                self.line.set_data(
                    [self.previous_point[0], x], [self.previous_point[1], y]
                )
                self.fig.canvas.draw()

    def __button_press_callback(self, event):
        if event.inaxes:
            x, y = event.xdata, event.ydata
            ax = event.inaxes
            if (
                event.button == 1 and not event.dblclick
            ):  # If you press the left button, single click
                if self.line is None:  # if there is no line, create a line
                    self.line = plt.Line2D([x, x], [y, y], marker="o", color=self.color)
                    self.start_point = [x, y]
                    self.previous_point = self.start_point
                    self.all_x_points = [x]
                    self.all_y_points = [y]

                    ax.add_line(self.line)
                    self.fig.canvas.draw()
                    # add a segment
                else:  # if there is a line, create a segment
                    self.line = plt.Line2D(
                        [self.previous_point[0], x],
                        [self.previous_point[1], y],
                        marker="o",
                        color=self.color,
                    )
                    self.previous_point = [x, y]
                    self.all_x_points.append(x)
                    self.all_y_points.append(y)

                    event.inaxes.add_line(self.line)
                    self.fig.canvas.draw()
            elif (
                (event.button == 1 and event.dblclick)
                or (event.button == 3 and not event.dblclick)
            ) and self.line is not None:  # close the loop and disconnect
                self.fig.canvas.mpl_disconnect(self.__ID1)  # joerg
                self.fig.canvas.mpl_disconnect(self.__ID2)  # joerg

                self.line.set_data(
                    [self.previous_point[0], self.start_point[0]],
                    [self.previous_point[1], self.start_point[1]],
                )
                ax.add_line(self.line)
                self.fig.canvas.draw()
                self.line = None

                if sys.flags.interactive:
                    pass
                else:
                    # figure has to be closed so that code can continue
                    plt.close(self.fig)
