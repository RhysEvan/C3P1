import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

import numpy as np




def intersection_between_2_lines(L1, L2):
    """
    Compute the intersection point and distance between two lines in 3D space.

    Parameters:
    L1, L2 : Line objects
        Line objects containing start points and direction vectors.

    Returns:
    Point : numpy array of shape (3,)
        The midpoint of the closest points on the two lines.
    distance : float
        The shortest distance between the two lines.
    """
    p1, u = L1.Ps, L1.V
    p3, v = L2.Ps, L2.V
    w = (p1 + u) - (p3 + v)
    b = np.sum(u * v, axis=1)
    d = np.sum(u * w, axis=1)
    e = np.sum(v * w, axis=1)
    D = 1 - b ** 2

    # Avoid division by zero for parallel lines
    D[D == 0] = np.finfo(float).eps

    sc = (b * e - d) / D
    tc = (e - b * d) / D

    point1 = p1 + u + (sc[:, np.newaxis] * u)
    point2 = p3 + v + (tc[:, np.newaxis] * v)

    Points = (point1 + point2) / 2

    dP = w + (sc[:, np.newaxis] * u) - (tc[:, np.newaxis] * v)
    distances = np.linalg.norm(dP, axis=1)

    return Points, distances

class Line:
    kind = 'Line'  # class variable shared by all instances

    def __init__(self, *args):
        if len(args) == 0:
            self.Ps = None
            self.Pe = None
        elif len(args) == 1:
            self.Plucker = args[0]
        elif len(args) == 2:
            self.Ps = args[0]
            self.Pe = args[1]
        else:
            raise ValueError("Invalid number of arguments!")

    @property
    def V(self):
        if self.Ps is not None and self.Pe is not None:
            return self._normalize_vectors(self.Pe - self.Ps)
        return None

    @V.setter
    def V(self, value):
        if self.Ps is not None:
            self.Pe = value + self.Ps

    @property
    def Plucker(self):
        V = self.V
        U = np.cross(self.Ps, self.Ps + V)
        return np.hstack((V, U))

    @Plucker.setter
    def Plucker(self, value):
        Vp = value[:, :3]
        Up = value[:, 3:]
        self.Ps = np.cross(Vp, Up)
        self.V = Vp

    @property
    def Plucker2(self):
        m = np.cross(self.Ps, self.Pe)
        return np.hstack((m, self.Pe - self.Ps))

    @Plucker2.setter
    def Plucker2(self, value):
        a = value[:, :3]
        b = value[:, 3:]
        self.Ps = np.column_stack((-a[:, 1] / b[:, 2], a[:, 0] / b[:, 2], np.zeros(a.shape[0])))
        self.V = b

    def GetAngle(self):
        v = np.array([0, 0, 1])
        ThetaInDegrees = np.zeros((self.Ps.shape[0], 1))
        for i in range(self.Ps.shape[0]):
            ThetaInDegrees[i, 0] = np.degrees(np.arctan2(np.linalg.norm(np.cross(self.V[i, :], v)), np.dot(self.V[i, :], v)))
        return ThetaInDegrees

    def TransformLines(self, H):

        self.Ps = H.transform(self.Ps)
        self.Pe = H.transform(self.Pe)


    def plot(self, limits=None, colors=None, linewidth=2, linestyle='-'):
        if limits is None:
            limits = [-5, 5, -5, 5, -5, 5]
        if colors is None:
            colors = plt.cm.jet(np.linspace(0, 1, self.Ps.shape[0]))
        elif colors.shape[0] < self.Ps.shape[0]:
            colors = np.tile(colors, (self.Ps.shape[0], 1))

        xmin, xmax, ymin, ymax, zmin, zmax = limits
        dirs = self.V
        k_min = (np.array([xmin, ymin, zmin]) - self.Ps) / dirs
        k_max = (np.array([xmax, ymax, zmax]) - self.Ps) / dirs

        valid_x_min = self._is_within_bounds(self.Ps + dirs * k_min[:, 0], ymin, ymax, zmin, zmax)
        valid_y_min = self._is_within_bounds(self.Ps + dirs * k_min[:, 1], xmin, xmax, zmin, zmax)
        valid_z_min = self._is_within_bounds(self.Ps + dirs * k_min[:, 2], xmin, xmax, ymin, ymax)

        valid_x_max = self._is_within_bounds(self.Ps + dirs * k_max[:, 0], ymin, ymax, zmin, zmax)
        valid_y_max = self._is_within_bounds(self.Ps + dirs * k_max[:, 1], xmin, xmax, zmin, zmax)
        valid_z_max = self._is_within_bounds(self.Ps + dirs * k_max[:, 2], xmin, xmax, ymin, ymax)

        valid = np.column_stack((valid_x_min, valid_y_min, valid_z_min, valid_x_max, valid_y_max, valid_z_max))
        assert np.all(np.sum(valid, axis=1) == 2), 'Not all lines fit in window range!'

        k = np.column_stack((k_min, k_max))
        k_valid = k[valid].reshape(2, -1).T
        start_points = self.Ps + dirs * k_valid[:, 0]
        end_points = self.Ps + dirs * k_valid[:, 1]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(start_points.shape[0]):
            ax.plot([start_points[i, 0], end_points[i, 0]],
                    [start_points[i, 1], end_points[i, 1]],
                    [start_points[i, 2], end_points[i, 2]],
                    color=colors[i], linewidth=linewidth, linestyle=linestyle)
        plt.show()

    def PlotLine(self, colori = 'g', linewidth = 2, *args):
        P1 = self.Ps
        if len(args) == 1:
            P3 = self.Pe
        else:
            P3 = self.Ps + self.V

        if P1.shape[0] > 500:
            P1 = self._downsample(P1, round(P1.shape[0] / 500) + 1)
            P3 = self._downsample(P3, round(P3.shape[0] / 500) + 1)
            print('Too many lines to plot, showing downsampled version')

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(P1.shape[0]):
            color = 'b' if i == 0 or i == 4 else 'k' if i == 4 else 'b'
            ax.plot([P1[i, 0], P3[i, 0]], [P1[i, 1], P3[i, 1]], [P1[i, 2], P3[i, 2]], color=color, linewidth=linewidth)
            if i > 500:
                print('Too many lines to plot')
                break
        plt.show()

    def FindXYZNearestLine(self, XYZ):
        distances = self.DistanceLinePoint(XYZ)
        return np.argmin(distances)

    def FitLine(self, XYZ):
        l = self._fitline3d(XYZ.T).T
        self.Ps = l[0]
        self.Pe = l[1]

    def FitLineRansac(self, XYZ, t=10):
        Ps, Pe = self._ransac_fit_line(XYZ.T, t)
        self.Ps = Ps.T
        self.Pe = Pe.T

    def NormaliseLine(self):
        try:
            scale = -self.Ps[:, 2] / self.V[:, 2]
            self.Ps = self.Ps + scale[:, np.newaxis] * self.V
        except:
            pass

    def DistanceLinePoint(self, XYZ):
        return np.linalg.norm(np.cross(self.Pe - self.Ps, self.Ps - XYZ), axis=1) / np.linalg.norm(self.Pe - self.Ps, axis=1)

    def Lenght(self):
        return np.linalg.norm(self.Ps - self.Pe, axis=1)

    @staticmethod
    def FromStartEnd(start_point, end_point):
        line_object = Line()
        line_object.Ps = start_point
        line_object.Pe = end_point
        return line_object

    @staticmethod
    def FromPlucker(VU):
        line_object = Line()
        line_object.Plucker = VU
        return line_object

    def _normalize_vectors(self, vectors):
        norms = np.linalg.norm(vectors, axis=1)
        return vectors / norms[:, np.newaxis]

    def _is_within_bounds(self, points, xmin, xmax, ymin, ymax, zmin, zmax):
        return (points[:, 0] >= xmin) & (points[:, 0] <= xmax) & \
               (points[:, 1] >= ymin) & (points[:, 1] <= ymax) & \
               (points[:, 2] >= zmin) & (points[:, 2] <= zmax)

    def _downsample(self, points, factor):
        return points[::factor]

    def _fitline3d(self, XYZ):
        # Placeholder for the actual fitline3d implementation
        pass
    def AngleBetweenLines(self,L1, L2):
        """ Returns the angle in radians between vectors 'L1' and 'L2'    """

        T1 = (np.cross(L1.V, L2.V)),
        T2 = (np.dot(L1.V, np.transpose(L2.V)))
        T1R = np.linalg.norm((T1))
        Theta = np.arctan2(T1R,T2)
        ThetaDegree = Theta * 180 / np.pi
        return (Theta,ThetaDegree)

    def _ransac_fit_line(self, XYZ, t):
        # Placeholder for the actual RANSAC fit line implementation
        pass

    def HomogeneousTransformation(self, H, points):
        # Placeholder for the actual homogeneous transformation implementation
        pass

    def GenerateRay(self, I, uv):
        # Generate rays, this builds a line object stored in 'rays' and a copy of in
        rays = Line()
        Ps = uv.copy()  # sensor points

        # Adjust sensor points
        Ps[:, 0] = Ps[:, 0] - I.cx
        Ps[:, 1] = Ps[:, 1] - I.cy
        schaal = 2 / 1  # just so the 'sensor' is not bigger than the object (only for visualization)
        Ps[:, 0] = Ps[:, 0] / I.fx * schaal
        Ps[:, 1] = Ps[:, 1] / I.fy * schaal  # f van Y
        Ps[:, 2] = 1 * schaal

        Pf = np.zeros(Ps.shape)
        Pf[:, 2] = 0 * schaal

        rays.Pe = Ps
        rays.Ps = Pf

        # Uncomment the following lines to visualize the rays
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.plot(rays.Pe[:, 0], rays.Pe[:, 1], rays.Pe[:, 2], 'g', linewidth=5)
        # ax.plot(rays.Ps[:, 0], rays.Ps[:, 1], rays.Ps[:, 2], 'g', linewidth=5)
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # ax.set_zlabel('z')
        # ax.set_aspect('equal')
        # plt.show()

        return rays

if __name__ == '__main__':
    L = Line()
    L.Ps = np.array([[1, 1, 0]])
    L.Pe = np.array([[2, 1, 0], ])
    print(L.V)
    L.PlotLine()
    L2 = Line()
    L2.Ps = np.array([[0, 0, 0]])
    L2.Pe = np.array([[20, 20, 0], ])
    print(L2.V)
    Hoek, HoekDegree = L.AngleBetweenLines(L, L2)
    print(HoekDegree)
    print(Hoek)
