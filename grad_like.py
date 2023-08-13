import cv2
import analysis
from analysis import ImageAnalysis
import numpy as np
from t1cv import *
import cProfile
import hist_check

def interpolate_color(colors):
    if not isinstance(colors, np.ndarray):
        colors = np.uint8(colors)
    def interpolate_color_func_over_iterations(iterations):
        def interpolate_color_func(i):
            t = i / iterations
            return colors[0] * (1 - t) + colors[1] * t
        return interpolate_color_func
    return interpolate_color_func_over_iterations

# colorsUp = interpolate_color(((150, 0, 100), (0, 255, 50)))
# colorsDown = interpolate_color(((150, 100, 0), (0, 50, 255)))
colorsUp = interpolate_color(((0, 0, 0), (0, 255, 0)))
colorsDown = interpolate_color(((0, 0, 0), (0, 0, 255)))

def stepf_generate(stepsize):
    if type(stepsize) in (float, int):
        stepsize = np.array((stepsize, stepsize))
    stepf = interpolate_color(stepsize)
    return stepf





class ImageGradStats:
    def __init__(self, IA: ImageAnalysis, stepsize = 0.007161179):
        self.IA:ImageAnalysis = IA
        self.laplacian_stored = None
        self.sobel_stored = None
        self.filter2d_stored = None
        self.filter2d_kernel_stored = None
        self.gray = self.IA.gray.copy()
        self.default_iterations=100
        self.default_stepsize = stepsize
        self.default_stepf = stepf_generate(self.default_stepsize)
        self.default_interval = 10
        self.blur_gray()
        
    def reset_gray(self):
        self.gray = self.IA.gray.copy()

    def blur_gray(self, ksize = 5):
        self.gray :np.ndarray = hist_check.apply_multi_low_pass(self.gray)
        self.gray :np.ndarray = cv2.GaussianBlur(self.gray, (ksize, ksize), 0)


    def sobel(self):
        return cv2.Sobel(self.gray, cv2.CV_64F, 1, 1, ksize=5)
    
    def dgrad(self, gray = None):
        gray = np.float64(self.gray) if gray is None else gray
        x = (np.roll(gray, 1, axis=0) - np.roll(gray, -1, axis=0))/2# + np.roll(self.gray, -1, axis=0)
        y = (np.roll(gray, 1, axis=1) - np.roll(gray, -1, axis=1))/2# + np.roll(self.gray, -1, axis=1)
        return np.dstack((x, y))
    
    def npdgrad(self):
        gray = np.float64(self.gray)
        x = np.gradient(gray, axis=0)
        y = np.gradient(gray, axis=1)
        return np.dstack((x, y))
    
    def dgrad_positive_side(self):
        gray = np.int16(self.gray)
        x = gray - np.roll(self.gray, 1, axis=0)# + np.roll(self.gray, -1, axis=0)
        y = gray - np.roll(self.gray, 1, axis=1)# + np.roll(self.gray, -1, axis=1)
        return np.dstack((x, y))
    
    def d2grad(self):
        grad = self.npdgrad()
        x = np.gradient(grad, axis=0)
        y = np.gradient(grad, axis=1)

        dxx = x[:,:,0]
        dxy = x[:,:,1]
        dyx = y[:,:,0]
        dyy = y[:,:,1]
        return np.dstack(((dxx, dxy), (dyx, dyy)))
    
    def d2gradx2y2(self):
        gray = np.float64(self.gray)
        x = np.gradient(grad, axis=0)
        y = np.gradient(grad, axis=1)
        dx = np.gradient(x, axis=0)
        y = np.gradient(grad, axis=1)

        return np.dstack(())

    def sobelxy(self, ksize = 5):
        b = np.float64(self.gray)
        # b = cv2.GaussianBlur(b, (ksize, ksize), 0)
        x = cv2.Sobel(b, cv2.CV_64F, 1, 0, ksize=ksize)
        y = cv2.Sobel(b, cv2.CV_64F, 0, 1, ksize=ksize)
        #return array shaped as (w, h, 2)
        x_b = cv2.GaussianBlur(x[:,:], (5,5), 0)
        y_b = cv2.GaussianBlur(y[:,:], (5,5), 0)
        return np.dstack((y_b, x_b)) # opencv backwards lol

    # def hillroll(self, landscape, xy, iterations:int = 100, stepsize = 0.001, canvas = None, color = None, normalize = False):
    #     xy = np.float64(xy)
    #     lines = []
    #     points = []
    #     color = np.float64(color)
    #     canvas = np.zeros_like(landscape) if canvas is None else canvas
    #     if len(color.shape) > 1:
    #         endcolor = color[1]
    #         startcolor = color[0]
    #     else:
    #         startcolor = color
    #         endcolor = color

    #     for i in range(iterations):
    #         x = int(xy[0])
    #         y = int(xy[1])
    #         try:
    #             direction = landscape[x, y]
    #         except IndexError:
    #             break
    #         if normalize:d
    #             direction = direction / (np.linalg.norm(irection) + 0.01)
    #         xy += direction * stepsize
    #         points += [xy]
    #         print(xy)
    #         color = startcolor + (endcolor - startcolor) * i / iterations
    #         canvas[x, y] = 255 if color is None else color
    #         # lines.append(self.hillroll_line(x, y))
    #     return points

    def draw_vector_field(self, landscape, canvas = None, color = None, scale = 0.1):
        canvas = np.zeros_like(landscape) if canvas is None else canvas
        canvas = np.array(canvas)
        for x in range(0, landscape.shape[0], 20):
            for y in range(0, landscape.shape[1], 20):
                direction = landscape[x, y] * scale
                print(direction)
                cv2.arrowedLine(canvas, (x, y), (x + int(direction[0]), y + int(direction[1])), (150, 50, 0), )
        return canvas 
    def do_rollfield(self, landscape, canvas = None, color = None, stepsize = 0.001, interval = 50, iterations = 200, normalize = False):
        canvas = np.zeros_like(landscape) if canvas is None else canvas
        for x in range(0, landscape.shape[0], interval):
            for y in range(0, landscape.shape[1], interval):

                self.hillroll(landscape, (x, y), canvas = canvas, color = color, stepsize=stepsize, iterations=iterations, normalize=normalize)
        return canvas
    
    def hillroll_line(self, x, y):
        line = []
        for i in range(100):
            line.append((x, y))
            x += int(self.laplacian_stored[x, y][0])
            y += int(self.laplacian_stored[x, y][1])
        return line
    

    def filter2d(self, kernel):
        self.filter2d_stored = cv2.filter2D(self.gray, cv2.CV_32F, kernel)
        # cv2.filter2D()
        return cv2.filter2D(self.gray, -1, kernel)
    

    

    def get_laplacian(self):
        blur = cv2.GaussianBlur(self.gray, (3,3), 0)
        self.laplacian_stored = cv2.Laplacian(blur, cv2.CV_64F)
        return self.laplacian_stored
    
    def hillroll_points_iterate(self, landscape, points, iterations = 100, stepsize = 0.001, canvas = None, color = None, normalize = False):
        assert points.shape[1:] == (2,)
        points = np.float64(points)
        canvas = np.zeros_like(landscape) if canvas is None else canvas
        pointints = points.astype(np.int16)
        print(pointints)

        stepsize = stepsize(iterations)
        color = color(iterations)
        dirs = np.zeros_like(points)
        active = np.ones_like(points[:,0]).astype(bool)
        # active[2:5] = True
        print(active.shape)
        for i in range(iterations):
            print(1)
            pointints[active,:]
            active_points = points[active]
            dirs = landscape[pointints[active,0],pointints[active,1], :] * stepsize(i)
            print(dirs.shape)
            print(active_points)
            # print(active_points.shape, dirs.shape, dirs[active].shape, pointints[active].shape, landscape[pointints[active,0], pointints[active, 1], :].shape, pointints[active,:].shape)
            # dirs[active] = landscape[pointints[active,0], pointints[active,1], :] * stepsize(i)
            points[active,:] += dirs
            active[points[:,0] < 0] = False
            active[points[:,1] < 0] = False
            active[points[:,0] > landscape.shape[0]] = False
            active[points[:,1] > landscape.shape[1]] = False
            # points[active, :] += dirs
            pointints[active, :] = np.dstack((np.clip(points[active,0], 0, landscape.shape[0] - 1), np.clip(points[active,1], 0, landscape.shape[1] - 1)))
            canvas[pointints[:,0], pointints[:,1]] += color(i)/50

    def active_points(self, points, active=None):
        active = np.ones_like(points[:,0]).astype(bool) if active is None else active
        active[points[:,0] < 0] = False
        active[points[:,1] < 0] = False
        active[points[:,0] >= self.gray.shape[0]] = False
        active[points[:,1] >= self.gray.shape[1]] = False
        return active


    def hillroll_points_iterate_velocity(self, landscape, points, velocities = None, iterations = 100, velocity_decay = 0.9, stepsize = 0.001, canvas = None, color = None, normalize = False, active=None, **kwargs):
        if kwargs != {}:
            print("hillroll warning: kwargs keys=", kwargs.keys(), " are not used")
        assert points.shape[1:] == (2,)
        points = np.float64(points) if points.dtype != np.float64 else points
        velocities = np.zeros_like(points) if velocities is None else velocities
        velocities = np.float64(velocities) if velocities.dtype != np.float64 else velocities
        if velocities.shape != points.shape:
            if velocities.shape == (2,):
                velocities = np.tile(velocities, (points.shape[0], 1))
        canvas = np.zeros_like(landscape) if canvas is None else canvas
        pointints = points.astype(np.int16)
        velocity_decay = 0.9 if velocity_decay is None else velocity_decay
        stepsize = stepsize(iterations)
        color = color(iterations)
        dirs = np.zeros_like(points)
        show = kwargs.get("show", False) if "show" in kwargs else False
        addcolors = False if "addcolors" not in kwargs else kwargs["addcolors"]
        # active = np.ones_like(points[:,0]).astype(bool)
        # active[2:5] = True
        active = self.active_points(points, active)
        for i in range(iterations):
            pointints[active,:]
            dirs = landscape[pointints[active,0],pointints[active,1], :] * stepsize(i)
            velocities[active,:] *= velocity_decay
            velocities[active,:] += dirs
            points[active,:] += velocities[active,:]
            
            active = self.active_points(points, active)
            pointints[active, :] = np.dstack((np.clip(points[active,0], 0, landscape.shape[0] - 1), np.clip(points[active,1], 0, landscape.shape[1] - 1)))
            if addcolors:
                canvas[pointints[active,0], pointints[active,1], 1:] += color(i)[1:].astype(canvas.dtype)
            else:
                canvas[pointints[active,0], pointints[active,1], 1:] = np.uint8(color(i)[1:])
            if show:
                self.IA.show(canvas)
        return {'points': points, 'velocities': velocities, 'canvas': canvas, 'active': self.active_points(pointints), "pointints" : pointints}
    #
    #  def hillroll_points(self, landscape, points = None, iterations = 100, stepsize = 0.001, canvas = None, color = None, normalize = False):
    #     assert points.shape[1:] == (2,)
    #     canvas = np.zeros_like(landscape) if canvas is None else canvas
    #     pointints = points.astype(np.int16)
    #     stepsize = stepsize(iterations)
    #     for i in iterations:
    #         dirs += landscape[pointints[:,0], pointints[:,1]] * stepsize(i)
    #         points += dirs
    #         pointints[:,:] = np.dstack((np.clip(points[:, 0], 0, landscape.shape[0] - 1), np.clip(points[:, 1], 0, landscape.shape[1] - 1)))
    #         canvas[pointints[:,0], pointints[:,1]] = color if type(color) is not function else color(i)
    
    def hillroll_points(self, points=None, up=False, both=False, **kwargs):
        interval = 10 if "interval" not in kwargs else kwargs["interval"]
        points = np.mgrid[0: self.gray.shape[0]: interval, 0: self.gray.shape[1]: interval].reshape(2, -1).T if points is None else points
        iterations=100 if "iterations" not in kwargs else kwargs["iterations"]
        stepsize =  self.default_stepsize if "stepsize" not in kwargs else kwargs["stepsize"]
        stepf = stepf_generate(stepsize) if "stepf" not in kwargs else kwargs["stepf"]
        nstepf = stepf_generate(-1 * stepsize) if "nstepf" not in kwargs else kwargs["nstepf"]
        dgrad = self.npdgrad()
        sobel = self.sobelxy(ksize=3) if "landscape" not in kwargs else kwargs["landscape"]
        landscape = dgrad
        landscape = (sobel + dgrad) / 2
        canvas = self.IA.img.copy() if "canvas" not in kwargs else kwargs["canvas"]
        if up:
            d = {'landscape' : landscape, 'points' : points,  'iterations':iterations, 'stepsize':stepsize, 'canvas':canvas, 'color' : colorsUp, 'stepsize' : stepf}
            d.update(kwargs)
            return self.hillroll_points_iterate_velocity(**d)
        if both or not up:
            d = {'landscape' : landscape, 'points' : points, 'iterations':iterations, 'stepsize':stepsize, 'canvas':canvas, 'color' : colorsDown, 'stepsize' : nstepf}
            d.update(kwargs)
            return self.hillroll_points_iterate_velocity(**d)
 


def main():
    IA = ImageAnalysis('437-1-03.tif')
    IA = ImageAnalysis('244773_02.tif')
    IA = ImageAnalysis('242316_01.tif')

    gs = ImageGradStats(IA)
    iterations=100
    stepsize = 0.007161179
    stepf = stepf_generate(stepsize)
    nstepf = stepf_generate(-1 * stepsize)
    interval = 10
    landscape = gs.dgrad()
    landscape = gs.sobelxy(ksize=3)

    # IA = ImageAnalysis('437-1-03.tif')
    print(gs.get_laplacian().shape)
    print(gs.sobel().shape)
    print(gs.filter2d(np.array([[1,1,1],[1,1,1],[1,1,1]])).shape)
    # show(gs.sobel())
    l = gs.get_laplacian()
    kernel = np.array([[1,1,1],[1,1,1],[1,1,1]])
    # b = cv2.GaussianBlur(l, (5,5), 0)
    erode = cv2.erode(l, kernel, iterations=1)

    print(erode.dtype)
    

    canvas = IA.img.copy()
    # landscape = landscape[::-1,::-1]
    # canvas = canvas[::-1,::-1]
    # vf = gs.draw_vector_field(landscape, canvas = canvas, scale = 0.1)
    # # gs.hillroll_points_iterate(landscape, points, canvas = canvas, color = colorsDown, stepsize = nstepf, iterations=iterations)

    # # gs.hillroll_points_iterate(landscape, points, canvas = canvas, color = colorsDown, stepsize = nstepf, iterations=iterations)
    # show(canvas)
    canvas = IA.img.copy()
    gs.hillroll_points(canvas = canvas, velocities = np.array([0,0]), iterations = 300)
    show(canvas)
    # canvas = IA.img.copy()
    gs.hillroll_points_iterate_velocity(landscape, points, canvas = canvas, color = colorsDown, stepsize = nstepf, iterations=iterations)
    show(canvas)
if __name__ == '__main__':
    main()





def darken_edge_points(imagestr, **kwargs):
    def color_half_function(iterations):
        dark = np.array((0, 0, 255), dtype=np.uint8)
        def color(i):
            if i > iterations / 2:
                return dark * i / iterations
            else:
                return dark * 0
        return color
    IA = ImageAnalysis(imagestr)

    gs = ImageGradStats(IA)
    iterations=100
    stepsize = 0.007161179
    stepf = stepf_generate(stepsize)
    nstepf = stepf_generate(-1 * stepsize)
    interval = 4
    landscape = gs.dgrad()
    landscape = gs.sobelxy(ksize=3)
    colors = interpolate_color(((0, 0, 0), (0, 0, 2)))
    # IA = ImageAnalysis('437-1-03.tif')
    print(gs.get_laplacian().shape)
    print(gs.sobel().shape)
    print(gs.filter2d(np.array([[1,1,1],[1,1,1],[1,1,1]])).shape)
    # show(gs.sobel())
    l = gs.get_laplacian()
    kernel = np.array([[1,1,1],[1,1,1],[1,1,1]])
    # b = cv2.GaussianBlur(l, (5,5), 0)
    erode = cv2.erode(l, kernel, iterations=1)

    print(erode.dtype)
    

    canvas = np.float64(IA.img.copy())
    # landscape = landscape[::-1,::-1]
    # canvas = canvas[::-1,::-1]
    # vf = gs.draw_vector_field(landscape, canvas = canvas, scale = 0.1)
    # # gs.hillroll_points_iterate(landscape, points, canvas = canvas, color = colorsDown, stepsize = nstepf, iterations=iterations)

    # # gs.hillroll_points_iterate(landscape, points, canvas = canvas, color = colorsDown, stepsize = nstepf, iterations=iterations)
    # show(canvas)
    # canvas = IA.img.copy()
    d = {"addcolors" : True, "color" : color_half_function, "iterations":iterations, "interval":interval, "velocity_decay" : 0.99}
    d = {** d, **kwargs}
    d1 = gs.hillroll_points(canvas = canvas, velocities = np.array([0,0]))
    # show(canvas)
    darkened = canvas
    dark_max_min = np.max(darkened, axis = (0,1)), np.min(darkened, axis = (0,1))
    darkened = (darkened - dark_max_min[1]) / (dark_max_min[0] - dark_max_min[1])
    # IA.show(darkened * 255)
    IA.show(darkened, ms = 100)
    return darkened, IA.gray.copy(), d1
    # canvas = IA.img.copy()
    show(canvas)




def edge_points(imagestr):
    IA = ImageAnalysis(imagestr)

    gs = ImageGradStats(IA)
    iterations=100
    stepsize = 0.007161179
    stepf = stepf_generate(stepsize)
    nstepf = stepf_generate(-1 * stepsize)
    interval = 5
    landscape = gs.dgrad()
    landscape = gs.sobelxy(ksize=3)

    # IA = ImageAnalysis('437-1-03.tif')
    print(gs.get_laplacian().shape)
    print(gs.sobel().shape)
    print(gs.filter2d(np.array([[1,1,1],[1,1,1],[1,1,1]])).shape)
    # show(gs.sobel())
    l = gs.get_laplacian()
    kernel = np.array([[1,1,1],[1,1,1],[1,1,1]])
    # b = cv2.GaussianBlur(l, (5,5), 0)
    erode = cv2.erode(l, kernel, iterations=1)

    print(erode.dtype)
    

    canvas = IA.img.copy()
    # landscape = landscape[::-1,::-1]
    # canvas = canvas[::-1,::-1]
    # vf = gs.draw_vector_field(landscape, canvas = canvas, scale = 0.1)
    # # gs.hillroll_points_iterate(landscape, points, canvas = canvas, color = colorsDown, stepsize = nstepf, iterations=iterations)

    # # gs.hillroll_points_iterate(landscape, points, canvas = canvas, color = colorsDown, stepsize = nstepf, iterations=iterations)
    # show(canvas)
    canvas = IA.img.copy()
    d = gs.hillroll_points(canvas = canvas, velocities = np.array([0,0]), )
    show(canvas)
    # canvas = IA.img.copy()
    # gs.hillroll_points_iterate_velocity(landscape, points, canvas = canvas, color = colorsDown, stepsize = nstepf, iterations=iterations)
    show(canvas)
