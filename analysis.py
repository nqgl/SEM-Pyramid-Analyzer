from t1cv import *
import json
# from regions import (regions.RegionContext, Region)
# import regions
import scipy
from scipy import signal
class ImageAnalysis():
    def __init__(self, imgstr):
        self.imgstr = imgstr
        if not os.path.exists(imgstr):
            self.imgstr = 'images_in/' + imgstr
        self.fullimg :np.ndarray = cv2.imread(self.imgstr)
        self.img :np.ndarray = self.fullimg[:-77,:]
        self.titlebar :np.ndarray = self.fullimg[-76:,:]
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY) 
        # gray = np.average(np.dstack((gray, cv2.imread('darkened_w_circles.png', cv2.IMREAD_GRAYSCALE))), axis = 2)
        self.img = np.dstack((self.gray, self.gray, self.gray))
        # self.graywave = w2d(cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY))
        self.imgwidth = self.img.shape[1]
        self.guibar_original = np.zeros((200, self.imgwidth, 3), dtype=np.uint8) + 255
        self.guibar = self.guibar_original.copy()

        self.render_canvas = self.zeros_color()
        # self.thresholds = get_multiple_threshold_levels(self.gray)
        # self.labels_mask_tuple_list = [label_mask(thresh) for thresh in self.thresholds]
        # self.labels_masks = [masks_from_label(l) for l in self.labels_mask_tuple_list]
        # self.cms = [get_contour_masks(t)[2] for t in self.thresholds]
        self.thresh_value = 110
        self.contour_masks = None #self.make_contour()
        self.region_contexts = [] #self.generate_regions()
        self.rotation = 0
        self.cross_value_landscape_stored = None
        self.claimed_territory_stored = None
        self.reset_canvas()
        self.laplacian_stored = None
        # cv2.destroyAllWindows()
        self.window = cv2.namedWindow('Image Analysis')
        self.capture = None
        self.pyramids = []
        self.pyramids_by_id = {}
        self.last_show = self.img.copy()
        self.prominence = 35

    def save_clipped(self):
        name = self.imgstr.split('/')[-1]
        name, ext = name.split('.')
        clipped_path =f'clipped/{name}_clipped.{ext}'
        clipped_path_gray =f'clipped/{name}_clipped_gray.{ext}'
        cv2.imwrite(clipped_path, self.img)
        cv2.imwrite(clipped_path_gray, self.gray)
        return clipped_path, clipped_path_gray


    # def generate_regions(self):
    #     self.region_contexts += [regions.RegionContext(self, self.contour_masks)]

    def show(self, img = None, ms=1, frames = 1, dont_save_last = False, titlebar = None):
        if img is None:
            img = self.last_show 
        elif not dont_save_last:
                self.last_show = img
        if len(img.shape) == 2 or img.shape[2] == 1:
            img = np.dstack((img, img, img)).reshape(-1, self.imgwidth, 3)
        titlebar = self.titlebar if titlebar is None else titlebar
        img = np.concatenate((img, titlebar, self.guibar), axis = 0)
        cv2.imshow('Image Analysis', img)
        if self.capture is not None:
            self.capture += [img.copy()] *1
        print("show", ms, "ms")
        if ms > 0:
            ms = 1
            cv2.waitKey(ms)
    def zeros(self):
        return np.uint8(np.zeros_like(self.gray))

    def zeros_color(self):
        return np.zeros_like(self.img)
    
    def reset_canvas(self):
        self.render_canvas = self.img.copy()

    def draw_cross(self, x, y, color=(0, 0, 255), lengths = [5,5,5,5], rotation = 0):
        x_int = int(x)
        y_int = int(y)
        for i in range(4):
             cv2.line(self.render_canvas, (x_int, y_int), 
                    (x_int + int(lengths[i] * np.cos(rotation + i * np.pi / 2)), 
                     y_int + int(lengths[i] * np.sin(rotation + i * np.pi / 2))),
                    color, 1)



    def render_display(self, **kwargs):
        # self.reset_canvas()
        # self.region_contexts[0].render_display()
        self.show(self.render_canvas, **kwargs)

    def consolidate_pyramids(self):
        """Absorb pyramids that overlap"""
        pyramid_map = np.zeros_like(self.gray, dtype=np.int16) - 1
        for pyr in self.pyramids:
            overlap = np.unique(pyramid_map[pyr.get_mask() > -1])
            # remove 0 from overlap:
            overlap = overlap[overlap > -1]
            for overlapping in overlap:
                pyr.absorb(self.pyramids_by_id[overlapping])
                self.pyramids_by_id[overlapping] = None
            pyramid_map[pyr.get_mask() > -1] = pyr.id
        self.pyramids = [p for p in self.pyramids_by_id.values() if p is not None]
        self.pyramids_by_id = {p.id : p for p in self.pyramids}


    def reindex_pyramids(self):
        for i, pyr in enumerate(self.pyramids):
            pyr.id = i
            pyr.mask[pyr.mask > 0] = i
        self.pyramids_by_id = {p.id : p for p in self.pyramids}

    def segment(self, segment_density = 18):
        """Detect pyramids """
        import segmenting
        import pyramid
        points = segmenting.segment(self, segment_density)
        pyramids = [pyramid.Pyramid(self, p, i) for i,  p in enumerate(points)]
        pyramid_map = np.zeros_like(self.gray, dtype=np.int16) - 1
        
        for pyr in pyramids:
            overlap = np.unique(pyramid_map[pyr.get_mask() > -1])
            # remove 0 from overlap:
            overlap = overlap[overlap > -1]
            for overlapping in overlap:
                pyr.absorb(pyramids[overlapping])
                pyramids[overlapping] = None
            pyramid_map[pyr.get_mask() > -1] = pyr.id
        self.pyramids = [p for p in pyramids if p is not None]
        self.pyramids_by_id = {p.id : p for p in self.pyramids}



    def draw_pyramids(self, meanonly=True):
        self.reset_canvas()
        for pyr in self.pyramids:
            p = pyr.mean_position
            # self.draw_cross(p[1], p[0], lengths = [5,5,5,5], color = (0, 0, 250))
        # self.render_display(ms=5000)
        self.render_canvas = self.img.copy()
        for pyr in self.pyramids:
            pyr.get_mask(canvas=self.render_canvas, color = (0, 0, 255), meanonly=meanonly)
        self.show(self.render_canvas, ms=50     , frames = 20)
        # canvas = self.gray.copy().T
        # for pyr in self.pyramids:
        #     pyr.get_mask(canvas=canvas, color = (0, 0, 255), meanonly=meanonly)
        # self.show(canvas, ms=5000)


    def get_leg_lengths(self):
        """Calculate leg lengths"""
        import scipy.interpolate as interpolate   
        
        
        direction_vectors = [(np.cos(self.rotation + i * np.pi / 2), np.sin(self.rotation + i * np.pi / 2)) for i in range(4)]
        canvas = self.img.copy()
        ll = canvas.shape[0] + canvas.shape[1]
        linspace = np.linspace  (0, ll, ll)
        for pyr in self.pyramids:
            for i, direction_vector in enumerate(direction_vectors):
                start_point = pyr.mean_position
                # get leGth to edge of image in direction of direction_vector
                length_max, p = 0, start_point
                direction_vector = np.array(direction_vector)
                direction_vector = direction_vector / np.linalg.norm(direction_vector)
                while 0 < p[0] < self.img.shape[0] - 1 and 0 < p[1] < self.img.shape[1]- 1 :
                    p = start_point + length_max * direction_vector
                    length_max += 1
                
                length_max -= 2
                length_max = max(length_max, 0)
                check_len = -1
                # samples = [ uuu
                # sample_locations = []
                # for i in range(int(length)):
                #     sample_location = start_point + direction_vector * i / int(length)
                #     sample = interpolate.interpn((np.arange(self.img.shape[0]), np.arange(self.img.shape[1])), self.gray, sample_location)
                #     samples += [sample]
                #     sample_locations += [sample_location] 
                # import matplotlib.pyplot as plt
                
                # redone calculation with numpy 
                while check_len < length_max:
                    check_len = min(check_len + 100, length_max)
                    samples = linspace[:check_len]
                    sample_locations = start_point.reshape(1, -1) + direction_vector.reshape(1, -1) * samples.reshape(-1, 1)
                    samples = interpolate.interpn((np.arange(self.img.shape[0]), np.arange(self.img.shape[1])), self.gray, sample_locations)
                    
                    samples = 255 - samples

                    # plt.plot(samples)
                    # plt.show()

                    peaks = signal.find_peaks(samples, prominence = self.prominence)
                    print(peaks)
                    
                    # for i, peak in enumerate(peaks[0]):
                    #     print(peak)
                    #     self.show(arrow(canvas, (0,0), sample_locations[peak, :], (0, 0, 255), 1))
                    peaks = peaks[0]
                    peak = peaks[0] if len(peaks) > 0 else check_len
                    pyr.cross_lengths[i] = peak
                    if peak < check_len - 1:
                        break
            self.draw_pyramids()
 

    def to_json(self):
        self.reindex_pyramids()
        data = {}
        data['pyramids'] = [pyr.mean_position.to_json() for pyr in self.pyramids]
        data['rotation'] = self.rotation
        data['imgstr'] = self.imgstr


def IA_from_json(path):
    import pyramid
    with open(path, 'r') as f:
        data = json.load(f)
    IA = ImageAnalysis(data['imgstr'])
    IA.rotation = data['rotation']
    IA.pyramids = [pyramid.Pyramid.from_json(p, IA) for p in data['pyramids']]
    IA.pyramids_by_id = {p.id : p for p in IA.pyramids}
    return IA

        


def main():
    IA = ImageAnalysis('437-1-03.tif')
    IA = ImageAnalysis('242316_01.tif') # this is the sparse one that gives the issues
    IA.capture = []

    IA.segment(20)

    IA.consolidate_pyramids()
    IA.get_leg_lengths()
    import imageio
    imageio.mimsave('test.gif', IA.capture, duration=50/1000)
    
if __name__ == '__main__':
    import cProfile
    cProfile.run('main()')

