from typing import List, Tuple, Union
from pyramid import Pyramid
# import guicontrol
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from pyramid import Pyramid
import cv2
import pandas as pd
import os
def pixels_to_um(self: "guicontrol.GuiControl", pixels: np.ndarray):
    return pixels * self.um_per_pixel()


def H(L, sin_degrees: str = "54.7"):

    sinfactor = np.sin(np.radians(float(sin_degrees)))
    H1pre = max(L)
    H2pre = min(L)
    H3pre = (L[0] + L[1] + L[2] + L[3]) / 4
    H1 = H1pre * sinfactor
    H2 = H2pre * sinfactor
    H3 = H3pre * sinfactor
    work1 = f"Maximum height = max(L1, L2, L3, L4) * sin({sin_degrees}) = {H1pre} * sin({sin_degrees}) = {H1} um"
    work2 = f"Minimum height = min(L1, L2, L3, L4) * sin({sin_degrees}) = {H2pre} * sin({sin_degrees}) = {H2} um"
    work3 = f"Average height = (L1 + L2 + L3 + L4) / 4 * sin({sin_degrees}) = {H3pre} * sin({sin_degrees}) = {H3} um"
    return ((H1, work1), (H2, work2), (H3, work3))

def height_calc(self: "guicontrol.GuiControl", pyramids: "List[Pyramid]") -> Tuple[float, str]:
    """
    Calculate the height of the pyramids in the image.
    """
    heights = [[],[],[]]
    work = [[], [], []]
    for p in pyramids:
        L = [self.pixels_to_um(l) for l in p.cross_lengths]
        H1, H2, H3 = H(L)
        heights[0].append(H1[0])
        work[0].append(H1[1])
        heights[1].append(H2[0])
        work[1].append(H2[1])
        heights[2].append(H3[0])
    return heights, work

def make_hist(self, heights: List[List[float]], work, save: bool = True, show: bool = True):
    fig, axs = plt.subplots(len(heights), sharex=True)
    fig.suptitle("Pyramid Heights Histogram")

    titles = ["Maximum Height", "Minimum Height", "Average Height"]
    for i, ax in enumerate(axs):
        hist, bins = np.histogram(heights[i], bins=np.arange(0, 3.05, 0.05))
        ax.bar(bins[:-1], hist, width=0.05, align='edge')
        ax.set_ylabel('Counts [pyramids]')
        ax.set_title(titles[i])
        ax.text(0.05, 0.8, 'Mean = {:.2f} um'.format(np.mean(heights[i])), transform=ax.transAxes)
        ax.text(0.05, 0.7, 'Total Pyramids = {}'.format(len(heights[i])), transform=ax.transAxes)
    plt.xlabel("Pyramid height [um]")

    if save:
        import os
        imgname = self.IA.imgstr.split("/")[-1].split(".")[0]
        os.makedirs("./output/" + imgname, exist_ok=True)
        plt.tight_layout()
        plt.savefig("./output/" + imgname + "/" + "_hist.png")
        dataframe = pd.DataFrame({"Max Height (um)":heights[0], "Max Height Calculations":work[0], "Min Height (um)":heights[1], "Min Height Calculations":work[1], "Avg Height (um)":heights[2], "Avg Height Calculations":work[2]})
        dataframe.to_excel(f'./output/{imgname}/heights.xlsx', index = False)
        for i in range(len(titles)):
            height_measure = titles[i].replace(" ", "")
            w = open("./output/" + imgname + "/" + height_measure + "_hist_calculations.txt", "w")
            for j in range(len(work[i])):
                w.write("Pyramid " + str(j) + ": " + work[i][j] + "\n")
            h = open("./output/" + imgname + "/" + height_measure + "_hist_heights.txt", "w")
            for j in range(len(heights[i])):
                h.write(str(heights[i][j]) + "\n")

    if show:
        plt.show()

def make_hist_with_subplots(self: "guicontrol.GuiControl", heights: "List[float]", height_measure:str, work: "List[str]", subplot_params) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make a histogram of the heights of the pyramids.
    """
    
    
    heights = np.array(heights)
    binsize = 0.05
    topbin = max(3, np.ceil(max(heights) / binsize) * binsize)
    bins = np.arange(0, topbin, binsize)
    hist, bins = np.histogram(heights, bins=bins)
    plt.bar(bins[:-1], hist, width=binsize, align="edge")
    plt.xlabel("Pyramid height [um]")
    plt.ylabel("Counts [pyramids]")
    plt.title("Pyramid heights for " + height_measure + " measure")
    mean = np.mean(heights)
    median = np.median(heights)
    std = np.std(heights)
    plt.legend(["Mean: " + str(mean) + " um"])
    if True:
        imgname = self.IA.imgstr.split("/")[-1].split(".")[0]
        os.makedirs("./output/" + imgname, exist_ok=True)
        plt.savefig("./output/" + imgname + "/" + height_measure + "_hist.png")
        w = open("./output/" + imgname + "/" + height_measure + "_hist_calculations.txt", "w")
        path = "./output/" + imgname + "/" + height_measure + "_hist.png"
        for i in range(len(work)):
            w.write("Pyramid " + str(i) + ": " + work[i] + "\n")
        h = open("./output/" + imgname + "/" + height_measure + "_hist_heights.txt", "w")
        for i in range(len(heights)):
            h.write(str(heights[i]) + "\n")
    if True:
        plt.show()
    return path

def save_histimg(self, img_array, save: bool = True):
    fig = plt.figure()
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1.2])
    fig.add_subplot(gs[0])
    plt.imshow(orig_img, aspect='equal')
    fig.add_subplot(gs[1])
    plt.imshow(proc_img, aspect='equal')
    fig.add_subplot(gs[2])
    self.make_hist(save=False, show=False)
    if save:
        plt.savefig("./output/" + orig_img_path.split("/")[-1].split(".")[0] + "/overlay_hist.png")

def height_calc_hist_and_save(self, pyramids: List["Pyramid"]):
    heights = self.height_calc(pyramids)
    self.make_hist(heights)
    return heights

def save_histimg(self: "guicontrol.GuiControl", orig_img, pyramids: "List[Pyramid]"):
    heights, work = self.height_calc(pyramids)   
    # self.make_hist_with(heights, work, save=False, show=False)
    titles = ["Maximum Height", "Minimum Height", "Average Height"]
    paths = []
    for i in range(len(titles)):
        paths.append(self.make_hist_with_subplots(heights[i], titles[i], work[i], subplot_params=False))
    images = list(map(Image.open, paths))
    widths, heights = zip(*(i.size for i in images))

    total_width = max(sum(widths), orig_img.shape[1])
    max_height = max(*heights) + orig_img.shape[0] 
    new_im = Image.new("RGB", (total_width, max_height))
    new_im.paste(Image.fromarray(orig_img), (0, 0))
    y_offset = orig_img.shape[0]
    x_offset = 0 
    for im in images:
        new_im.paste(im, (x_offset, y_offset))
        x_offset += im.size[0]
    new_im.save("./output/" + self.IA.imgstr.split("/")[-1].split(".")[0] + "/overlay_hist.png")
    
    # fig.canvas.draw()
    # img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # plt.close(fig)

    # scale_factor = orig_img.shape[0] / img.shape[0]
    # img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation = cv2.INTER_CUBIC)

    # final_img = cv2.hconcat([orig_img, img])
    # cv2.imwrite('./output/combined.png', final_img)
