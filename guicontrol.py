import analysis
import cv2
from typing import Optional, Dict, Any, List, Tuple, Callable
import numpy as np
from t1cv import *
import pyramid
import tkinter as tk
from tkinter.filedialog import askopenfilename

# Key mappings
K_ESC, K_ENTER, K_SPACE, K_BACKSPACE = 27, 13, 32, 8
K_UP, K_DOWN, K_LEFT, K_RIGHT, K_TAB = 82, 84, 81, 83, 9

class GuiControl():
    def __init__(self, IA: "Optional[analysis.ImageAnalysis]" = None):
        self.IA: "Optional[analysis.ImageAnalysis]" = IA
        if self.IA is None:
            self.open_image()
        self.mode_function = None
        self.selected_pyramid = None
        self.click_point = None
        self.direction_vector = [0,1]
        self.direction_vector_arrow = [[0,0],[0,0]]

        self.micrometer_scale_length = 1
        self.micrometer_scale_line = np.array(((23,734), (95,734)))
        self.quit, self.savedstr, self.old_str, self.keys, self.mousedrags, self.selected_pyramids = False, "", "", [], [], []
        self.guicanvas = None
        self.prev_guicanvas = self.IA.img.copy()
        self.leg_index = 0
        self.length_in_um = ""
        self.titlebar_original = self.IA.titlebar.copy()
        self.button_bounding_boxes = []
        self.legend = {}
        self.prev_menus = []
        self.saved_guibar = self.IA.guibar.copy()
        self.leg_length_step = 2
    def mouse_event_callback(self, action, x, y, flags, *userdata):
        # print(x, y)
        # for idx, bbox in enumerate(self.button_bounding_boxes):
        #     if bbox[0] <= y <= bbox[2] and bbox[1] <= x <= bbox[3]:
        #         box = self.button_bounding_boxes[idx]
        #         box = (box[0] - self.IA.fullimg.shape[0] - 4, box[1] + 10, box[2] - self.IA.fullimg.shape[0] + 6, box[3] + 10)
        #         box = (box[1], box[0], box[3], box[2])
        #         guibar = self.saved_guibar.copy()
        #         cv2.rectangle(guibar, box[0:2], box[2:4], (23, 100, 182), 3)
        #         self.IA.guibar = guibar
        #         self.IA.show()
                
        if action == cv2.EVENT_LBUTTONDOWN:
            # for idx, bbox in enumerate(self.button_bounding_boxes):
            #     if bbox[0] <= y <= bbox[2] and bbox[1] <= x <= bbox[3]:
            #         nextmode = self.legend[list(self.legend.keys())[idx]]
            #         if type(nextmode) is not str:
            #             ls = nextmode()
            #             if ls is not None:
            #                 legend, state = ls
            #                 self.prev_menus.append((self.mode_function, self.legend, self.button_bounding_boxes))
            #                 self.mode_function = nextmode
            #             else:
            #                 legend, state = self.mode_function()
            #         else:
            #             legend, state = self.mode_function()


            # self.set_guibar_text("mode", "Left Button Down")
            self.click_point =  (y, x)
            # self.IA.show(self.IA.render_canvas)
        elif action == cv2.EVENT_LBUTTONUP:
            if self.click_point is not None:
                dragtup = np.array((self.click_point, (y, x)), dtype = np.int32)
                self.click_point = None
                self.mode_function(mousedrag = dragtup)
                self.update_display(*self.mode_function())
            
    def set_menu_bounding_boxes(self, text_list):
        self.button_bounding_boxes = []
        if len(text_list) == 0:
            self.button_bounding_boxes = [[0, 0, 0, 0]]
            return
        prev_button_bounding_boxes = self.prev_menus[-1][2] if self.prev_menus else []
        offset = 0
        for i, (prev_menu_function, prev_legend, prev_button_bounding_boxes) in enumerate(self.prev_menus):
            farthest_right = max([bbox[3] for bbox in prev_button_bounding_boxes]) if len(prev_button_bounding_boxes) > 0 else 0
            if self.IA.fullimg.shape[1] - 10 < farthest_right + 10 + offset + max([cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0][0] for text in text_list]):
                offset = 25
            else:
                offset = 10 + farthest_right
        for i, text in enumerate(text_list):
            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            x, y = 10 + offset, self.IA.fullimg.shape[0] + 30 + 25 * i
            self.button_bounding_boxes.append((y - h, x, y, x + w))

    def open_image(self):
        # Use tkinter file dialog to select and open an image file.
        root = tk.Tk()
        root.withdraw()  # Hide unnecessary window
        file_path = askopenfilename()
        if file_path:
            self.IA = analysis.ImageAnalysis(file_path)
            root.destroy()
        else:
            print("No file selected. Exiting...")
            exit()

    def control_loop(self):
        cv2.setMouseCallback("Image Analysis", self.mouse_event_callback)
        if self.IA is None:
            self.open_image()
        k = -1
        nextmode = None
        while k != ord("y"):
            if self.mode_function == None:
                self.mode_function = self.default_mode
            if nextmode is not None:
                ls = nextmode()
                if ls is not None:
                    legend, state = ls
                    # self.prev_menus.append((self.mode_function, self.legend, self.button_bounding_boxes))
                    self.mode_function = nextmode
                else:
                    legend, state = self.mode_function()
            else:
                legend, state = self.mode_function()
            self.update_display(legend, state)
            k = cv2.waitKey(0)
            if k == K_ESC:
                nextmode = None
                # if len[self.prev_menus]:
                #     self.mode_function, self.legend, self.button_bounding_boxes = self.prev_menus.pop()
                #     continue
                self.prev_menus = []
                self.mode_function = self.default_mode
            else:
                nextmode = self.select_mode(legend, k)
                if nextmode is None:
                    self.mode_function(key = k)

        # cv2.setMouseCallback("Image Analysis",    None)

    def update_display(self, legend, state):
        self.legend = legend
        text_list = self.set_guibar_text(legend, state)
        # self.set_menu_bounding_boxes(text_list)
        self.show_pyramids()
        self.saved_guibar = self.IA.guibar.copy()


    def parse_state_key(self, key: str):
        if "(" in key:
            lsplit = key.split("(")
            if len(lsplit) == 1:
                return lsplit[0], None, None
            l,r = lsplit[0:1], lsplit[1:]
            while ")" not in r:
                l += r[0]
                r = r[1:]
            rsplit = r.split(")")
            inside, post = rsplit[0], rsplit[1:].join(")")


    def state_list_to_dict(self, state: "List[Any[str, Tuple[str]]]"):
        d = {}
        for s in state:
            if type(s) is str:
                label, key = s, s
            else:
                label, key = s
            if key in self.__dict__:
                d[label] = self.__dict__[key]
            elif key in GuiControl.__dict__:
                d[label] = GuiControl.__dict__[key](self)
            else:
                raise KeyError(f"Key {key} not found in GuiControl")
            
        return d

    def show_pyramids(self):
        self.IA.draw_pyramids()
        if self.guicanvas is None:
            self.guicanvas = self.IA.last_show.copy()
        for sp in self.selected_pyramids:
            sp.get_mask(self.guicanvas, color = (200, 200, 0), meanonly = True)
        if self.leg_index is not None:
            l = [0, 0, 0, 0]
            l[self.leg_index] = 1
            l = np.array(l)
            for sp in self.selected_pyramids:
                lengths = np.array(sp.cross_lengths) * l
                sp.get_mask(self.guicanvas, color = (200, 0, 200), meanonly = True, lengths = lengths)
        prevrender = self.IA.last_show.copy()
        self.IA.show(self.guicanvas)
        self.prev_guicanvas = self.guicanvas
        self.guicanvas = None

        self.IA.last_show = prevrender

    def gui_text(self, legend, status, fv = None):
        l = []
        for k in legend:
            if type(legend[k]) is str:
                l += [k + ": " + legend[k]]
            else:
                s = legend[k].__doc__.split("\n")[0]
                l += [k.upper() + ": " + s]
            if fv == legend[k]:
                l[-1] = "(" + l[-1] + ")"
        s = []
        status = self.state_list_to_dict(status)
        for k in status:
            s += [k + "=" + str(status[k])]
        return l, s

    def puttext_in_guibar(self, text_list, guibar, bounding_boxes, draw_bounding_boxes = False):
        if len(text_list) == 0:
            return
        offset = bounding_boxes[0][1]
        for i in range(len(text_list)):
            cv2.putText(guibar, text_list[i], (10 + offset, 30 + 25 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1) 
            if draw_bounding_boxes > i and len(bounding_boxes) > i:
                box = bounding_boxes[i]
                box = (box[0] - self.IA.fullimg.shape[0] - 4, box[1] + 10, box[2] - self.IA.fullimg.shape[0] + 6, box[3] + 10)
                box = (box[1], box[0], box[3], box[2])
                cv2.rectangle(guibar, box[0:2], box[2:4], (0, 0, 0), 1)


    def set_guibar_text(self, 
                        legend:"Dict[str, Any[Callable, str]]" = {},
                        status:"List[Any[str, Tuple[str]]]" = []):
        guibar = self.IA.guibar_original.copy()
        l = []
        # todo add prev legends and use bounding boxes to determine where to print
        # make separate function for legend and bounding boxe to call putText

        l, s = self.gui_text(legend, status)
        pl = l + s
        self.set_menu_bounding_boxes(l)  # Calculate bounding boxes for text
        for t in self.prev_menus:
            f, l, b = t
            l, s = self.gui_text(l, [], fv = self.mode_function)
            self.puttext_in_guibar(l, guibar, b) 
        # todo Draw bounding boxes also for just the current legend
        self.puttext_in_guibar(pl, guibar, self.button_bounding_boxes)        
        self.IA.guibar = guibar
        return l

    def select_mode(self, legend, key):
        if chr(key) in legend:
            k = chr(key)
        elif key in legend:
            k = key
        else:
            return None
        selected = legend[chr(key)]
        if isinstance(selected, Callable):
            return selected
        return None
        




    def default_mode(self, key = None, mousedrag = None):
        legend = {
            "s": self.set_up,
            "a": self.add_pyramid,
            "p": self.pyramid_edit,
            "c": self.computations_menu,
                            }
        state = []
        return legend, state

    def set_up(self, key=None, mousedrag=None):
        """Set up the image analysis"""
        legend = {
            "m": self.set_micrometer_scale,
            "p": self.IA.segment,
            "d": self.set_direction,

        }
        return legend, []


    def pyramid_count(self):
        return len(self.IA.pyramids)

    def selected_pyramid_count(self):
        return len(self.selected_pyramids)        

    def pyramid_edit(self, key = None, mousedrag = None):
        """Edit pyramids"""
        if key is not None:
            pass
        elif mousedrag is not None:
            selected = []
            for p in self.IA.pyramids:
                if p.is_inside_box(mousedrag  ):
                    selected += [p]
            if len(selected) == 0:
                self.selected_pyramids = []
            else:
                self.selected_pyramids = selected
            
        if len(self.selected_pyramids) == 0:
            legend = {"click and drag a box": "select pyramids"}
        else:
            legend = {
                      "d": self.delete_pyramids,
                      "l" : self.edit_pyramid_legs,
                      "click and drag a box": "select pyramids"}
        return legend, ["pyramid_count", "selected_pyramid_count"]
    
    def add_pyramid(self, key = None, mousedrag = None):
        """Add a pyramid"""
        if key is not None:
            pass
        elif mousedrag is not None:
            pos = mousedrag[0]
            length = max(np.linalg.norm(mousedrag[1] - mousedrag[0]), 4)
            self.IA.pyramids += [pyramid.Pyramid(self.IA, pos, id = len(self.IA.pyramids))]
            self.IA.pyramids[-1].set_cross_lengths([length, length, length, length])
            self.selected_pyramids = [self.IA.pyramids[-1]]
            # self.IA.pyramids[-1].get_mask(self.guicanvas, color = (200, 200, 0), meanonly = True)
        else:
            return {'click and drag': 'add pyramid with leg lengths as long as the drag at the click location'}, []

    def delete_pyramids(self, key = None, mousedrag = None):
        """Delete selected pyramids"""
        self.IA.pyramids = [p for p in self.IA.pyramids if p not in self.selected_pyramids]
        self.IA.pyramids_by_id = {p.id: p for p in self.IA.pyramids}
        self.selected_pyramids = []
        # self.pyramids_b

    

    def set_micrometer_scale(self, key = None, mousedrag = None):
        """Set the micrometer scale length"""
        if key is not None:
            pass
        elif mousedrag is not None:
            self.micrometer_scale_line = mousedrag
            self.micrometer_scale_length = np.linalg.norm(self.micrometer_scale_line[1] - self.micrometer_scale_line[0])
        else:
            arrow = self.micrometer_scale_line - np.array((self.IA.gray.shape[0], 0))
            titlebar = self.titlebar_original.copy()
            self.IA.titlebar = titlebar
            arrow = line(titlebar, arrow[0], arrow[1], (0, 200, 15), 2)
            self.IA.show()
            return {'Click and drag' : 'sets the line for the scale', 'n' : self.set_micrometer_scale_length}, ['micrometer_scale_length', 'um_per_pixel']
    

    def set_micrometer_scale_length(self, key = None, mousedrag = None):
        """set the numerical value of the micrometer scale length
        Type the number on the slide legend and press enter"""
        if key is not None:
            if key == K_BACKSPACE:
                self.length_in_um = self.length_in_um[:-1]
            elif key == K_ENTER:
                self.micrometer_scale_length = float(self.length_in_um)
                self.length_in_um = ""
                self.mode_function = self.set_micrometer_scale
            else:
                self.length_in_um += chr(key)
        elif mousedrag is not None:
            pass
        else:
            return {"Enter" : "submit new length", "Backspace" : "delete last character"}, ["length_in_um"]


    def um_per_pixel(self):
        pix = np.linalg.norm(self.micrometer_scale_line[1] - self.micrometer_scale_line[0])
        um = self.micrometer_scale_length
        return um / pix

    def set_direction(self, key  = None, mousedrag = None):
        """Set the direction of the pyramid legs"""
        if mousedrag is not None:
            # convert mousedrag ((fromx, fromy),(tox, toy)) to angle in radians
            self.direction_vector = np.array(mousedrag[1]) - np.array(mousedrag[0])
            self.IA.rotation = np.arctan2(self.direction_vector[1], self.direction_vector[0])
            self.direction_vector_arrow = mousedrag
        arrow(self.guicanvas, self.direction_vector_arrow[0], self.direction_vector_arrow[1], (0, 0, 255), 2)
        self.IA.show(self.guicanvas)
        return {}, ["direction_vector"]
    


    def edit_pyramid_legs(self, key = None, mousedrag = None):
        """Edit the pyramid legs"""
        
        if key is not None:
            if key == ord("=") or key == ord("-"):
                if key == ord("="):
                    d = 1 # increase
                else:
                    d = -1 # decrease
                for p in self.selected_pyramids:
                    p.cross_lengths[self.leg_index] =  max(p.cross_lengths[self.leg_index] + d, 1)
            self.IA.draw_pyramids(meanonly=True)
        return {"q" : self.change_selected_leg, "+" : self.increase_leg_length, "-": self.decrease_leg_length, "1" : self.setstep1, "5" : self.setstep5}, []
    
    def setstep1(self):
        """Set the step size for the leg length to 1"""
        self.leg_length_step = 1

    def setstep5(self):
        """Set the step size for the leg length to 5"""
        self.leg_length_step = 5
     

    def increase_leg_length(self):
        """Increase the length of the selected leg"""
        for p in self.selected_pyramids:
            p.cross_lengths[self.leg_index] += self.leg_length_step
        self.IA.draw_pyramids(meanonly=True)

    def decrease_leg_length(self):
        """Decrease the length of the selected leg"""
        for p in self.selected_pyramids:
            p.cross_lengths[self.leg_index] = max(p.cross_lengths[self.leg_index] - self.leg_length_step, 1)
        self.IA.draw_pyramids(meanonly=True)
    

    def change_selected_leg(self):
        """Select the next leg"""
        self.leg_index = (self.leg_index + 1) % 4 if self.leg_index is not None else 0
        
    def computations_menu(self, key = None, mousedrag = None):
        """Perform computations"""
        legend = {
            "s": self.IA.segment, 
            "l" : self.IA.get_leg_lengths, 
            "a": self.IA.consolidate_pyramids,
            "h" : self.make_histogram}
        return legend, []

    def calculate_leg_lengths(self, key = None, mousedrag = None):
        """Calculate the leg lengths"""
        if len(self.selected_pyramids) == 0:
            self.IA.get_leg_lengths(self.selected_pyramids)
        else:
            self.IA.get_leg_lengths()                   


    from calculate import pixels_to_um, make_hist, height_calc_hist_and_save, height_calc, save_histimg, make_hist_with_subplots


    def make_histogram(self, key = None, mousedrag = None):
        """Make a histogram of the pyramid heights"""

         
        self.save_histimg(np.concatenate((self.IA.last_show, self.IA.titlebar), axis = 0), self.IA.pyramids)
        # self.height_calc_hist_and_save(self.IA.pyramids)
        self.mode_function = self.default_mode





def main():
    IA = analysis.ImageAnalysis('437-1-03.tif')
    IA = analysis.ImageAnalysis("242316_01.tif")
    gui = GuiControl()
    IA = gui.IA
    IA.capture = []
    gui.control_loop()
    import imageio
    imageio.mimsave('test.gif', IA.capture, duration=50/1000)


if __name__ == "__main__":
    main()
