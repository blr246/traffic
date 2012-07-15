"""
Copyright (C) 2011-2012 Brandon L. Reiss

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Displays either day or nighttime traffic image processing in a mock-up UI
based on the HTC Desire smartphone.
"""
import numpy as np
import scipy
import scipy.ndimage as ndimg
from collections import deque
from copy import *
import PIL
import ImageOps
import pylab
import cv2
import os
import fnmatch
import sys
import pymorph
import night
import day
import argparse

class PhoneDemo(object):
    """ Object to run the phone UI demo. """

    TYPE_DAY = "DAY"
    TYPE_NIGHT = "NIGHT"
    HISTORY_FRAMES = 600

    class DayProcessor(object):
        """ Object used to process day sequences. """

        GMM_K = 3
        GMM_NUM_FRAMES = 25
        GMM_W_INIT = 0.1
        GMM_VAR_INIT = 20
        GMM_MAHA_THRESH = 3
        MASK_OVERLAY_ALPHA = 0.4

        def __init__(self, rgb):

            assert(rgb.dtype == 'uint8')

            self._gmm = day.GaussianMixtureModelUV(self.GMM_K, rgb.shape,
                                                   self.GMM_NUM_FRAMES,
                                                   self.GMM_W_INIT,
                                                   self.GMM_VAR_INIT,
                                                   self.GMM_MAHA_THRESH)
            self._ycbcr = np.zeros(rgb.shape, dtype='uint8')
            self._mask = np.zeros(rgb.shape[:2], dtype='uint8')
            self._red_mask = np.zeros(rgb.shape, dtype='uint8')
            self._rgb_red_masked = np.zeros(rgb.shape, dtype='uint8')
            self._process_count = 0

        def next(self, rgb):
            """ Process the next file and return the results. """

            # Do GMM steps.
            self._gmm.rgb2ycbcr(rgb, self._ycbcr)
            self._gmm.segment_cl(self._mask)
            self._gmm.update_cl(self._ycbcr)
            # Save total pixels in foreground.
            fg_pixel_count = np.sum(self._mask)
            # Pull alpha and render red overlay
            # (channels are reversed RGB = BGR).
            self._red_mask[:,:,2] = self._mask * 255
            self._rgb_red_masked[:,:] = \
                (self.MASK_OVERLAY_ALPHA * self._red_mask) + \
                ((1. - self.MASK_OVERLAY_ALPHA) * rgb)

            # Ignore the first GMM_NUM_FRAMES / 2 frames.
            self._process_count = self._process_count + 1
            if self._process_count > self.GMM_NUM_FRAMES / 2:
                return fg_pixel_count, self._rgb_red_masked
            else:
                return 0, self._rgb_red_masked


    class NightProcessor(object):
        """ Object used to process day sequences. """

        def __init__(self, rgb):

            pass

        def next(self, rgb):
            """ Process the next file and return the results. """

            def blackout_date_regions(image, blackout_rects):
                """ Black out specified regions. """

                for rect in blackout_rects:
                    image[rect[1]:rect[3], rect[0]:rect[2]] = 0

            # Do bright object detection.
            blackout_date_regions(rgb, night.BLACKOUT_RECTS)
            steps = night.bright_object_detection(rgb)
            # Return results (channels are reversed RGB = BGR).
            label_img = pymorph.overlay(steps['luminance'].astype('uint8'),
                                        blue=steps['detect_dilate'])
            return steps['bright_blob_count'], label_img


    def __init__(self):

      # Initialize plotting parameters.
      self._history_raw = deque()
      self._history_filtered = deque()
      self._max_sample = 0.001
      self._ui = PhoneDisplay()
      self._filter_exp = 0.1
      self._sample_exp_filter = 0.

    def run_sequence(self, type, filepath, seq_range=None, filter_exp=None):
        """ Run a TYPE_DAY or TYPE_NIGHT sequence. """

        QUIT_KEY_CODES = [ 27, 113, 81 ]
        PAUSE_KEY_CODES = [ 32, 112, 80 ]

        def pause():
            """ Poll input until the pause key is pressed. """

            while True:
                key = cv2.waitKey(100)
                if PAUSE_KEY_CODES.count(key) > 0:
                    break

        def bound_queue_push(val, q, maxlen=None):
            """ Push to bounded queue. """

            q.append(val)
            if maxlen is not None and len(q) > maxlen:
                q.popleft()

        assert(type == self.TYPE_DAY or type == self.TYPE_NIGHT)

        # TODO(reissb) -- The history frames and filtering need to become
        #   parameterized in some way. The history frames is fixed by the
        #   camera framerate. The filtering is fixed by the required
        #   detection sensitivity.
        if filter_exp is not None:
            self._filter_exp = filter_exp
        else:
            self._filter_exp = 0.1

        # Clear state.
        self._ui.clear()
        self._history_raw = deque()
        self._history_filtered = deque()
        self._max_sample = 0.001
        self._sample_exp_filter = 0.

        # Extract command-line parameters. This is the name of one file in the
        # series.
        path, filename = os.path.split(filepath)
        file_name, file_ext = os.path.splitext(os.path.basename(filename))
        series_name_end = file_name.rindex('_')
        series_name = file_name[:series_name_end]
        print "Processing image series {0} in path {1}.".format(series_name,
                                                                path)
        files_in_path = os.listdir(path)
        series_pattern = series_name + '_[0-9]*' + file_ext
        print "Processing files matching pattern {0}.".format(series_pattern)
        series_suffixes = [int(os.path.splitext(fn)[0].split('_')[-1]) \
                           for fn in files_in_path \
                           if fnmatch.fnmatch(fn, series_pattern)]
        series_suffixes.sort()
        num_files = len(series_suffixes)
        print "Found {0} files in image series {1}.".format(num_files,
                                                            series_name)
        # Check for limited range.
        if seq_range is not None:
            assert(seq_range[1] > seq_range[0] and seq_range[0] >= 0)
            print "Filtering series range [{},{}).".format(seq_range[0],
                                                           seq_range[1])
            series_suffixes = np.array(series_suffixes)
            f = (series_suffixes >= seq_range[0]) * \
                (series_suffixes < seq_range[1])
            series_suffixes = np.sort(series_suffixes * f)
            remove_count = len(series_suffixes) - np.sum(f)
            series_suffixes = np.delete(series_suffixes, range(remove_count))

        # Load first file and process.
        series_filename = series_name + '_' + str(series_suffixes[0]) + \
                          file_ext
        rgb = ndimg.imread(os.path.join(path, series_filename))
        # Initilaize the processor.
        type_processor = self.DayProcessor(rgb) if type is self.TYPE_DAY \
                         else self.NightProcessor(rgb)
        # Process the files.
        quit_flag = False
        process_count = 0
        history_n = int(self.HISTORY_FRAMES / \
                        (self._ui.history_frame_count - 1))
        for suffix in series_suffixes:
            # Process the next file.
            series_filename = series_name + '_' + str(suffix) + file_ext
            print "Processing file {0}.".format(series_filename)
            rgb = ndimg.imread(os.path.join(path, series_filename))
            sample_raw, display_img = type_processor.next(rgb)
            self._sample_exp_filter = \
                ((1. - self._filter_exp) * self._sample_exp_filter) + \
                (self._filter_exp * sample_raw)
            # Put new samples on queues.
            bound_queue_push(sample_raw,
                             self._history_raw, self.HISTORY_FRAMES)
            bound_queue_push(self._sample_exp_filter,
                             self._history_filtered, self.HISTORY_FRAMES)
            # Update UI.
            self._max_sample = max(self._max_sample,
                                   self._sample_exp_filter * 1.1)
            ybound = (0, self._max_sample)
            plot_img = self.plot_history(self._history_raw,
                                         self._history_filtered,
                                         ybound)
            self._ui.set_main_video_frame(display_img)
            self._ui.set_plot(plot_img)
            # Space history frames evenly over interval.
            if 0 == (process_count % history_n):
                self._ui.push_history_frame(display_img)
            process_count = process_count + 1
            # Show UI.
            cv2.imshow("Phone Display", self._ui.ui_image)
            key = cv2.waitKey(1)
            # Catch escape or 'q' or 'Q':
            if QUIT_KEY_CODES.count(key) > 0:
                quit_flag = True
                break
            # Catch spacebar, 'p' or 'P':
            if PAUSE_KEY_CODES.count(key) > 0:
                pause()

        # Cleanup GUI on complete.
        if not quit_flag:
            cv2.waitKey(-1)
        cv2.destroyAllWindows()

    @staticmethod
    def plot_history(raw, filtered, ybound):
        """ Make plot of raw and history and return as image. """

        p = pylab.subplot('111')
        p.clear()
        p.figure.set_size_inches(4, 3);
        p.plot(raw, '.r')
        p.plot(filtered, '-b')
        p.axes.set_ybound(ybound)
        p.figure.canvas.draw()
        buf = np.fromstring(p.figure.canvas.tostring_rgb(), dtype='uint8')
        h, w = p.figure.canvas.get_width_height()
        buf.shape = (w, h, 3)
        return buf


class PhoneDisplay(object):

    # Dictionary of UI resources.
    RESOURCES = {
        # Name of the phone UI image.
        "UI_BASE": "ui_base.jpg"
        }
    # Coordinates for the phone image display area.
    UI_LAYOUT = {
        "MARGIN": 15,
        "DISPLAY_RECT": { "UpperLeft": (255, 59), "LowerRight": (1106, 596) },
        "MAIN_VIDEO_SIZE": (360, 480),
        "PLOT_SIZE": (240, 320),
        "HISTORY_FRAME_COUNT": 5,
        "CLEAR_COLOR": np.array([60, 85, 45])
        }

    def __init__(self):
        """ Setup phone UI. """

        # Load UI base resource and set slice of display area.
        self._ui_base = ndimg.imread(self.RESOURCES["UI_BASE"])
        up_lt = self.UI_LAYOUT["DISPLAY_RECT"]["UpperLeft"]
        lw_rt = self.UI_LAYOUT["DISPLAY_RECT"]["LowerRight"]
        self._ui_display_area = self._ui_base[up_lt[1]:lw_rt[1] + 1,
                                              up_lt[0]:lw_rt[0] + 1]
        self._ui_display_area[:,:] = self.UI_LAYOUT["CLEAR_COLOR"]
        self._ui_display_area_size = self._ui_display_area.shape[:2]
        self._ui_main_video_size = self.UI_LAYOUT["MAIN_VIDEO_SIZE"]
        self._plot_size = self.UI_LAYOUT["PLOT_SIZE"]

        margin = self.UI_LAYOUT["MARGIN"]
        # Get main video frame area.
        vid_frm_x1 = self._ui_display_area_size[1] - margin
        vid_frm_x0 = vid_frm_x1 - self._ui_main_video_size[1]
        vid_frm_y0 = margin
        vid_frm_y1 = vid_frm_y0 + self._ui_main_video_size[0]
        self._ui_main_video_frame = self._ui_display_area[vid_frm_y0:
                                                          vid_frm_y1,
                                                          vid_frm_x0:
                                                          vid_frm_x1]
        # Get plot area.
        plt_frm_x0 = margin
        plt_frm_x1 = plt_frm_x0 + self._plot_size[1]
        plt_frm_y0 = margin
        plt_frm_y1 = plt_frm_y0 + self._plot_size[0]
        self._ui_plot_frame = self._ui_display_area[plt_frm_y0: plt_frm_y1,
                                                    plt_frm_x0: plt_frm_x1]
        # Compute history frame areas.
        his_frm_count = self.UI_LAYOUT["HISTORY_FRAME_COUNT"]
        his_frm_wid = int((self._ui_display_area_size[1] -
                          ((his_frm_count + 1)  * margin)) / his_frm_count)
        his_frm_ht = int((3./4.) * his_frm_wid)
        self._ui_history_frame_size = (his_frm_ht, his_frm_wid)
        his_frm_y0 = (2 * margin) + self._ui_main_video_size[0]
        his_frm_x0_fn = lambda n: margin + ((margin + his_frm_wid) * n)
        his_frm_x1_fn = lambda n: (margin + his_frm_wid) * (n + 1)
        self._ui_history_frames = map(
            lambda n: self._ui_display_area[
                his_frm_y0: his_frm_y0 + his_frm_ht,
                his_frm_x0_fn(n):his_frm_x1_fn(n)],
            range(self.UI_LAYOUT["HISTORY_FRAME_COUNT"]))

    def clear_display_area(self, color=np.array([0, 0, 0])):
        """ Clear UI base display area to given color. """

        self._ui_display_area[:,:] = color

    def set_main_video_frame(self, frame):
        """ Set the main video frame in the UI layout. """

        h, w = self._ui_main_video_size
        img = np.array(ImageOps.fit(PIL.Image.fromarray(frame), (w, h)))
        self._ui_main_video_frame[:,:] = img

    def push_history_frame(self, frame):
        """ Push a frame to the top of the history images. """

        # Shift back.
        for n in range(len(self._ui_history_frames) - 1):
            self._ui_history_frames[n][:,:] = self._ui_history_frames[n+1][:,:]
        # Update.
        h, w = self._ui_history_frame_size
        img = np.array(ImageOps.fit(PIL.Image.fromarray(frame), (w, h)))
        self._ui_history_frames[-1][:,:] = img

    def set_plot(self, plot):
        """ Set the plot image in the UI layout. """

        h, w = self._plot_size
        img = np.array(ImageOps.fit(PIL.Image.fromarray(plot), (w, h)))
        self._ui_plot_frame[:,:] = img

    def clear(self):
        """ Reset the UI. """

        ui_base = ndimg.imread(self.RESOURCES["UI_BASE"])
        self._ui_base[:,:,:] = ui_base
        self._ui_display_area[:,:] = self.UI_LAYOUT["CLEAR_COLOR"]

    def get_ui_image(self):
        return self._ui_base
    def get_history_frame_count(self):
        return len(self._ui_history_frames)

    ui_image = property(get_ui_image, doc="The main display image.")
    history_frame_count = property(get_history_frame_count,
                                   doc="Count of history frames.")


def main():
    #  Parse arguments for
    #    PhoneDemo.run_sequence(type, filepath, seq_range, filter_exp):
    parser = argparse.ArgumentParser(description='Run a UI demo of ' +
                                                 'image-based traffic ' +
                                                 'analysis algorithms.')
    parser.add_argument('SEQUENCE_TYPE', type=str, choices=('D', 'N'),
                        help='day or night image type')
    parser.add_argument('SEQUENCE_IMAGE_PATH', type=str,
                        help='path to an image within sequence')
    parser.add_argument('-r', '--range', default=None, nargs=2, type=int,
                        help='range of frames to process as in \'-r 0 100\'')
    parser.add_argument('-e', '--filter_exp', default=None, type=float,
                        help='exponential filter strength')
    args = parser.parse_args(sys.argv[1:])
    demo = PhoneDemo()
    demo.run_sequence(PhoneDemo.TYPE_DAY if 'D' == args.SEQUENCE_TYPE
                                         else PhoneDemo.TYPE_NIGHT,
                      args.SEQUENCE_IMAGE_PATH, args.range, args.filter_exp)

if __name__ == "__main__":
    main()

