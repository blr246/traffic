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

Processes a sequence of nighttime traffic images and extracts the bright
objects.

Author: Brandon Reiss
Date: 20111105
Email: blr246@nyu.edu
"""
import sys
import pylab as plab
import scikits.image.color
import scipy
import scipy.ndimage as ndimg
import numpy as np
import PIL
import pymorph
import os
import fnmatch
import datetime

# Median filter size.
MED_SIZE = 5
# Averaging kernel size.
AVG_SIZE = 15
# Number of standard deviations local pixel from local mean.
NUM_STDDEV = 1.6
# Percentage of local bright object intensities to keep.
GLOBAL_BRIGHT_PCT = 0.2

# Default rectangles in Kenya images to blackout (white text).
BLACKOUT_RECTS = [(0, 0, 155, 10), (0, 470, 115, 480)]

def bright_object_detection(image):
    """ Perform bright object detection on an array image."""

    # Store all intermediate steps in a dictionary. Useful for debugging.
    steps = dict()
    steps['input'] = image

    # Reduce noise using a median filter.
    med_filter_size = (MED_SIZE, MED_SIZE, MED_SIZE)
    steps['median'] = ndimg.median_filter(steps['input'], med_filter_size)

    # Convert median filtered image to grayscale.
    steps['luminance'] = scikits.image.color.rgb2gray(steps['median']) * 255.
    
    # Compute local pixel average.
    k_avg = np.ones((AVG_SIZE, AVG_SIZE)) / AVG_SIZE**2
    steps['average'] = ndimg.convolve(steps['luminance'], k_avg)

    # Compute local pixel variance.
    steps['diff_mean'] = steps['luminance'] - steps['average']
    steps['diff_mean_sq'] = steps['diff_mean'] * steps['diff_mean']
    steps['variance'] = ndimg.convolve(steps['diff_mean_sq'], k_avg)
    
    # Compute binary threshold image using mahalonobis distance. Use the sign
    # of the difference between the pixel and its local mean to ignore dark
    # pixels.
    steps['maha_sq'] = (steps['diff_mean'] > 0) * steps['diff_mean_sq'] / \
                       steps['variance']
    steps['thresh_maha'] = (steps['maha_sq'] > (NUM_STDDEV * NUM_STDDEV))
    
    # Integrate global illumination effects by taking a top percentage of
    # intensities from the detected light regions.
    steps['masked_regions_lum'] = steps['thresh_maha'] * steps['luminance']
    steps['masked_regions_hist'] = pymorph.histogram(steps['masked_regions_lum'])
    steps['global_bright_thresh'] = int((len(steps['masked_regions_hist']) * \
                                         (1.0 - GLOBAL_BRIGHT_PCT)) + 0.5)
    steps['thresh_global'] = steps['masked_regions_lum'] >= \
                             steps['global_bright_thresh']

    # Morphological operations on detected blobs.
    steps['detect_erode'] = pymorph.erode(steps['thresh_global'])
    steps['detect_dilate'] = pymorph.dilate(steps['detect_erode'])
    
    # Count bright objects. Connected components and raw pixels.
    steps['detect_labels'] = pymorph.label(steps['detect_dilate'])
    steps['bright_blob_count'] = steps['detect_labels'].max()
    steps['bright_pixel_count'] = sum(steps['masked_regions_hist']
                                           [steps['global_bright_thresh']:])
    return steps


def exponential_moving_average(samples, new_sample_wt):
    """ Compute exponential moving average of sample set. """
    if len(samples) is 0:
        return None
    else:
        exp_avg_filtered = samples[:1]
        for s in samples[1:]:
            s_filtered = ((1.0 - new_sample_wt) * exp_avg_filtered[-1]) + \
                         (new_sample_wt * s)
            exp_avg_filtered.append(s_filtered)
        return exp_avg_filtered

def main(argv):
    now = datetime.datetime.now()
    # Need one command line argument.
    if len(argv) is not 2:
        print "Usage: {0} SERIES_FILE_PATH".format(argv[0])
        raise Exception("Invalid command-line parameters.")
    # Extract command-line parameters. This is the name of one file in the
    # series.
    path, filename = os.path.split(argv[1])
    file_name, file_ext = os.path.splitext(os.path.basename(filename))
    series_name_end = file_name.rindex('_')
    series_name = file_name[:series_name_end]
    print "Processing image series {0} in path {1}.".format(series_name, path)
    files_in_path = os.listdir(path)
    series_pattern = series_name + '_[0-9]*' + file_ext
    print "Processing files matching pattern {0}.".format(series_pattern)
    series_suffixes = [int(os.path.splitext(fn)[0].split('_')[-1]) \
                       for fn in files_in_path \
                       if fnmatch.fnmatch(fn, series_pattern)]
    series_suffixes.sort()
    print "Found {0} files in image series {1}.".format(len(series_suffixes),
                                                        series_name)
    label_img_path = "{}{:02d}{:02d}_{:02d}{:02d}{:02d}_{}_labeled".format(
                         now.year, now.month, now.day,
                         now.hour, now.minute, now.second, series_name)
    os.mkdir(label_img_path)
    # Process the files.
    bright_object_counts = []
    for suffix in series_suffixes:
        series_filename = series_name + '_' + str(suffix) + file_ext
        print "Processing file {0}.".format(series_filename)
        image = ndimg.imread(os.path.join(path, series_filename))
        # Black out specified regions.
        for rect in BLACKOUT_RECTS:
            image[rect[1]:rect[3], rect[0]:rect[2]] = 0
        steps = bright_object_detection(image)
        bright_object_counts.append((steps['bright_blob_count'],
                                     steps['bright_pixel_count']))
        # Save labeled file.
        label_img_filename = os.path.join(label_img_path,
                                          str(suffix) + file_ext)
        label_img = pymorph.overlay(steps['luminance'].astype('uint8'),
                                    steps['detect_dilate'])
        PIL.Image.fromarray(label_img).save(label_img_filename)
    # Compute exponential moving averages.
    blob_counts = [count[0] for count in bright_object_counts]
    blob_exp_avg = exponential_moving_average(blob_counts, 0.1)
    blob_exp_avg_norm = np.array(blob_exp_avg) / max(blob_exp_avg)
    pixel_counts = [count[1] for count in bright_object_counts]
    pixel_exp_avg = exponential_moving_average(pixel_counts, 0.1)
    pixel_exp_avg_norm = np.array(pixel_exp_avg) / max(pixel_exp_avg)
    # Make plots.
    p = plab.subplot('111')
    p.plot(blob_exp_avg_norm, 'b', label='blob_count')
    p.plot(pixel_exp_avg_norm, 'g', label='pixel_count')
    p.set_title("Normalized bright object count in {0} sequence."
                .format(series_name))
    p.set_ylabel('normalized count')
    p.set_xlabel('frame index')
    p.legend(loc=4)
    # Save plots
    plot_filename = "{}{:02d}{:02d}_{:02d}{:02d}{:02d}_{}_results.png".format(
                        now.year, now.month, now.day,
                        now.hour, now.minute, now.second, series_name)
    plab.savefig(plot_filename)
    # Write results.
    resuts_filename = "{}{:02d}{:02d}_{:02d}{:02d}{:02d}_{}_results".format(
                          now.year, now.month, now.day,
                          now.hour, now.minute, now.second, series_name)
    results_file = os.open(resuts_filename, os.O_CREAT | os.O_WRONLY)
    os.write(results_file,
             "# List of result tuples (blob_count, pixel_count).\n")
    os.write(results_file, str(bright_object_counts))
    os.close(results_file)
    return 0

if __name__ == "__main__":
    main(sys.argv)

