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
"""
import sys
import pylab as plab
import scikits.image.color
import scipy
import scipy.ndimage as ndimg
import numpy as np
import pymorph



MED_SIZE = 5
AVG_SIZE = 15
NUM_STDDEV = 1.6
GLOBAL_BRIGHT_PCT = 0.2

steps = dict()

blackout_rects = [(0, 0, 155, 10), (0, 470, 115, 479)]

# <demo> auto

###############################################################################
# This section loads the image and displays it.

# Extract image filename to load.
filename = sys.argv[1]
print "Image file name:", filename
steps['input'] = ndimg.imread(filename)
# Black out specified regions.
for rect in blackout_rects:
    steps['input'][rect[1]:rect[3], rect[0]:rect[2]] = 0

###############################################################################
# This section performs a median filter for noise reduction.

med_filter_size = (MED_SIZE, MED_SIZE, MED_SIZE)
steps['median'] = ndimg.median_filter(steps['input'], med_filter_size)
print "Median filter with kernel size {0}:".format(MED_SIZE)
plab.imshow(steps['median'])

###############################################################################
# This section converts the image to grayscale.

steps['luminance'] = scikits.image.color.rgb2gray(steps['median']) * 255.
print "Image converted to grey:"
plab.imshow(steps['luminance'], cmap='gray')

# <demo> stop
# <demo> auto

###############################################################################
# Compute local pixel average kernel.

k_avg = np.ones((AVG_SIZE, AVG_SIZE)) / AVG_SIZE**2
steps['average'] = ndimg.convolve(steps['luminance'], k_avg)
print "Average kernel luminance image size " + str(k_avg.shape) + ":"
plab.imshow(steps['average'], cmap='gray')

###############################################################################
# Compute local pixel variance.

steps['diff_mean'] = steps['luminance'] - steps['average']
steps['diff_mean_sq'] = steps['diff_mean'] * steps['diff_mean']
steps['variance'] = ndimg.convolve(steps['diff_mean_sq'], k_avg)
print "Pixel-wise variance:"
plab.imshow(steps['variance'], cmap='gray')

# <demo> stop
# <demo> auto

###############################################################################
# Three sigmas image and binary threshold.

steps['maha_sq'] = (steps['diff_mean'] > 0) * steps['diff_mean_sq'] / \
                   steps['variance']
steps['thresh_maha'] = (steps['maha_sq'] > (NUM_STDDEV * NUM_STDDEV))
print "Bianry image from local maha:"
plab.imshow(steps['thresh_maha'], cmap='gray')

# <demo> stop
# <demo> auto

print "Detected light regions using maha with {0} "\
      "standard deviations:".format(NUM_STDDEV)
plab.imshow(pymorph.overlay(steps['luminance'].astype('uint8'),
                            steps['thresh_maha']))

# <demo> stop
# <demo> auto

###############################################################################
# Integrate global illumination effects by taking a top percentage of
# intensities from the detected light regions.

steps['masked_regions_lum'] = steps['thresh_maha'] * steps['luminance']
steps['masked_regions_hist'] = pymorph.histogram(steps['masked_regions_lum'])
steps['global_bright_thresh'] = int((len(steps['masked_regions_hist']) * \
                                     (1.0 - GLOBAL_BRIGHT_PCT)) + 0.5)
steps['thresh_global'] = steps['masked_regions_lum'] >= \
                         steps['global_bright_thresh']
print "Global filtered mask:"
plab.imshow(pymorph.overlay(steps['luminance'].astype('uint8'),
                            steps['thresh_global']))

###############################################################################
# Morpohological operations on detected blobs.

# <demo> stop
# <demo> auto

steps['detect_erode'] = pymorph.erode(steps['thresh_global'])
steps['detect_dilate'] = pymorph.dilate(steps['detect_erode'])
print "Morphed mask (erode, dilate):"
plab.imshow(pymorph.overlay(steps['luminance'].astype('uint8'),
                            steps['detect_dilate']))

# <demo> stop
# <demo> auto

# Count bright objects. Connected components and raw pixels.
steps['detect_labels'] = pymorph.label(steps['detect_dilate'])
steps['bright_blob_count'] = steps['detect_labels'].max()
print "Bright blob count:", steps['bright_blob_count']
steps['bright_pixel_count'] = sum(steps['masked_regions_hist']
                                       [steps['global_bright_thresh']:])
print "Bright pixel count:", steps['bright_pixel_count']

print "Input image:"
plab.imshow(steps['input'])

# <demo> stop

