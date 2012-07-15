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
"""
import numpy as np
import pyopencl as cl
import scipy.ndimage as ndimg
import pylab
import os
import sys
import fnmatch
from day import *

# <demo> auto

# Extract command-line parameters. This is the name of one file in the
# series.
path, filename = os.path.split(sys.argv[1])
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
# Create background model.
series_filename = series_name + '_' + str(series_suffixes[0]) + file_ext
rgb = ndimg.imread(os.path.join(path, series_filename))
ycbcr = np.zeros(rgb.shape, dtype='uint8')
mask = np.zeros(rgb.shape[:2], dtype='uint8')
gmm = GaussianMixtureModelUV(3, ycbcr.shape, 25, 0.1, 20, 3)
# Process the files.
suffix_iter = iter(series_suffixes)

# <demo> stop
# <demo> auto

series_filename = series_name + '_' + str(next(suffix_iter)) + file_ext
print "Processing file {0}.".format(series_filename)
rgb = ndimg.imread(os.path.join(path, series_filename))
gmm.rgb2ycbcr(rgb, ycbcr)
gmm.segment_cl(mask)
gmm.update_cl(ycbcr)

# TODO(reissb) -- 20111206 -- Write gamma uncorrect shader and add to the
#   GMM pipeline. Removing gamma correction gives better results. Need to
#   experiment a bit further with gamma values.

# <demo> stop
# <demo> auto

pylab.imshow(mask)

# <demo> stop
# <demo> auto

print "Finished."

