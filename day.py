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
import pyopencl as cl
import PIL
import pymorph
import os
import fnmatch
import datetime
import fractions

# GMM parameters for script-run.
GMM_K = 3
GMM_NUM_FRAMES = 25
GMM_W_INIT = 0.1
GMM_VAR_INIT = 20
GMM_MAHA_THRESH = 3

def rgb2ycbcr(image):
    """
    Convert a unit8 RGB image to full-range Y'CbCr image. This is not intended
    for use directly on broadcast video systems.
    """

    assert(image.dtype == 'uint8')

    # Matrix for RGB to full-range YCbCr conversion.
    #   http://www.equasys.de/colorconversion.html
    A = np.array([[ 0.299,  0.587,  0.114],
                  [-0.169, -0.331,  0.500],
                  [ 0.500, -0.419, -0.081]], dtype='f4')
    # Convert.
    yuv = np.dot(image.astype('f4'), A.T) + \
          np.array([0., 128.5, 128.5], dtype='f4')
    return np.clip(yuv, 0, 255).astype('uint8')

# GMM:
#  * Use u_t in expression for s_t. Update mean first.
#  * w_K^N+1 = ((1 - a) * w_k^N) + (a * p(k|x_N+1, theta_t-1))
#    where a = (1 / (N + 1))
#  * u_K^N+1 = ((1 - p_k) * u_k^N) + (p_k * x_N+1)
#    where p_k = (a * p(k|x_t, theta^old)) / w_k^new
#  * s_k^N+1 = ((1 - p_k) * s_k^N) + (p_k * (x_N+1 - u_k^N+1) *
#                                           (x_N+1 - u_k^N+1)^T)
#    where p_k = (a * p(k|x_t, theta^old)) / w_k^new
class GaussianMixtureModelUV(object):
    """
    A Gaussian Mixture Model (GMM) is an online statistical background model
    invented by Stauffer and Grimson. It attempts to learn a k-modal
    mixure of gaussian distributions corresponding to k surfaces observed
    at pixel [i,j] of the image. For pixels exibiting unimodal through
    k-modal behavior, the GMM will adapt the appropriate number of modes
    evidenced by the allocation of k weights among its k mixture components.
    """

    # Number of parameters required to model one gaussian component.
    GAUSS_COMPONENT_SIZE = 5
    # Mean used during model initialization.
    MU_INIT = 32.
    # Minimum variance.
    VAR_MIN = 2.
    # Total density included in the model.
    DENSITY_THRESH = 0.8

    def __init__(self, k, size, num_frames, w_init, var_init, maha_thresh):

        # Collect CPU style params.
        self._k = k
        self._size = tuple(size[:2])
        self._num_frames = num_frames
        self._alpha = 1. / (num_frames + 1)
        self._w_init = w_init
        self._var_init = var_init
        self._maha_thresh = maha_thresh

        # Collect GPU style params.
        self._params = np.zeros((1,), dtype=[('size','2i4'),
                                             ('k','i4'),
                                             ('maha_thresh','f4'),
                                             ('alpha','f4'),
                                             ('w_init','f4'),
                                             ('var_init','f4'),
                                             ('var_min','f4'),
                                             ('density_thresh','f4')])
        self._params['size'] = np.array(self._size)
        self._params['k'] = self._k
        self._params['maha_thresh'] = self._maha_thresh
        self._params['alpha'] = self._alpha
        self._params['w_init'] = self._w_init
        self._params['var_init'] = self._var_init
        self._params['var_min'] = self.VAR_MIN
        self._params['density_thresh'] = self.DENSITY_THRESH

        # Create the model.
        model_shape = tuple(size[:2]) + (k, self.GAUSS_COMPONENT_SIZE,)
        self._model = np.zeros(model_shape, dtype='f4')

        # Initialize model.
        self._model[:,:,:,:] = np.array([1. / k,
                                         self.MU_INIT, self.MU_INIT,
                                         var_init, var_init])

        # Setup OpenCL.
        self._setup_opencl()
        self._setup_opencl_buffers()

    def _setup_opencl(self):
        """ Setup OpenCL elements for GMM. """

        # Setup OpenCL device and queue.
        os.environ['PYOPENCL_CTX'] = '1'
        self._context = cl.create_some_context()
        self._queue = cl.CommandQueue(self._context)
        # Load OpenCL program.
        program_text = open('./gmm.cl').read();
        self._program = cl.Program(self._context, program_text).build()
        # Determine work sizes.
        local_mem_size = cl.device_info.LOCAL_MEM_SIZE
        sizeof_gaussians = self._model[0][0].nbytes
        max_local_work = int(local_mem_size / sizeof_gaussians)
        local_work_dim = int(np.sqrt(max_local_work))
        self._local_work_size = (local_work_dim, local_work_dim)
        self._local_gauss_buf_size = local_work_dim * local_work_dim * \
                                     sizeof_gaussians
        assert(self._local_gauss_buf_size <= local_mem_size)
        global_work_blocks_x = int((float(self._size[0]) / local_work_dim) +
                                   0.5)
        global_work_blocks_y = int((float(self._size[1]) / local_work_dim) +
                                   0.5)
        self._global_work_size = (local_work_dim * global_work_blocks_x,
                                  local_work_dim * global_work_blocks_y)
        max_work_group_dim = int(np.sqrt(cl.device_info.MAX_WORK_GROUP_SIZE))
        rgb2ycbcr_local_work_dim = fractions.gcd(
            fractions.gcd(self._size[0], max_work_group_dim),
            fractions.gcd(self._size[1], max_work_group_dim))
        self._rgb2ycbcr_local_work_size = (rgb2ycbcr_local_work_dim,
                                           rgb2ycbcr_local_work_dim)

    def _setup_opencl_buffers(self):
        """ Setup OpenCL buffers for GMM. """

        # Setup buffers for kernels.
        self._params_buf = cl.Buffer(self._context,
                                     cl.mem_flags.READ_ONLY |
                                     cl.mem_flags.COPY_HOST_PTR,
                                     hostbuf=self._params)
        self._gauss_buf = cl.LocalMemory(self._local_gauss_buf_size)
        self._model_buf = cl.Buffer(self._context,
                                    cl.mem_flags.READ_WRITE |
                                    cl.mem_flags.COPY_HOST_PTR,
                                    hostbuf=self._model)
        image_size = self._size[0] * self._size[1] * 3
        self._img_read_buf = cl.Buffer(self._context,
                                       cl.mem_flags.READ_ONLY,
                                       size=image_size)
        self._img_write_buf = cl.Buffer(self._context,
                                        cl.mem_flags.WRITE_ONLY,
                                        size=image_size)
        mask_size = self._size[0] * self._size[1]
        self._mask_buf = cl.Buffer(self._context,
                                   cl.mem_flags.READ_WRITE,
                                   size=mask_size)
        self._queue.finish()

    def rgb2ycbcr(self, rgb, ycbcr):
        """
        Convert RGB image to ycbcr.
        WARNING: Assumes that the image is gamma-corrected and makes the
                 conversion to linear intensities.
                 See gmm.cl for more details.
        """

        assert('uint8' == rgb.dtype)
        assert(rgb.nbytes == self._img_read_buf.size)
        assert('uint8' == ycbcr.dtype)
        assert(ycbcr.nbytes == self._img_write_buf.size)

        # Convert to Y'CbCr.
        cl.enqueue_copy(self._queue, self._img_read_buf, rgb)
        self._queue.finish()
        convert_ycbcr_args = (self._img_read_buf,
                              self._params_buf,
                              self._img_write_buf)
#        print "self._rgb2ycbcr_local_work_size:", \
#              self._rgb2ycbcr_local_work_size
        self._program.rgb2ycbcr(self._queue,
                                self._global_work_size,
                                self._rgb2ycbcr_local_work_size,
                                *(convert_ycbcr_args))
        self._queue.finish()
        cl.enqueue_copy(self._queue, ycbcr, self._img_write_buf)

    def update_cl(self, ycbcr):
        """ Update the model parameters from the givem image using OpenCL. """

        assert('uint8' == ycbcr.dtype)
        assert(ycbcr.nbytes == self._img_read_buf.size)

        # Upload image.
        cl.enqueue_copy(self._queue, self._img_read_buf, ycbcr)
        self._queue.finish()
        # Execute the update kernel.
#        print "model[0][0] before:", self._model[0][0]
#        print "global_work_size:", self._global_work_size
#        print "local_work_size:", self._local_work_size
        update_gmm_args = (self._img_read_buf,
                           self._params_buf,
                           self._gauss_buf,
                           self._model_buf)
        self._program.update_gmm(self._queue,
                                 self._global_work_size,
                                 self._local_work_size,
                                 *(update_gmm_args))
        self._queue.finish()
        # Update model.
        cl.enqueue_copy(self._queue, self._model, self._model_buf)
        self._queue.finish()
#        print "model[0][0] after:", self._model[0][0]

    def segment_cl(self, mask, ycbcr=None):
        """ Segement the image using the model. """

        assert('uint8' == mask.dtype)
        assert(mask.nbytes == self._mask_buf.size)

        # Upload image.
        if ycbcr is None:
            # Copy device -> device.
            cl.enqueue_copy(self._queue, self._img_read_buf,
                                         self._img_write_buf)
        else:
            assert('uint8' == ycbcr.dtype)
            assert(ycbcr.nbytes == self._img_read_buf.size)
            cl.enqueue_copy(self._queue, self._img_read_buf, ycbcr)
        self._queue.finish()
        segment_gmm_args = (self._img_read_buf,
                            self._params_buf,
                            self._model_buf,
                            self._mask_buf)
        # Execute kernel.
#        print "global_work_size:", self._global_work_size
#        print "local_work_size:", self._local_work_size
        self._program.segment_gmm(self._queue,
                                  self._global_work_size,
                                  self._local_work_size,
                                  *(segment_gmm_args))
        self._queue.finish()
        # Update mask.
        cl.enqueue_copy(self._queue, mask, self._mask_buf)
        self._queue.finish()

    def update_cpu(self, image):
        """ Update the model parameters from the given image.  """

        match = np.zeros(self._size + (self._k,), dtype='bool')
        for k in range(self._k):
            diff = self._model[:,:,k,1:3] - image[:,:,1:]
            maha_sq = (diff * diff) / self._model[:,:,k,3:5]
            maha_sq_thresh = (maha_sq < self._maha_thresh**2)
            match[:,:,k] = (maha_sq_thresh[:,:,0] * maha_sq_thresh[:,:,1])

        # Go over each pixel and complete update.
        k_indices = np.array(range(self._k))
        for r in range(self._size[0]):
            for c in range(self._size[1]):
                px_match = match[r,c]
                matches = np.sum(px_match)
                # Create new model component?
                if matches == 0:
                    pass
                else:
                    # Exactly one match or tiebreaker on multiple matches.
                    if matches == 1:
                        match_idx = np.dot(px_match, k_indices)
                    else:
                        max_match = float("-inf")
                        for k, matched in zip(k_indices, px_match):
                            if not matched:
                                continue
                            w_k = self._model[r,c,k,0]
                            var_k = self._model[r,c,k,1:3]
                            match_mag = w_k**4 / np.dot(var_k, var_k)
                            if match_mag > max_match:
                                max_match = match_mag
                                match_idx = k
                    assert((match_idx >= 0) and (match_idx < self._k))

                    model_component = self._model[r,c,match_idx]
                    # Update weight.
                    model_component[0] = ((1 - self._alpha) *
                                          model_component[0]) + self._alpha
                    # Compute update for mean/variance.
                    p_m = self._alpha / model_component[0]
                    # Update mean.
                    px = image[r,c,1:]
                    model_component[1:3] = ((1 - p_m) *
                                            model_component[1:3]) + (p_m * px)
                    diff_mean = px.astype('float') - model_component[1:3]
                    model_component[3:5] = ((1 - p_m) *
                                            model_component[3:5]) + \
                                           (p_m * diff_mean * diff_mean)

                    # Renormalize weights.
                    self._model[r,c,:,0] /= np.sum(self._model[r,c,:,0])
                    self._model[r,c] = np.sort(self._model[r,c], axis=0)
                    # Sort and reverse.
                    self._model[r,c] = np.sort(self._model[r,c], axis=0)[::-1]


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
    if len(argv) < 2 or len(argv) > 4 or len(argv) is 3:
        print "Usage: {0} SERIES_FILE_PATH [START STOP]".format(argv[0])
        raise Exception("Invalid command-line parameters.")
    # Extract command-line parameters. This is the name of one file in the
    # series.
    path, filename = os.path.split(argv[1].strip('"\''))
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
    if len(argv) is 4:
        start = eval(argv[2])
        stop = eval(argv[3])
        print "Filtering series range [{},{}).".format(start, stop)
        series_suffixes = np.array(series_suffixes)
        f = (series_suffixes >= start) * (series_suffixes < stop)
        series_suffixes = np.sort(series_suffixes * f)
        series_suffixes = np.delete(series_suffixes,
                                    range(len(series_suffixes) - np.sum(f)))
    os.mkdir(label_img_path)
    # Init GMM.
    series_filename = series_name + '_' + str(series_suffixes[0]) + file_ext
    rgb = ndimg.imread(os.path.join(path, series_filename))
    gmm = GaussianMixtureModelUV(GMM_K, rgb.shape, GMM_NUM_FRAMES,
                                 GMM_W_INIT, GMM_VAR_INIT, GMM_MAHA_THRESH)
    ycbcr = np.zeros(rgb.shape, dtype='uint8')
    mask = np.zeros(rgb.shape[:2], dtype='uint8')
    # Process the files.
    labeled_foreground_pixels = []
    for suffix in series_suffixes:
        series_filename = series_name + '_' + str(suffix) + file_ext
        print "Processing file {0}.".format(series_filename)
        rgb = ndimg.imread(os.path.join(path, series_filename))
        gmm.rgb2ycbcr(rgb, ycbcr)
        gmm.segment_cl(mask)
        gmm.update_cl(ycbcr)
        # Save total pixels in foreground.
        labeled_foreground_pixels.append(np.sum(mask))
        # Save labeled file.
        label_img_filename = os.path.join(label_img_path,
                                          str(suffix) + file_ext)
        PIL.Image.fromarray(mask * 255).save(label_img_filename)
    # Compute exponential moving averages.
    plot_frames = labeled_foreground_pixels[2 * GMM_NUM_FRAMES:]
    pixel_exp_avg = exponential_moving_average(plot_frames, 0.1)
    pixel_exp_avg_norm = np.array(pixel_exp_avg) / max(pixel_exp_avg)
    # Make plots.
    p = plab.subplot('111')
    p.plot(pixel_exp_avg_norm, 'b', label='pixel_count')
    p.set_title("Normalized foreground pixel count in {0} sequence."
                .format(series_name))
    p.set_ylabel('normalized count')
    p.set_xlabel('frame index')
#    p.legend(loc=4)
    # Save plot.
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
    os.write(results_file, str(labeled_foreground_pixels))
    os.close(results_file)
    return 0

if __name__ == "__main__":
    main(sys.argv)

