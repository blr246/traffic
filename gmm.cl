/*******************************************************************************
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

The following OpenCL shader implements a Gaussian Mixture Model background
subtraction algorithm of the type originall report by Sauffer and Grimson
in "Adaptive background mixture models for real-time tracking".
*******************************************************************************/


// A GMM model component.
typedef struct __attribute__ ((packed)) _Gaussian
{
  // Mixture density.
  float w;
  // Mean.
  float2 mu;
  // Variance.
  float2 var;
} Gaussian;
enum { FloatsPerGauss = sizeof(Gaussian) / sizeof(float), };

// The GMM parameters.
typedef struct __attribute__ ((packed)) _GmmParams
{
  // Image size.
  int2 size;
  // Number of model components.
  int k;
  // Number of standard deviations to consider a match.
  float maha_thresh;
  // Learning rate.
  float alpha;
  // Initial component mixture density.
  float w_init;
  // Initial component variance.
  float var_init;
  // Minimum variance.
  float var_min;
  // Total model density.
  float density_thresh;
} GmmParams;
enum { IntsPerGmmParams = sizeof(GmmParams) / sizeof(int), };

// Never have more than 10 modes.
#define MAX_GMM_K 10
// Image gamma correction. This is inverted by the conversion to Y'CbCr.
#define GAMMA 2.2f

// Struct to assist in matching model components.
typedef struct _ModelMatchParams
{
  int idx;
  float dist;
} ModelMatchParams;

// Compute the match distance metric.
float match_dist_w_var(const float w, const float2* var)
{
  return pown(w, 4) / dot(*var, *var);
}

// Compute the match distance metric.
float match_dist(__local const Gaussian* gauss)
{
  const float2 var = gauss->var;
  return match_dist_w_var(gauss->w, &var);
}

// Compute Mahaolonobis distance squared.
float maha_dist_sq_mu_var(const float2* uv,
                          const float2* mu, const float2* var)
{
  const float2 diff = *uv - *mu;
  const float2 maha_sq_ele = (diff * diff) / *var;
  return dot(maha_sq_ele, (float2)(1.0f, 1.0f));
}

// Compute Mahaolonobis distance squared.
float maha_dist_sq(const float2* uv, __local const Gaussian* gauss)
{
  const float2 diff = *uv - gauss->mu;
  const float2 maha_sq_ele = (diff * diff) / gauss->var;
  return dot(maha_sq_ele, (float2)(1.0f, 1.0f));
}

// Update the GMM with the current image data.
//   image   -> Input image used to update model.
//   params_ -> GMM parameters (size, thresholds, etc).
//   gauss   -> Local storage for model update.
//   model   -> The global model [in, out].
__kernel void update_gmm(__constant uchar* image,
                         __constant GmmParams* params_,
                         __local Gaussian* gauss,
                         __global Gaussian* model)
{
  // Get global threadid.
  const int2 gid = (int2)(get_global_id(0), get_global_id(1));
  const int2 gid_size = (int2)(get_global_size(0), get_global_size(1));

  // reissb -- 20111204 -- May be better to load one __local copy of
  //   params or some other method. This is easiest for now.
  // Just make per-thread copies of params.
  GmmParams params = *params_;

  // Only process within image.
  if ((gid.x >= params.size.x) || (gid.y >= params.size.y))
  {
    return;
  }

  // Get local threadid.
  const int2 lid = (int2)(get_local_id(0), get_local_id(1));
  const int2 lid_size = (int2)(get_local_size(0), get_local_size(1));

  // Compute pixel index and local index. Inputs are (rows, cols).
  const int gid_offset = (gid.x * gid_size.y) + gid.y;
  const int lid_offset = (lid.x * lid_size.y) + lid.y;

  // Compute model pointers.
  __global Gaussian* const model_begin = model + (gid_offset * params.k);
  __local Gaussian* const gauss_begin = gauss + (lid_offset * params.k);

//  // 20111204 -- reissb -- Debug theadids.
//  (model_begin + 2)->mu = convert_float2(gid);
//  (model_begin + 2)->var = convert_float2(lid);
//  return;

//  // Copy first model and async copy rest.
//  gauss_begin[0] = model_begin[0];
//  // Async copy must be the same for the entire work group.
//  event_t model_copy_event = 0;
//  {
//    // Get work group origin in global coordinates.
//    const int2 gid_lcl_o = gid - lid;
//    const int gid_lcl_o_offset = (gid_lcl_o.x * gid_size.y) + gid_lcl_o.y;
//    const int gid_lcl_o_model_idx = gid_lcl_o_offset * params.k;
//    __global const float* const src_base = (__global const float*)
//                                           (model + gid_lcl_o_model_idx);
//    const int model_pitch = gid_size.y * params.k * FloatsPerGauss;
//    // Compute floats per local row.
//    const int gauss_per_l_row = lid_size.y * params.k;
//    const int num_elements = gauss_per_l_row * FloatsPerGauss;
//    for (int l_row = 0; l_row < lid_size.x; ++l_row)
//    {
//      __global const float* const src = src_base + (l_row * model_pitch);
//      __local float* const dst = (__local float*)
//                                 (gauss + (l_row * gauss_per_l_row));
//      // Take only the model copy event for your local row.
//      event_t l_row_model_copy_event;
//      l_row_model_copy_event = async_work_group_copy(dst, src,
//                                                     num_elements, 0);
//      if (lid.x == l_row)
//      {
//        model_copy_event = l_row_model_copy_event;
//      }
//    }
//  }

  // Get image pixel.
  __constant uchar* yuv = image + (gid_offset * 3);
  const float2 uv = (float2)(*(yuv + 1), *(yuv + 2));

  // Find closest mode.
  const float maha_thresh_sq = params.maha_thresh * params.maha_thresh;
  __local const Gaussian* gauss_comp = gauss_begin;
  ModelMatchParams match_params;
  {
    // Copy model first.
    gauss_begin[0] = model_begin[0];
    // See if first model matches.
    const float maha_sq = maha_dist_sq(&uv, gauss_comp);
    if (maha_sq < maha_thresh_sq)
    {
      match_params.idx = 0;
      match_params.dist = match_dist(gauss_comp);
    }
    // Initialize match to nothing.
    else
    {
      match_params.idx = -1;
      match_params.dist = -INFINITY;
    }
  }
//  // Wait on async copy.
//  wait_group_events(1, &model_copy_event);
  // Loop to match remaining Gaussians.
  ++gauss_comp;
  for (int comp_idx = 1; comp_idx < params.k; ++comp_idx, ++gauss_comp)
  {
    // Copy model first.
    gauss_begin[comp_idx] = model_begin[comp_idx];
    const float maha_sq = maha_dist_sq(&uv, gauss_comp);
    const float dist = match_dist(gauss_comp);
    if ((maha_sq < maha_thresh_sq) && (dist > match_params.dist))
    {
      match_params.idx = comp_idx;
      match_params.dist = dist;
    }
  }

  // No match?
  if (match_params.idx < 0)
  {
    // Create new model component (replace last).
    __local Gaussian* const gauss_upd = gauss_begin + (params.k - 1);
    gauss_upd->w = params.w_init;
    gauss_upd->mu = uv;
    gauss_upd->var = (float2)(params.var_init, params.var_init);
  }
  // Update matched model.
  else
  {
    __local Gaussian* const gauss_upd = gauss_begin + match_params.idx;
    gauss_upd->w = ((1.0f - params.alpha) * gauss_upd->w) + params.alpha;
    const float p_m = params.alpha / gauss_upd->w;
    gauss_upd->mu = ((1.0f - p_m) * gauss_upd->mu) + (p_m * uv);
    const float2 diff = uv - gauss_upd->mu;
    gauss_upd->var = ((1.0f - p_m) * gauss_upd->var) + (p_m * (diff * diff));
    // Enforce minimum variance.
    gauss_upd->var = max(params.var_min, gauss_upd->var);
  }

  // reissb -- 20111204 -- This is computed explicitly to avoid accumulating
  //   errors from floating-point computation. We could just take the change
  //   in mixture density during the update, but that may carry error.
  // Normalize mixture densities.
  {
    float total_density = 0.0f;
    // Sum.
    {
      gauss_comp = gauss_begin;
      for (int comp_idx = 0; comp_idx < params.k; ++comp_idx, ++gauss_comp)
      {
        total_density += gauss_comp->w;
      }
    }
    // Divide.
    {
      __local Gaussian* gauss_upd = gauss_begin;
      for (int comp_idx = 0; comp_idx < params.k; ++comp_idx, ++gauss_upd)
      {
        gauss_upd->w /= total_density;
      }
    }
  }

  // Update global model. Check if matched first.
  {
    if (match_params.idx >= 0)
    {
      // Sort gaussians and copy them back.
      float sort_scores[MAX_GMM_K];
      gauss_comp = gauss_begin;
      for (int comp_idx = 0; comp_idx < params.k; ++comp_idx, ++gauss_comp)
      {
        sort_scores[comp_idx] = match_dist(gauss_comp);
      }
      // Double loop to find max score and update per iteration.
      __global Gaussian* model_upd = model_begin;
      for (int comp_idx = 0; comp_idx < params.k; ++comp_idx, ++model_upd)
      {
        int max_score_idx = 0;
        float max_score = sort_scores[0];
        for (int score_idx = 1; score_idx < params.k; ++score_idx)
        {
          if (sort_scores[score_idx] > max_score)
          {
            max_score_idx = score_idx;
            max_score = sort_scores[score_idx];
          }
        }
        // Copy model component and cross out score.
        *model_upd = gauss_begin[max_score_idx];
        sort_scores[max_score_idx] = -1.0f;
      }
    }
    else
    {
      // Just copy back (sorted already).
      gauss_comp = gauss_begin;
      __global Gaussian* model_upd = model_begin;
      for (int comp_idx = 0;
           comp_idx < params.k;
           ++comp_idx, ++gauss_comp, ++model_upd)
      {
        *model_upd = *gauss_comp;
      }
    }
  }
}

// Segment the GMM with the current image data.
//   image   -> Input image used to update model.
//   model   -> The global model.
//   params_ -> GMM parameters (size, thresholds, etc).
//   gauss   -> Local storage for model update.
//   mask    -> The segmented background where 1=foreground.
__kernel void segment_gmm(__constant uchar* image,
                          __constant GmmParams* params_,
                          __constant Gaussian* model,
                          __global uchar* mask)
{
  // Get global threadid.
  const int2 gid = (int2)(get_global_id(0), get_global_id(1));
  const int2 gid_size = (int2)(get_global_size(0), get_global_size(1));

  // reissb -- 20111204 -- May be better to load one __local copy of
  //   params or some other method. This is easiest for now.
  // Just make per-thread copies of params.
  GmmParams params = *params_;

  // Only process within image.
  if ((gid.x >= params.size.x) || (gid.y >= params.size.y))
  {
    return;
  }

  // Get local threadid.
  const int2 lid = (int2)(get_local_id(0), get_local_id(1));
  const int2 lid_size = (int2)(get_local_size(0), get_local_size(1));

  // Compute pixel index and local index. Inputs are (rows, cols).
  const int gid_offset = (gid.x * gid_size.y) + gid.y;

  // Compute model pointers.
  __constant Gaussian* const model_begin = model + (gid_offset * params.k);

  // Get image pixel.
  __constant uchar* yuv = image + (gid_offset * 3);
  const float2 uv = (float2)(*(yuv + 1), *(yuv + 2));
  // Get output pixel.
  __global uchar* masked = mask + gid_offset;

  // Go through modes up to density_thresh looking for a match.
  *masked = 1;
  float density = 0;
  const float maha_thresh_sq = params.maha_thresh * params.maha_thresh;
  __constant Gaussian* model_comp = model_begin;
  for (int k = 0; k < params.k; ++k, ++model_comp)
  {
    const Gaussian gauss_comp = *model_comp;
    const float maha_sq = maha_dist_sq_mu_var(&uv, &gauss_comp.mu,
                                                   &gauss_comp.var);
    // Is background?
    if (maha_sq < maha_thresh_sq)
    {
      *masked = 0;
      break;
    }
    // Model density reached?
    density += gauss_comp.w;
    if (density >= params.density_thresh)
    {
      break;
    }
  }
}

// Convert the rgb image to Y'CbCr.
//   image_rgb   -> Input image in RGB.
//   image_ycbcr -> RGB image converted to Y'CbCr.
__kernel void rgb2ycbcr(__constant uchar* rgb_image,
                        __constant GmmParams* params,
                        __global uchar* ycbcr_image)
{
  // Get global threadid.
  const int2 gid = (int2)(get_global_id(0), get_global_id(1));
  const int2 gid_size = (int2)(get_global_size(0), get_global_size(1));

  // Only process within image.
  const int2 size = params->size;
  if ((gid.x >= size.x) || (gid.y >= size.y))
  {
    return;
  }

  // Compute pixel index. Inputs are (rows, cols).
  const int gid_offset = (gid.x * gid_size.y) + gid.y;

  // Ported from Numpy with the following conversion:
  //   A = np.array([[ 0.299,  0.587,  0.114],
  //                 [-0.169, -0.331,  0.500],
  //                 [ 0.500, -0.419, -0.081]])
  //   Y'CbCr = np.clip(np.dot(rgb, A.T) +
  //                    np.array([0, 128.5, 128.5]), 0, 255).astype('uint8')
  __constant uchar* in_rgb = rgb_image + (gid_offset * 3);
  const float3 rgb_gamma_norm = (float3)(*(in_rgb + 0),
                                         *(in_rgb + 1),
                                         *(in_rgb + 2)) / 255.0f;
  const float3 rgb = pow(rgb_gamma_norm, GAMMA) * 255.0f;
  const float y  = dot(rgb, (float3)( 0.299f,  0.587f,  0.114f));
  const float cb = dot(rgb, (float3)(-0.169f, -0.331f,  0.500f));
  const float cr = dot(rgb, (float3)( 0.500f, -0.419f, -0.081f));
  const float3 off = (float3)(0.0f, 128.5f, 128.5f);
  const float3 ycbcr = clamp((float3)(y, cb, cr) + off, 0.0f, 255.0f);
  __global uchar* out_ycbcr = ycbcr_image + (gid_offset * 3);
  *(out_ycbcr + 0) = ycbcr.x;
  *(out_ycbcr + 1) = ycbcr.y;
  *(out_ycbcr + 2) = ycbcr.z;
//#define INT_SCALE (1000)
//  const int r = pow(rgb_gamma_norm.x, GAMMA) * 255.0f * INT_SCALE;
//  const int g = pow(rgb_gamma_norm.y, GAMMA) * 255.0f * INT_SCALE;
//  const int b = pow(rgb_gamma_norm.z, GAMMA) * 255.0f * INT_SCALE;
//  const int y  = (r *  299) + (g *  587) + (b *  114);
//  const int cb = (r * -169) + (g * -331) + (b *  500) + 128500;
//  const int cr = (r *  500) + (g * -419) + (b * -81)  + 128500;
//  const int3 ycbcr = clamp((int3)(y, cb, cr), 0, 255000);
//  __global uchar* out_ycbcr = ycbcr_image + (gid_offset * 3);
//  *(out_ycbcr + 0) = ycbcr.x / INT_SCALE;
//  *(out_ycbcr + 1) = ycbcr.y / INT_SCALE;
//  *(out_ycbcr + 2) = ycbcr.z / INT_SCALE;
}
