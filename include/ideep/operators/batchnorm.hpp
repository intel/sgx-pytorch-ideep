#ifndef IDEEP_OPERATORS_BATCHNORM_HPP
#define IDEEP_OPERATORS_BATCHNORM_HPP
#include "sum.hpp"

#include <unistd.h>

namespace ideep {

struct batch_normalization_forward_inference
    : public dnnl::batch_normalization_forward {

  using super = dnnl::batch_normalization_forward;

  static void compute(const tensor& src,
                      const tensor& scale,
                      const tensor& shift,
                      tensor& dst,
                      float epsilon,
                      const engine& aengine = engine::cpu_engine()) {
    static tensor dummy;
    compute_impl</*use_stats=*/false>(
        src, dummy, dummy, scale, shift, dst, epsilon, NULL, 0, NULL, 0, NULL, 0, NULL, 0, NULL, NULL, aengine);
  }

  static void compute(const tensor& src,
                      const tensor& mean,
                      const tensor& variance,
                      const tensor& scale,
                      const tensor& shift,
                      tensor& dst,
                      float epsilon,
		      void* weight_iv_mac = NULL,
		      size_t weight_meta_size = 0,
		      void* bias_iv_mac = NULL,
		      size_t bias_meta_size = 0,
		      void* running_mean_iv_mac = NULL,
		      size_t running_mean_meta_data_size = 0,
		      void* running_var_iv_mac = NULL,
		      size_t running_var_meta_data_size = 0,
		      void* model_id = NULL,
		      sgx_enclave_id_t *eid = NULL,
                      const engine& aengine = engine::cpu_engine()) {
    compute_impl</*use_stats=*/true>(
        src, mean, variance, scale, shift, dst, epsilon, weight_iv_mac, weight_meta_size, bias_iv_mac, bias_meta_size, running_mean_iv_mac, running_mean_meta_data_size, running_var_iv_mac, running_var_meta_data_size, model_id, eid, aengine);
  }
 private:
  template <bool use_stats>
  static void compute_impl(const tensor& src,
                           const tensor& mean,
                           const tensor& variance,
                           const tensor& scale,
                           const tensor& shift,
                           tensor& dst,
                           float epsilon,
			   void* weight_iv_mac,
			   size_t weight_meta_size,
			   void* bias_iv_mac,
			   size_t bias_meta_size,
                           void* running_mean_iv_mac,
			   size_t running_mean_meta_data_size,
			   void* running_var_iv_mac,
			   size_t running_var_meta_data_size,
			   void* model_id,
			   sgx_enclave_id_t *eid,
                           const engine& aengine) {
    auto flags = batch_normalization_flag::use_scale_shift;
    if (use_stats)
      flags |= batch_normalization_flag::use_global_stats;

    // workaround: use src.get_desc() once issue intel/mkl-dnn#588 is resolved
    auto src_desc = src._get_unblocked_desc_if_4c_blocked();
    // auto src_desc = src.get_desc();

    auto pd = primitive_desc(
        {prop_kind::forward_inference, src_desc, epsilon, flags}, aengine);

    tensor scale_shift {pd.weights_desc()};
    auto* scale_shift_buf = static_cast<char *>(scale_shift.get_data_handle());
    std::memcpy(scale_shift_buf, scale.get_data_handle(), scale.get_size());
    std::memcpy(scale_shift_buf + scale.get_size(),
                shift.get_data_handle(), shift.get_size());
    auto expected_src = src.reorder_if_differ_in(pd.src_desc());
    dst.reinit_if_possible(pd.dst_desc());

    if (weight_iv_mac and bias_iv_mac) {
    auto bn_src_handle = src.get_data_handle();
    auto bn_var_handle = variance.get_data_handle();
    auto bn_mean_handle = mean.get_data_handle();
    auto bn_scale_shift_handle = scale_shift.get_data_handle();

    size_t bn_src_size = src.get_size();
    size_t bn_var_size = variance.get_size();
    size_t bn_mean_size = mean.get_size();
    size_t bn_scale_shift_size = scale_shift.get_size();

    void* void_dst = (void*)(dst.get_data_handle());
    size_t dst_data_size = dst.get_desc().get_size();

    auto bn_pd = dnnl::batch_normalization_forward::desc(prop_kind::forward_inference, pd.src_desc(), epsilon, flags);
    void* void_bn_pd = (void*)&bn_pd;
    size_t bn_pd_size = sizeof(bn_pd);
    
    uint32_t model_id_ = *((uint32_t*)model_id);

    if (eid == NULL)
            return;

    if (*eid == 0) {
        if (initialize_enclave(eid) < 0) {
            printf("initialize enclave failed... \n");
            return;
        }
        printf("batch norm initialize enclave success: eid is %d.\n", *eid);
    }

    sgx_status_t retval;
    sgx_status_t ret = ecall_batch_norm_dnnl_function(*eid, &retval, void_bn_pd, bn_pd_size, bn_src_handle, bn_src_size, bn_var_handle, bn_var_size, bn_mean_handle, bn_mean_size, bn_scale_shift_handle, bn_scale_shift_size, scale.get_size(), shift.get_size(), void_dst, dst_data_size, weight_iv_mac, weight_meta_size, bias_iv_mac, bias_meta_size, running_mean_iv_mac, running_mean_meta_data_size, running_var_iv_mac, running_var_meta_data_size, model_id_);
    }
    else{
    if (use_stats) {
//      std::cout << "use_state true and in compute_impl" << std::endl;
      auto expected_mean = mean.reorder_if_differ_in(pd.mean_desc());
/*      if (pd.mean_desc() != mean.get_desc()){
          std::cout << "mean not equal" <<std::endl;
      }
      else {
          std::cout << "mean equal" <<std::endl;
      }
*/
      auto expected_var = variance.reorder_if_differ_in(pd.variance_desc());
/*      if (pd.variance_desc() != variance.get_desc()){
          std::cout << "var not equal" <<std::endl;
      }
      else {
          std::cout << "var equal" <<std::endl;
      }
*/
      super(pd).execute(stream::default_stream(),
                        {{DNNL_ARG_SRC, expected_src},
                         {DNNL_ARG_SCALE_SHIFT, scale_shift},
                         {DNNL_ARG_VARIANCE, expected_var},
                         {DNNL_ARG_MEAN, expected_mean},
                         {DNNL_ARG_DST, dst}});
    } else {
//      std::cout << "use_state false and in compute_impl" << std::endl;
      super(pd).execute(stream::default_stream(),
                        {{DNNL_ARG_SRC, expected_src},
                         {DNNL_ARG_SCALE_SHIFT, scale_shift},
                         {DNNL_ARG_DST, dst}});
    }
    }// if weight_iv_mac and bias_iv_mac

  }
};

struct batch_normalization_forward_training
    : public dnnl::batch_normalization_forward {

  using super = dnnl::batch_normalization_forward;

  static void compute(const tensor& src,
                      const tensor& scale,
                      const tensor& shift,
                      tensor& dst,
                      tensor& mean,
                      tensor& variance,
                      float momentum,
                      float epsilon,
                      const engine& aengine = engine::cpu_engine()) {
    auto flags = batch_normalization_flag::use_scale_shift;

    // workaround: use src.get_desc() once issue intel/mkl-dnn#588 is resolved
    auto src_desc = src._get_unblocked_desc_if_4c_blocked();
    // auto src_desc = src.get_desc();

    auto pd = primitive_desc(
        {prop_kind::forward_training, src_desc, epsilon, flags}, aengine);

    tensor scale_shift {pd.weights_desc()};
    auto* scale_shift_buf = static_cast<char *>(scale_shift.get_data_handle());
    std::memcpy(scale_shift_buf, scale.get_data_handle(), scale.get_size());
    std::memcpy(scale_shift_buf + scale.get_size(),
                shift.get_data_handle(), shift.get_size());
    auto expected_src = src.reorder_if_differ_in(pd.src_desc());
    mean.reinit_if_possible(pd.mean_desc());
    variance.reinit_if_possible(pd.variance_desc());
    dst.reinit_if_possible(pd.dst_desc());

    super(pd).execute(stream::default_stream(),
                      {{DNNL_ARG_SRC, expected_src},
                       {DNNL_ARG_SCALE_SHIFT, scale_shift},
                       {DNNL_ARG_MEAN, mean},
                       {DNNL_ARG_VARIANCE, variance},
                       {DNNL_ARG_DST, dst}});
  }

  static void compute(const tensor& src,
                      const tensor& scale,
                      const tensor& shift,
                      tensor& dst,
                      tensor& mean,
                      tensor& variance,
                      tensor& running_mean,
                      tensor& running_var,
                      float momentum,
                      float epsilon) {
   compute(src, scale, shift, dst, mean, variance, momentum, epsilon);
   ideep::sum::compute({momentum, 1 - momentum}, {running_mean, mean},
                       running_mean);
   ideep::sum::compute({momentum, 1 - momentum}, {running_var, variance},
                       running_var);
  }
};

struct batch_normalization_backward
    : public dnnl::batch_normalization_backward {

  using super = dnnl::batch_normalization_backward;

  static void compute(const tensor& src,
                      const tensor& mean,
                      const tensor& variance,
                      const tensor& diff_dst,
                      const tensor& scale,
                      tensor& diff_src,
                      tensor& diff_scale_shift,
                      float epsilon,
                      const engine& aengine = engine::cpu_engine()) {
    // TODO: support no-affine model
    auto flags = batch_normalization_flag::use_scale_shift;
    auto src_desc = src.get_desc();
    auto forward_hints = dnnl::batch_normalization_forward::primitive_desc(
        {prop_kind::forward_training, src_desc, epsilon, flags}, aengine);

    auto pd = primitive_desc(
        {prop_kind::backward, forward_hints.dst_desc(), src_desc, epsilon, flags},
        aengine, forward_hints);

    auto expected_diff_dst = diff_dst.reorder_if_differ_in(pd.diff_dst_desc());
    auto expected_src = src.reorder_if_differ_in(pd.src_desc());
    auto expected_mean = mean.reorder_if_differ_in(pd.mean_desc());
    auto expected_variance = variance.reorder_if_differ_in(pd.variance_desc());
    diff_src.reinit_if_possible(pd.diff_src_desc());
    diff_scale_shift.reinit_if_possible(pd.diff_weights_desc());

    super(pd).execute(stream::default_stream(),
                      {{DNNL_ARG_SRC, expected_src},
                      {DNNL_ARG_DIFF_DST, expected_diff_dst},
                      {DNNL_ARG_SCALE_SHIFT, scale}, // only need scale
                      {DNNL_ARG_MEAN, expected_mean},
                      {DNNL_ARG_VARIANCE, expected_variance},
                      {DNNL_ARG_DIFF_SRC, diff_src},
                      {DNNL_ARG_DIFF_SCALE_SHIFT, diff_scale_shift}});   
  }

  static void compute(const tensor& src,
                      const tensor& mean,
                      const tensor& variance,
                      const tensor& diff_dst,
                      const tensor& scale,
                      tensor& diff_src,
                      tensor& diff_scale,
                      tensor& diff_shift,
                      float epsilon,
                      const engine& aengine = engine::cpu_engine()) {
  tensor diff_scale_shift;
  compute(src, mean, variance, diff_dst, scale, diff_src, diff_scale_shift,
          epsilon, aengine);
  diff_scale.reinit_if_possible(scale.get_desc());
  diff_shift.reinit_if_possible(scale.get_desc());
  auto* diff_scale_shift_buf =
      static_cast<char*>(diff_scale_shift.get_data_handle());
  std::memcpy(diff_scale.get_data_handle(), diff_scale_shift_buf,
              diff_scale.get_size());
  std::memcpy(diff_shift.get_data_handle(),
              diff_scale_shift_buf + diff_scale.get_size(),
              diff_shift.get_size());
  }
};

}  // namespace ideep

#endif
