#ifndef IDEEP_OPERATORS_INNER_PRODUCT_HPP
#define IDEEP_OPERATORS_INNER_PRODUCT_HPP

#include <unistd.h>

namespace ideep {

struct inner_product_forward : public dnnl::inner_product_forward {

  using super = dnnl::inner_product_forward;

  static void compute(const tensor& src,
                      const tensor& weights,
                      const tensor& bias,
                      tensor& dst,
		      void* weight_iv_mac = NULL,
		      size_t weight_meta_size = 0,
		      void* bias_iv_mac = NULL,
		      size_t bias_meta_size = 0,
		      void* model_id = NULL,
		      sgx_enclave_id_t *eid = NULL,
                      const scale_t& src_scales = scale_t(),
                      const scale_t& weights_scales = scale_t(),
                      const scale_t& dst_scales = scale_t(),
                      const attr_t& attr = attr_t(),
                      const prop_kind aprop_kind = prop_kind::forward,
                      const lowp_kind alowp_kind = u8s8,
                      const engine& aengine = engine::cpu_engine()) {
    compute_impl</*with_bias=*/true>(src, weights, bias, dst, weight_iv_mac, weight_meta_size, bias_iv_mac, bias_meta_size, model_id, eid, src_scales,
                                     weights_scales, dst_scales, attr,
                                     aprop_kind, alowp_kind, aengine);
  }

  static void compute(const tensor& src,
                      const tensor& weights,
                      tensor& dst,
                      void* weight_iv_mac = NULL,
                      size_t weight_meta_size = 0,
		      void* model_id = NULL,
		      sgx_enclave_id_t *eid = NULL,
                      const scale_t& src_scales = scale_t(),
                      const scale_t& weights_scales = scale_t(),
                      const scale_t& dst_scales = scale_t(),
                      const attr_t& attr = attr_t(),
                      const prop_kind aprop_kind = prop_kind::forward,
                      const lowp_kind alowp_kind = u8s8,
                      const engine& aengine = engine::cpu_engine()) {
    static tensor dummy_bias;
    void* bias_iv_mac = NULL;
    size_t bias_meta_size = 0;
    compute_impl</*with_bias=*/false>(src, weights, dummy_bias, dst, weight_iv_mac, weight_meta_size, bias_iv_mac, bias_meta_size, model_id, eid, src_scales,
                                      weights_scales, dst_scales, attr,
                                      aprop_kind, alowp_kind, aengine);
  }


  static tensor::desc expected_weights_desc(
      const dims& weights_dims,
      const dims& src_dims = dims(),
      data_type dtype = data_type::f32,
      data_type x_dtype = data_type::f32,
      prop_kind aprop_kind = prop_kind::forward,
      const engine& aengine = engine::cpu_engine()) {
    auto x_dims = weights_dims;
    x_dims[0] = src_dims.empty() ? 1 : src_dims[0];
    auto y_dims = {x_dims[0], weights_dims[0]};
    auto ndims = weights_dims.size();
    auto y_dtype = (dtype != data_type::s8) ? dtype : data_type::s32;

    IDEEP_ENFORCE(x_dims.size() == weights_dims.size(),
                  "Invalid dims for data and weights");
    tensor::desc src_desc(x_dims, x_dtype, tag::any);
    tensor::desc dst_desc(y_dims, y_dtype, tag::any);
    tensor::desc weights_desc(weights_dims, dtype, tag::any);
    auto pd =
        primitive_desc({aprop_kind, src_desc, weights_desc, dst_desc}, aengine);
    return pd.weights_desc();
  }

private:
  template <bool with_bias>
  static void compute_impl(const tensor& src,
                           const tensor& weights,
                           const tensor& bias,
                           tensor& dst,
                           void* weight_iv_mac,
                           size_t weight_meta_size,
                           void* bias_iv_mac,
                           size_t bias_meta_size,
			   void* model_id,
			   sgx_enclave_id_t *eid,
                           const scale_t& src_scales,
                           const scale_t& weights_scales,
                           const scale_t& dst_scales,
                           const attr_t& attr,
                           const prop_kind aprop_kind,
                           const lowp_kind alowp_kind,
                           const engine& aengine) {
    // workaround: src and weights from caffe2 may have different dims.
    // It would be better for caffe2 to do this reshape anyway.
    auto src_ = src;
    if (src.ndims() != weights.ndims()) {
      auto new_dims = weights.get_dims();
      new_dims[0] = src.get_dim(0);
      src_.reshape(new_dims);
    }
    compute_impl_<with_bias>(src_, weights, bias, dst, weight_iv_mac, weight_meta_size, bias_iv_mac, bias_meta_size, model_id, eid, src_scales,
                             weights_scales, dst_scales, attr, aprop_kind,
                             alowp_kind, aengine);
  }

  template <bool with_bias>
  static void compute_impl_(const tensor& src,
                            const tensor& weights,
                            const tensor& bias,
                            tensor& dst,
                            void* weight_iv_mac,
                            size_t weight_meta_size,
                            void* bias_iv_mac,
                            size_t bias_meta_size,
			    void* model_id,
			    sgx_enclave_id_t *eid,
                            const scale_t& src_scales,
                            const scale_t& weights_scales,
                            const scale_t& dst_scales,
                            const attr_t& attr,
                            const prop_kind aprop_kind,
                            const lowp_kind alowp_kind,
                            const engine& aengine) {
    tensor::desc src_desc, weights_desc, bias_desc;
    attr_t op_attr, src_attr, weights_attr, bias_attr;
    scale_t dst_scales_in;
    data_type dst_data_type;
    auto dst_dims = {src.get_dim(0), weights.get_dim(0)};

    auto weights_scales_in =
        weights.has_scale() ? weights.get_scale() : weights_scales;

    // TODO(xpz): Remove int8 inner product implementation. We are switching to
    // matmul for quantized *mm ops
    if (!weights_scales_in.empty()) {
      IDEEP_ENFORCE(alowp_kind == u8s8 || alowp_kind == s8s8,
                    "Unsupported lowp kind");

      auto src_scales_in =
          src.has_scale() ? src.get_scale()
                          : src_scales.empty() ? IDEEP_DEF_SCALE : src_scales;

      src_desc = {src.get_dims(),
                  alowp_kind == u8s8 ? data_type::u8 : data_type::s8,
                  tag::any};
      if (src.get_data_type() == data_type::f32) {
        src_attr = {0, src_scales_in};
      }

      int scale_size = weights_scales_in.size() > 1 ? weights.get_dim(0) : 1;

      weights_desc = {weights.get_dims(), data_type::s8, tag::any};
      if (weights.get_data_type() == data_type::f32) {
        weights_attr = {utils::tensor_scale_mask(scale_size, false),
                        weights_scales_in};
      }

      // determine dst data type
      if (dst_scales.empty() || dst_scales == IDEEP_DEF_SCALE) {
        dst_data_type = data_type::f32;
      } else if (attr.non_negitive_output()) {
        dst_data_type = data_type::u8;
      } else {
        dst_data_type = data_type::s8;
      }

      // fill primitive attr
      scale_t op_scales(scale_size), bias_scales(scale_size);
      dst_scales_in = dst_scales.empty() || dst_data_type == data_type::f32
                          ? IDEEP_DEF_SCALE
                          : dst_scales;
      for (int i = 0; i < scale_size; i++) {
        bias_scales[i] = src_scales_in[0] * weights_scales_in[i];
        op_scales[i] = dst_scales_in[0] / bias_scales[i];
      }
      op_attr.set_output_scales(utils::op_scale_mask(scale_size), op_scales);

      if (with_bias) {
        bias_desc = {bias.get_dims(), data_type::s32, format_tag::any};
        if (bias.get_data_type() == data_type::f32) {
          bias_attr = {utils::tensor_scale_mask(scale_size, false),
                       bias_scales};
        }
      }
    } else {
      op_attr = attr;
      src_desc = {src.get_dims(), data_type::f32, format_tag::any};
      if (src.has_scale()) {
        auto src_scale = src.get_scale();
        src_scale[0] = 1.f / src_scale[0];
        src_attr = {0, src_scale};
      }

      IDEEP_ENFORCE(utils::one_of(weights.get_data_type(),
                                  data_type::f32, data_type::bf16),
              "Incorrect data type in weights");

      // align weights data type with src
      dst_data_type = src.get_data_type() == data_type::bf16 ? data_type::bf16
                                                             : data_type::f32;
      src_desc = src.get_desc().to_type(dst_data_type);
      weights_desc = weights.get_desc().to_type(dst_data_type);
      // Don't set weight to format any in case DNNL brings in extra reorders.
      // Reordering huge weights during inference might cause performance loss
      if (dst_data_type != data_type::f32) {
        weights_desc = weights_desc.to_format_any();
      }
      if (with_bias) {
        IDEEP_ENFORCE(utils::one_of(bias.get_data_type(),
                                    data_type::f32, data_type::bf16),
                      "Incorrect data type in bias");
        bias_desc = bias.get_desc().to_format_any();
      }
    }

    tensor::desc dst_desc(dst_dims, dst_data_type, format_tag::any);
    auto pd = with_bias
       ? primitive_desc({aprop_kind, src_desc, weights_desc, bias_desc,
                         dst_desc}, op_attr, aengine)
       : primitive_desc({aprop_kind, src_desc, weights_desc, dst_desc},
                        op_attr, aengine);

    auto expected_src = src.reorder_if_differ_in(pd.src_desc(), src_attr);
    auto expected_weights = weights.reorder_if_differ_in(pd.weights_desc(), weights_attr);
    dst.reinit_if_possible(pd.dst_desc());
    if (!dst_scales.empty() && dst.get_data_type() != data_type::f32) {
      dst.set_scale(dst_scales_in);
    }

    if (weight_iv_mac and bias_iv_mac) {
    printf("compute with bias, get weight_meta_size: %d, get bias_meta_size: %d\n", weight_meta_size, bias_meta_size);
    for (int i=0; i<28; i++)
        printf("%X ", *((uint8_t*)weight_iv_mac+i));
    printf("\n");
    for (int i=0; i<28; i++)
        printf("%X ", *((uint8_t*)bias_iv_mac+i));
    printf("\n");
    auto src_desc_query = src_desc.to_format_any();
    auto weights_desc_query = weights_desc.to_format_any();
    auto bias_desc_query = with_bias ? bias_desc.to_format_any() : tensor::desc();
    auto dst_desc_query = dst_desc.to_format_any();

    auto inner_product_desc = dnnl::inner_product_forward::desc(aprop_kind,
            src_desc_query, weights_desc_query,
            bias_desc_query, dst_desc_query);

    void* void_inner_product_desc = (void*)&inner_product_desc;
    size_t inner_product_desc_size = sizeof(inner_product_desc);

    auto src_handle = src.get_data_handle();
    size_t src_data_size = src.get_desc().get_size();

    auto inner_product_pri_desc = dnnl::inner_product_forward::desc(aprop_kind,
            src.get_desc(), weights.get_desc(),
            bias.get_desc(), dst_desc_query);

    void* void_inner_product_pri_desc = (void*)&inner_product_pri_desc;
    size_t inner_product_pri_size = sizeof(inner_product_pri_desc);

    auto weight_handle = weights.get_data_handle();
    size_t weight_data_size = weights.get_desc().get_size();
    auto bias_handle = bias.get_data_handle();
    size_t bias_data_size = bias.get_desc().get_size();

    float *bias_data = (float *)(bias.get_data_handle());
    float bias_float = bias_data[0];

    void* void_dst = (void*)(dst.get_data_handle());
    size_t dst_data_size = dst.get_desc().get_size();

    uint32_t model_id_ = *((uint32_t*)model_id);

    if (eid == NULL)
	    return;

    if (*eid == 0) {
        if (initialize_enclave(eid) < 0) {
            printf("initialize enclave failed... \n");
            return;
        }
        printf("inner product initialize enclave success: eid is %d.\n", *eid);
    }

    sgx_status_t retval;
    sgx_status_t ret = ecall_inner_product_dnnl_function(*eid, &retval, void_inner_product_desc, inner_product_desc_size, src_handle, src_data_size, void_inner_product_pri_desc, inner_product_pri_size, weight_handle, weight_data_size, with_bias?1:0, bias_handle, bias_data_size, void_dst, dst_data_size, weight_iv_mac, weight_meta_size, bias_iv_mac, bias_meta_size, model_id_);
    }//if weight_iv_mac and bias_iv_mac
    else {
    if (with_bias){
      printf("call in mkldnn inner product with bias");
      auto expected_bias = bias.reorder_if_differ_in(pd.bias_desc(), bias_attr);
      super(pd).execute(stream::default_stream(),
                        {{DNNL_ARG_SRC, expected_src},
                         {DNNL_ARG_WEIGHTS, expected_weights},
                         {DNNL_ARG_BIAS, expected_bias},
                         {DNNL_ARG_DST, dst}});
    } else {
      printf("call in mkldnn inner product without bias");
      super(pd).execute(stream::default_stream(),
                        {{DNNL_ARG_SRC, expected_src},
                         {DNNL_ARG_WEIGHTS, expected_weights},
                         {DNNL_ARG_DST, dst}});
    }
    }//if weight_iv_mac and bias_iv_mac else 

    if (attr.non_negitive_output() && dst.get_data_type() == data_type::s8) {
      dst.to_type(data_type::u8);
    }
  }
};


struct inner_product_backward_data : public dnnl::inner_product_backward_data {

  using super = dnnl::inner_product_backward_data;

  static void compute(const tensor& diff_dst,
                      const tensor& weights,
                      const dims& diff_src_dims,
                      tensor& diff_src,
                      const engine& aengine = engine::cpu_engine()) {
    auto weights_ = weights;
    if (diff_dst.get_data_type() == data_type::bf16) {
      weights_.init(weights.get_desc().to_type(data_type::bf16));
      weights_.reorder_from(weights);
    }

    // workaround: diff_src and weights from caffe2 may have different dims.
    // It would be better for caffe2 to do this reshape anyway.
    if (diff_src_dims.size() != weights.ndims()) {
      auto new_dims = diff_src_dims;
      new_dims[0] = weights.get_dim(0);
      weights_.reshape(new_dims);
    }

    auto diff_dst_desc = diff_dst.get_desc().to_format_any();
    auto weights_desc = weights_.get_desc();
    auto diff_src_desc =
        tensor::desc(diff_src_dims, diff_dst.get_data_type(), tag::any);

    auto forward_hints =
        inner_product_forward::primitive_desc(
            {prop_kind::forward, diff_src_desc, weights_desc, diff_dst_desc},
            aengine);

    auto pd = primitive_desc(
        {diff_src_desc, weights_desc, diff_dst_desc}, aengine, forward_hints);

    auto expected_diff_dst = diff_dst.reorder_if_differ_in(pd.diff_dst_desc());
    auto expected_weights = weights_.reorder_if_differ_in(pd.weights_desc());
    diff_src.reinit_if_possible(pd.diff_src_desc());

    super(pd).execute(stream::default_stream(),
                      {{DNNL_ARG_DIFF_DST, expected_diff_dst},
                       {DNNL_ARG_WEIGHTS, expected_weights},
                       {DNNL_ARG_DIFF_SRC, diff_src}});
  }
};

struct inner_product_backward_weights
    : public dnnl::inner_product_backward_weights {

  using super = dnnl::inner_product_backward_weights;

  static void compute(const tensor& src,
                      const tensor& diff_dst,
                      tensor& diff_weights,
                      tensor& diff_bias,
                      const engine& aengine = engine::cpu_engine()) {
    compute_impl</*with_diff_bias=*/true>(
        src, diff_dst, diff_weights, diff_bias);
  }

  static void compute(const tensor& src,
                      const tensor& diff_dst,
                      tensor& diff_weights,
                      const engine& aengine = engine::cpu_engine()) {
    static tensor dummy_diff_bias;
    compute_impl</*with_diff_bias=*/false>(
        src, diff_dst, diff_weights, dummy_diff_bias);
  }

private:
  template<bool with_diff_bias = true>
  static void compute_impl(const tensor& src,
                           const tensor& diff_dst,
                           tensor& diff_weights,
                           tensor& diff_bias,
                           const engine& aengine = engine::cpu_engine()) {
    auto src_desc = src.get_desc().to_format_any();
    auto diff_dst_desc = diff_dst.get_desc().to_format_any();
    auto diff_weights_dims = src.get_dims();
    diff_weights_dims[0] = diff_dst.get_dim(1);
    auto diff_weights_desc =
        tensor::desc(diff_weights_dims, diff_dst.get_data_type(), tag::any);

    // TODO: bf16 diff_bias
    auto diff_bias_desc =
        tensor::desc({diff_dst.get_dim(1)}, data_type::f32, tag::any);

    auto forward_hints = with_diff_bias
        ? inner_product_forward::primitive_desc({prop_kind::forward, src_desc,
            diff_weights_desc, diff_bias_desc, diff_dst_desc}, aengine)
        : inner_product_forward::primitive_desc({prop_kind::forward, src_desc,
            diff_weights_desc, diff_dst_desc}, aengine);
    auto pd = with_diff_bias
        ? primitive_desc({src_desc, diff_weights_desc, diff_bias_desc,
                          diff_dst_desc}, aengine, forward_hints)
        : primitive_desc({src_desc, diff_weights_desc, diff_dst_desc},
                          aengine, forward_hints);

    auto expected_diff_dst = diff_dst.reorder_if_differ_in(pd.diff_dst_desc());
    auto expected_src = src.reorder_if_differ_in(pd.src_desc());
    diff_weights.reinit_if_possible(pd.diff_weights_desc());

    exec_args args {{DNNL_ARG_DIFF_DST, expected_diff_dst},
                    {DNNL_ARG_SRC, expected_src},
                    {DNNL_ARG_DIFF_WEIGHTS ,diff_weights}};

    if (with_diff_bias) {
      diff_bias.reinit_if_possible(pd.diff_bias_desc());
      args.insert({DNNL_ARG_DIFF_BIAS, diff_bias});
    }

    super(pd).execute(stream::default_stream(), args);
  }
};

}  // namespace ideep

#endif
