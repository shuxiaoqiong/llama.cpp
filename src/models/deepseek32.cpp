#include "models.h"

#include "llama-kv-cache.h"
#include "llama-kv-cache-dsa.h"

static size_t deepseek32_hc_elem_offset(const ggml_tensor * t, int64_t i) {
    return ggml_row_size(t->type, i);
}

static ggml_tensor * deepseek32_hc_view_1d(ggml_context * ctx, ggml_tensor * t, int64_t ne0, int64_t i0) {
    return ggml_view_1d(ctx, t, ne0, deepseek32_hc_elem_offset(t, i0));
}

static ggml_tensor * deepseek32_hc_view_2d(
        ggml_context * ctx, ggml_tensor * t, int64_t ne0, int64_t ne1, int64_t i0) {
    return ggml_view_2d(ctx, t, ne0, ne1, t->nb[1], deepseek32_hc_elem_offset(t, i0));
}

static ggml_tensor * deepseek32_hc_mean(ggml_context * ctx, ggml_tensor * x) {
    const int64_t hc = x->ne[1];
    ggml_tensor * acc = ggml_view_2d(ctx, x, x->ne[0], x->ne[2], x->nb[2], 0);
    for (int64_t stream = 1; stream < hc; ++stream) {
        acc = ggml_add(ctx, acc, ggml_view_2d(ctx, x, x->ne[0], x->ne[2], x->nb[2], stream*x->nb[1]));
    }
    return ggml_scale(ctx, acc, 1.0f / hc);
}

static ggml_tensor * deepseek32_hc_pre(
        ggml_context * ctx,
        ggml_tensor * x,
        ggml_tensor * hc_fn,
        ggml_tensor * hc_scale,
        ggml_tensor * hc_base,
        float norm_rms_eps,
        float hc_eps,
        uint32_t sinkhorn_iters,
        int64_t n_embd,
        ggml_tensor ** post,
        ggml_tensor ** comb) {
    const int64_t hc = x->ne[1];
    const int64_t nt = x->ne[2];
    const int64_t hc_dim = hc * n_embd;
    const int64_t hc_mix_dim = (2 + hc) * hc;

    GGML_ASSERT(hc == 4 && hc_fn && hc_scale && hc_base);
    GGML_ASSERT(hc_fn->ne[1] == hc_mix_dim && hc_base->ne[0] == hc_mix_dim && hc_scale->ne[0] == 3);

    ggml_tensor * flat = ggml_reshape_2d(ctx, x, hc_dim, nt);
    ggml_tensor * flat_norm = ggml_rms_norm(ctx, flat, norm_rms_eps);
    ggml_tensor * mixes = ggml_mul_mat(ctx, hc_fn, flat_norm);

    ggml_tensor * scale_pre = deepseek32_hc_view_1d(ctx, hc_scale, 1, 0);
    ggml_tensor * scale_post = deepseek32_hc_view_1d(ctx, hc_scale, 1, 1);
    ggml_tensor * scale_comb = deepseek32_hc_view_1d(ctx, hc_scale, 1, 2);
    ggml_tensor * base_pre = deepseek32_hc_view_2d(ctx, hc_base, hc, 1, 0);
    ggml_tensor * base_post = deepseek32_hc_view_2d(ctx, hc_base, hc, 1, hc);
    ggml_tensor * base_comb = deepseek32_hc_view_2d(ctx, hc_base, hc * hc, 1, 2 * hc);

    ggml_tensor * pre = deepseek32_hc_view_2d(ctx, mixes, hc, nt, 0);
    pre = ggml_add(ctx, ggml_mul(ctx, pre, scale_pre), base_pre);
    pre = ggml_sigmoid(ctx, pre);

    *post = deepseek32_hc_view_2d(ctx, mixes, hc, nt, hc);
    *post = ggml_add(ctx, ggml_mul(ctx, *post, scale_post), base_post);
    *post = ggml_scale(ctx, ggml_sigmoid(ctx, *post), 2.0f);

    *comb = deepseek32_hc_view_2d(ctx, mixes, hc * hc, nt, 2 * hc);
    *comb = ggml_add(ctx, ggml_mul(ctx, *comb, scale_comb), base_comb);
    *comb = ggml_clamp(ctx, *comb, -30.0f, 30.0f);
    *comb = ggml_reshape_3d(ctx, *comb, hc, hc, nt);
    *comb = ggml_soft_max(ctx, *comb);

    ggml_tensor * eps = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    eps = ggml_fill(ctx, eps, hc_eps);
    auto norm_dst = [&]() {
        ggml_tensor * sum = ggml_cont(ctx, ggml_permute(ctx, *comb, 1, 0, 2, 3));
        sum = ggml_add(ctx, ggml_sum_rows(ctx, sum), eps);
        *comb = ggml_div(ctx, *comb, ggml_permute(ctx, sum, 1, 0, 2, 3));
    };
    auto norm_src = [&]() {
        ggml_tensor * sum = ggml_add(ctx, ggml_sum_rows(ctx, *comb), eps);
        *comb = ggml_div(ctx, *comb, sum);
    };
    norm_dst();
    for (uint32_t iter = 1; iter < sinkhorn_iters; ++iter) {
        norm_src();
        norm_dst();
    }

    ggml_tensor * result = nullptr;
    for (int64_t stream = 0; stream < hc; ++stream) {
        ggml_tensor * x_stream = ggml_view_2d(ctx, x, n_embd, nt, x->nb[2], stream*x->nb[1]);
        ggml_tensor * gate = ggml_view_2d(ctx, pre, 1, nt, pre->nb[1], stream*pre->nb[0]);
        ggml_tensor * term = ggml_mul(ctx, x_stream, gate);
        result = result ? ggml_add(ctx, result, term) : term;
    }
    return result;
}

static ggml_tensor * deepseek32_hc_post(
        ggml_context * ctx,
        ggml_tensor * x,
        ggml_tensor * residual,
        ggml_tensor * post,
        ggml_tensor * comb,
        int64_t n_embd) {
    const int64_t hc = residual->ne[1];
    const int64_t nt = x->ne[1];
    ggml_tensor * out = nullptr;
    for (int64_t dst = 0; dst < hc; ++dst) {
        ggml_tensor * post_dst = ggml_view_2d(ctx, post, 1, nt, post->nb[1], dst*post->nb[0]);
        ggml_tensor * cur = ggml_mul(ctx, x, post_dst);
        for (int64_t src = 0; src < hc; ++src) {
            ggml_tensor * res_src = ggml_view_2d(ctx, residual, n_embd, nt, residual->nb[2], src*residual->nb[1]);
            ggml_tensor * comb_src_dst = ggml_view_2d(ctx, comb, 1, nt, comb->nb[2], src*comb->nb[0] + dst*comb->nb[1]);
            cur = ggml_add(ctx, cur, ggml_mul(ctx, res_src, comb_src_dst));
        }
        cur = ggml_reshape_3d(ctx, cur, n_embd, 1, nt);
        out = out ? ggml_concat(ctx, out, cur, 1) : cur;
    }
    return out;
}

void llama_model_deepseek32::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,    hparams.f_norm_rms_eps);
    hparams.f_norm_eps = 1e-6;  // eps for layer norm
    ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS, hparams.rope_sections, 4, false);

    // MoE parameters
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,         hparams.n_expert_shared);
    ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,   hparams.n_layer_dense_lead, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,         hparams.expert_weights_norm, false);

    // deepseek MLA parameters
    ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK,      hparams.n_lora_q);
    ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK,     hparams.n_lora_kv);
    ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_MLA,   hparams.n_embd_head_k_mla_impl, false);
    ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_MLA, hparams.n_embd_head_v_mla_impl, false);
    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,        hparams.n_expert_shared);

    // DSA parameters
    ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k);

    // Optional mHC metadata keeps existing DeepSeek32 GGUFs on the single-stream path.
    ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT, hparams.dsv4_hc_mult, false);
    if (hparams.dsv4_hc_mult > 0) {
        ml.get_key(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, hparams.dsv4_hc_sinkhorn_iters);
        if (!ml.get_key(LLM_KV_HYPER_CONNECTION_EPSILON, hparams.dsv4_hc_eps, false)) {
            hparams.dsv4_hc_eps = 1e-6f;
        }
        if (hparams.dsv4_hc_mult != 4 || hparams.dsv4_hc_sinkhorn_iters == 0) {
            throw std::runtime_error("DEEPSEEK32 mHC requires 4 residual streams and non-zero Sinkhorn iterations");
        }
    }

    // Expert gating function
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC, hparams.expert_gating_func);

    if (ml.get_key(LLM_KV_ROPE_SCALING_YARN_LOG_MUL, hparams.rope_yarn_log_mul, 0.0f)) {
        // [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]
        // cancel the factor from the convert script
        hparams.rope_yarn_log_mul /= 0.1f;
    }

    switch (hparams.n_layer()) {
        case 61: type = LLM_TYPE_685B_A37B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_deepseek32::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const auto nf = nextn_flags(ml);
    const int trunk_flags = nf.trunk;
    const int mtp_flags   = nf.mtp;

    const bool is_mla = hparams.is_mla();
    if (!is_mla) {
        throw std::runtime_error("DEEPSEEK32 architecture requires MLA");
    }

    // note: these are the actual head sizes you get when treating as MHA or after "decompression" using wv_b for MLA
    const int64_t n_embd_head_k_mla = hparams.n_embd_head_k_mla();
    const int64_t n_embd_head_v_mla = hparams.n_embd_head_v_mla();

    const int64_t n_embd_head_qk_rope = hparams.n_rot();
    const int64_t n_embd_head_qk_nope = n_embd_head_k_mla - n_embd_head_qk_rope;

    const int64_t q_lora_rank  = hparams.n_lora_q;
    const int64_t kv_lora_rank = hparams.n_lora_kv;

    const int64_t n_ff_exp        = hparams.n_ff_exp();
    const int64_t n_expert_shared = hparams.n_expert_shared;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    // try to load output.weight, if not found, use token_embd (tied embeddings)
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    if (!output) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    for (int i = 0; i < n_layer_all; ++i) {
        const int flags = (i >= n_layer) ? mtp_flags : trunk_flags;

        auto & layer = layers[i];

        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, flags);
        layer.attn_q_a_norm  = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM, "weight", i), {q_lora_rank}, flags);
        layer.attn_kv_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_NORM, "weight", i), {kv_lora_rank}, flags);

        layer.wq_a = create_tensor(tn(LLM_TENSOR_ATTN_Q_A, "weight", i), {n_embd, q_lora_rank}, flags);
        layer.wq_b = create_tensor(tn(LLM_TENSOR_ATTN_Q_B, "weight", i), {q_lora_rank, n_head * n_embd_head_k_mla}, flags);

        layer.wkv_a_mqa = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_MQA, "weight", i), {n_embd, kv_lora_rank + n_embd_head_qk_rope}, flags);

        // note: only old legacy GGUF files will have the unsplit wkv_b tensor in
        layer.wk_b = create_tensor(tn(LLM_TENSOR_ATTN_K_B, "weight", i), {n_embd_head_qk_nope, kv_lora_rank, n_head}, flags);
        layer.wv_b = create_tensor(tn(LLM_TENSOR_ATTN_V_B, "weight", i), {kv_lora_rank, n_embd_head_v_mla, n_head}, flags);

        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_head * n_embd_head_v_mla, n_embd}, flags);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, flags);

        // DSA indexer
        layer.indexer_k_norm   = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM,   "weight", i), {hparams.indexer_head_size}, flags);
        layer.indexer_k_norm_b = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM,   "bias",   i), {hparams.indexer_head_size}, flags);
        layer.indexer_proj     = create_tensor(tn(LLM_TENSOR_INDEXER_PROJ,     "weight", i), {n_embd, hparams.indexer_n_head}, flags);
        layer.indexer_attn_k   = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_K,   "weight", i), {n_embd, hparams.indexer_head_size}, flags);
        layer.indexer_attn_q_b = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_Q_B, "weight", i), {q_lora_rank, hparams.indexer_n_head * hparams.indexer_head_size}, flags);

        if (hparams.dsv4_hc_mult > 0) {
            const int64_t hc_dim = hparams.dsv4_hc_mult * n_embd;
            const int64_t hc_mix_dim = (2 + hparams.dsv4_hc_mult) * hparams.dsv4_hc_mult;
            layer.hc_attn_fn = create_tensor(tn(LLM_TENSOR_HC_ATTN_FN, "weight", i), {hc_dim, hc_mix_dim}, flags);
            layer.hc_attn_base = create_tensor(tn(LLM_TENSOR_HC_ATTN_BASE, "weight", i), {hc_mix_dim}, flags);
            layer.hc_attn_scale = create_tensor(tn(LLM_TENSOR_HC_ATTN_SCALE, "weight", i), {3}, flags);
            layer.hc_ffn_fn = create_tensor(tn(LLM_TENSOR_HC_FFN_FN, "weight", i), {hc_dim, hc_mix_dim}, flags);
            layer.hc_ffn_base = create_tensor(tn(LLM_TENSOR_HC_FFN_BASE, "weight", i), {hc_mix_dim}, flags);
            layer.hc_ffn_scale = create_tensor(tn(LLM_TENSOR_HC_FFN_SCALE, "weight", i), {3}, flags);
        }
        if (i < (int) hparams.n_layer_dense_lead) {
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, flags);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, flags);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, flags);
        } else {
            layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, flags);
            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), {n_expert}, TENSOR_NOT_REQUIRED);

            if (n_expert == 0) {
                throw std::runtime_error("n_expert must be > 0");
            }
            if (n_expert_used == 0) {
                throw std::runtime_error("n_expert_used must be > 0");
            }

            // MoE branch
            layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {  n_embd, n_ff_exp, n_expert}, flags);
            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp,   n_embd, n_expert}, flags);
            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {  n_embd, n_ff_exp, n_expert}, flags);

            // Shared expert branch
            layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, n_ff_exp * n_expert_shared}, flags);
            layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {        n_ff_exp * n_expert_shared, n_embd}, flags);
            layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, n_ff_exp * n_expert_shared}, flags);
        }

        // NextN/MTP tensors - conditionally load for last nextn_predict_layers
        if (i >= n_layer) {
            layer.nextn.eh_proj          = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ, "weight", i), { 2 * n_embd, n_embd }, flags);
            layer.nextn.enorm            = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM, "weight", i), { n_embd }, flags);
            layer.nextn.hnorm            = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM, "weight", i), { n_embd }, flags);

            // Optional tensors
            layer.nextn.embed_tokens     = create_tensor(tn(LLM_TENSOR_NEXTN_EMBED_TOKENS, "weight", i), { n_embd, n_vocab }, flags | TENSOR_NOT_REQUIRED);
            layer.nextn.shared_head_head = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", i), { n_embd, n_vocab }, flags | TENSOR_NOT_REQUIRED);
            layer.nextn.shared_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", i), { n_embd }, flags | TENSOR_NOT_REQUIRED);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_deepseek32::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(*this, params);
    }
    return std::make_unique<graph>(*this, params);
}

llama_model_deepseek32::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    const bool is_mla = hparams.is_mla();
    GGML_ASSERT(is_mla);

    // note: these are the actual head sizes you get when treating as MHA or after "decompression" using wv_b for MLA
    const int64_t n_embd_head_k = hparams.n_embd_head_k_mla();
    const int64_t n_embd_head_v = hparams.n_embd_head_v_mla();
    GGML_UNUSED(n_embd_head_v);

    const int64_t n_embd_head_qk_rope = hparams.n_rot();
    const int64_t n_embd_head_qk_nope = n_embd_head_k - n_embd_head_qk_rope;

    const int64_t n_indexer_head = hparams.indexer_n_head;
    const int64_t n_embd_indexer_head = hparams.indexer_head_size;
    const uint32_t n_indexer_top_k = hparams.indexer_top_k;

    // the indexer head layous is [rope | nope]
    GGML_ASSERT(hparams.n_rot() <= n_embd_indexer_head);

    const uint32_t kv_lora_rank = hparams.n_lora_kv;

    // We have to pre-scale kq_scale and attn_factor to make the YaRN RoPE work correctly.
    // See https://github.com/ggml-org/llama.cpp/discussions/7416 for detailed explanation.
    // And also: https://github.com/ggml-org/llama.cpp/pull/17945 [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]

    // first cancel the adjustment from llama_hparams::yarn_attn_factor_adjust to get the original attn_factor
    GGML_ASSERT(ext_factor >= 0.0f);
    const float attn_factor_org = attn_factor * (1.0f + 0.1f * logf(1.0f / freq_scale));

    // use the original attn_factor to pre-scale the kq_scale
    const float mscale   = attn_factor_org * (1.0f + 0.1f * hparams.rope_yarn_log_mul * logf(1.0f / freq_scale));
    const float kq_scale = 1.0f * mscale * mscale / sqrtf(float(n_embd_head_k));

    ggml_tensor * cur;
    ggml_tensor * inpL;

    const bool use_hc = hparams.dsv4_hc_mult > 0;
    const int64_t hc = hparams.dsv4_hc_mult;

    // {n_embd, n_tokens}
    inpL = build_inp_embd(model.tok_embd);
    if (use_hc) {
        inpL = ggml_reshape_3d(ctx0, inpL, n_embd, 1, n_tokens);
        inpL = ggml_repeat_4d(ctx0, inpL, n_embd, hc, n_tokens, 1);
    }

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    llm_graph_input_attn_k_dsa * inp_attn_dsa = build_attn_inp_k_dsa();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * residual = inpL;
        ggml_tensor * post = nullptr;
        ggml_tensor * comb = nullptr;

        if (use_hc) {
            cur = deepseek32_hc_pre(ctx0, inpL,
                    model.layers[il].hc_attn_fn, model.layers[il].hc_attn_scale, model.layers[il].hc_attn_base,
                    norm_rms_eps, hparams.dsv4_hc_eps, hparams.dsv4_hc_sinkhorn_iters,
                    n_embd, &post, &comb);
        } else {
            cur = inpL;
        }
        cur = build_norm(cur, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self_attention
        {
            ggml_tensor * qr = ggml_mul_mat(ctx0, model.layers[il].wq_a, cur);
            cb(qr, "qr", il);

            qr = build_norm(qr, model.layers[il].attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
            cb(qr, "qr", il);

            ggml_tensor * top_k = nullptr;

            // lightning indexer
            {
                ggml_tensor * indexer_q = ggml_mul_mat(ctx0, model.layers[il].indexer_attn_q_b, qr);
                cb(indexer_q, "indexer_q", il);

                // {n_embd_indexer_head, n_indexer_head, n_tokens}
                indexer_q = ggml_reshape_3d(ctx0, indexer_q, n_embd_indexer_head, n_indexer_head, n_tokens);
                indexer_q = ggml_rope_ext(ctx0, indexer_q, inp_pos, nullptr, n_rot,
                                     LLAMA_ROPE_TYPE_NEOX, n_ctx_orig, freq_base, freq_scale,
                                     ext_factor, attn_factor, beta_fast, beta_slow);
                cb(indexer_q, "indexer_q", il);

                ggml_tensor * indexer_k = ggml_mul_mat(ctx0, model.layers[il].indexer_attn_k, cur);
                cb(indexer_k, "indexer_k", il);

                indexer_k = build_norm(indexer_k, model.layers[il].indexer_k_norm, model.layers[il].indexer_k_norm_b, LLM_NORM, il);
                cb(indexer_k, "indexer_k", il);

                // {n_embd_indexer_head, 1, n_tokens}
                indexer_k = ggml_reshape_3d(ctx0, indexer_k, n_embd_indexer_head, 1, n_tokens);
                indexer_k = ggml_rope_ext(ctx0, indexer_k, inp_pos, nullptr, n_rot,
                                     LLAMA_ROPE_TYPE_NEOX, n_ctx_orig, freq_base, freq_scale,
                                     ext_factor, attn_factor, beta_fast, beta_slow);
                cb(indexer_k, "indexer_k", il);

                // perform Hadamard transform on indexer q and k
                indexer_q = ggml_mul_mat(ctx0, inp_attn_dsa->self_k_rot_lid, indexer_q);
                cb(indexer_q, "indexer_q", il);
                indexer_k = ggml_mul_mat(ctx0, inp_attn_dsa->self_k_rot_lid, indexer_k);
                cb(indexer_k, "indexer_k", il);

                // store indexer keys to KV cache
                const auto * mctx_lid = inp_attn_dsa->mctx->get_lid();
                const auto & k_idxs_lid = inp_attn_dsa->get_k_idxs_lid();
                ggml_build_forward_expand(gf, mctx_lid->cpy_k(ctx0, indexer_k, k_idxs_lid, il));

                // prepare indexer weights
                ggml_tensor * indexer_weights = ggml_mul_mat(ctx0, model.layers[il].indexer_proj, cur);
                cb(indexer_weights, "indexer_weights", il);

                // get cached indexer keys
                indexer_k = mctx_lid->get_k(ctx0, il);

                // split the batch into streams if needed
                const auto n_stream = indexer_k->ne[3];
                indexer_q = ggml_view_4d(ctx0, indexer_q, indexer_q->ne[0], indexer_q->ne[1], indexer_q->ne[2]/n_stream, n_stream, indexer_q->nb[1], indexer_q->nb[2], indexer_q->nb[3]/n_stream, 0);
                indexer_weights = ggml_view_4d(ctx0, indexer_weights, indexer_weights->ne[0], indexer_weights->ne[1]/n_stream, indexer_weights->ne[2], n_stream, indexer_weights->nb[1], indexer_weights->nb[2]/n_stream, indexer_weights->nb[3]/n_stream, 0);

                // pre-scale weights to avoid scaling operations on huge indexer_score tensor
                indexer_weights = ggml_scale(ctx0, indexer_weights, 1.0f / sqrtf(float(n_embd_indexer_head * n_indexer_head)));
                cb(indexer_weights, "indexer_weights", il);

                ggml_tensor * indexer_score = nullptr;
                if (cparams.fused_lid) {
                    indexer_score = ggml_lightning_indexer(ctx0, indexer_q, indexer_k, indexer_weights, inp_attn_dsa->get_kq_mask_lid());
                    cb(indexer_score, "indexer_score", il);
                    res->add_fused_node({LLM_FUSED_OP_LIGHTNING_INDEXER, indexer_score, il});
                } else {
                    // calculate indexer kq
                    indexer_q = ggml_permute(ctx0, indexer_q, 0, 2, 1, 3);
                    cb(indexer_q, "indexer_q", il);
                    indexer_k = ggml_permute(ctx0, indexer_k, 0, 2, 1, 3);
                    cb(indexer_k, "indexer_k", il);

                    ggml_tensor * indexer_kq = ggml_mul_mat(ctx0, indexer_k, indexer_q);
                    cb(indexer_kq, "indexer_kq", il);

                    // ReLU requires contiguous tensors
                    indexer_kq = ggml_cont(ctx0, ggml_permute(ctx0, indexer_kq, 2, 1, 0, 3));
                    cb(indexer_kq, "indexer_kq", il);

                    // apply ReLU
                    indexer_score = ggml_relu(ctx0, indexer_kq);
                    cb(indexer_score, "indexer_score", il);

                    // multiply scores by indexer weights
                    indexer_score = ggml_mul(ctx0, indexer_score, indexer_weights);
                    cb(indexer_score, "indexer_score", il);

                    // sum by q n_indexer_head dimension
                    indexer_score = ggml_sum_rows(ctx0, indexer_score);
                    cb(indexer_score, "indexer_score", il);

                    // permute result to match KQ mask
                    indexer_score = ggml_cont(ctx0, ggml_permute(ctx0, indexer_score, 2, 1, 0, 3));
                    cb(indexer_score, "indexer_score", il);

                    // mask indexer scores
                    ggml_tensor * indexer_kq_mask = inp_attn_dsa->get_kq_mask_lid();
                    indexer_score = ggml_add(ctx0, indexer_score, indexer_kq_mask);
                    cb(indexer_score, "indexer_score", il);
                }

                // get indices of top k indexer scores
                uint32_t n_top_k = indexer_score->ne[0] < n_indexer_top_k ? indexer_score->ne[0] : n_indexer_top_k;
                top_k = ggml_cont(ctx0, ggml_top_k(ctx0, indexer_score, n_top_k));
                cb(top_k, "top_k", il);
            }

            ggml_tensor * q = ggml_mul_mat(ctx0, model.layers[il].wq_b, qr);
            cb(q, "q", il);

            // split into {n_embd_head_qk_nope, n_head, n_tokens}
            ggml_tensor * q_nope =
                ggml_view_3d(ctx0, q, n_embd_head_qk_nope, n_head, n_tokens, ggml_row_size(q->type, n_embd_head_k),
                             ggml_row_size(q->type, n_embd_head_k) * n_head, 0);
            cb(q_nope, "q_nope", il);

            // and {n_embd_head_qk_rope, n_head, n_tokens}
            ggml_tensor * q_pe = ggml_view_3d(
                ctx0, q, n_embd_head_qk_rope, n_head, n_tokens, ggml_row_size(q->type, n_embd_head_k),
                ggml_row_size(q->type, n_embd_head_k) * n_head, ggml_row_size(q->type, n_embd_head_qk_nope));
            cb(q_pe, "q_pe", il);

            ggml_tensor * kv_cmpr_pe = ggml_mul_mat(ctx0, model.layers[il].wkv_a_mqa, cur);
            cb(kv_cmpr_pe, "kv_cmpr_pe", il);

            // split into {kv_lora_rank, n_tokens}
            ggml_tensor * kv_cmpr =
                ggml_view_2d(ctx0, kv_cmpr_pe, kv_lora_rank, n_tokens,
                             ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope), 0);
            cb(kv_cmpr, "kv_cmpr", il);

            // and {n_embd_head_qk_rope, 1, n_tokens}
            ggml_tensor * k_pe = ggml_view_3d(ctx0, kv_cmpr_pe, n_embd_head_qk_rope, 1, n_tokens,
                                              ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                                              ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                                              ggml_row_size(kv_cmpr_pe->type, kv_lora_rank));
            cb(k_pe, "k_pe", il);

            q_pe = ggml_rope_ext(ctx0, q_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                 ext_factor, attn_factor, beta_fast, beta_slow);
            cb(q_pe, "q_pe", il);

            k_pe = ggml_rope_ext(ctx0, k_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                 ext_factor, attn_factor, beta_fast, beta_slow);
            cb(k_pe, "k_pe", il);

            kv_cmpr = build_norm(kv_cmpr, model.layers[il].attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
            cb(kv_cmpr, "kv_cmpr", il);

            // MLA attention
            {
                // {n_embd_head_qk_nope, n_tokens, n_head}
                q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);
                cb(q_nope, "q_nope_perm", il);

                // {n_embd_head_qk_nope, kv_lora_rank, n_head} x {n_embd_head_qk_nope, n_tokens, n_head}
                ggml_tensor * q_nope_absorbed = ggml_mul_mat(ctx0, model.layers[il].wk_b, q_nope);
                cb(q_nope_absorbed, "q_nope_absorbed", il);

                // {kv_lora_rank, n_head, n_tokens}
                q_nope_absorbed = ggml_permute(ctx0, q_nope_absorbed, 0, 2, 1, 3);
                cb(q_nope_absorbed, "q_nope_absorbed_perm", il);

                // {n_embd_head_qk_rope + kv_lora_rank, n_head, n_tokens}
                // note: rope must go first for in-place context shifting in build_rope_shift()
                ggml_tensor * Qcur = ggml_concat(ctx0, q_nope_absorbed, q_pe, 0);
                cb(Qcur, "Qcur", il);

                kv_cmpr = ggml_reshape_3d(ctx0, kv_cmpr, kv_lora_rank, 1, n_tokens);
                cb(kv_cmpr, "kv_cmpr_reshape", il);

                // {n_embd_head_qk_rope + kv_lora_rank, 1, n_tokens}
                ggml_tensor * Kcur = ggml_concat(ctx0, kv_cmpr, k_pe, 0);
                cb(Kcur, "Kcur", il);

                // {kv_lora_rank, 1, n_tokens}
                ggml_tensor * Vcur = kv_cmpr;
                cb(Vcur, "Vcur", il);

                // note: MLA with the absorption optimization converts into MQA (ie: GQA with 1 group)
                cur = build_attn(inp_attn_dsa,
                        model.layers[il].wo, NULL, model.layers[il].wo_s,
                        Qcur, Kcur, Vcur, nullptr, nullptr, model.layers[il].wv_b, top_k, kq_scale, il);
            }
        }
        // when unmasked nextn embeddings are requested, t_h_nextn must keep all rows,
        // so the early output masking has to be skipped (it is applied after the final norm instead)
        if (il == n_layer - 1 && crop_before_nextn(inp_out_ids)) {
            cur = ggml_get_rows(ctx0, cur, inp_out_ids);
            if (use_hc) {
                ggml_tensor * residual_flat = ggml_reshape_2d(ctx0, residual, n_embd * hc, n_tokens);
                residual_flat = ggml_get_rows(ctx0, residual_flat, inp_out_ids);
                residual = ggml_reshape_3d(ctx0, residual_flat, n_embd, hc, n_outputs);
                post = ggml_get_rows(ctx0, post, inp_out_ids);
                ggml_tensor * comb_flat = ggml_reshape_2d(ctx0, comb, hc * hc, n_tokens);
                comb_flat = ggml_get_rows(ctx0, comb_flat, inp_out_ids);
                comb = ggml_reshape_3d(ctx0, comb_flat, hc, hc, n_outputs);
            } else {
                residual = ggml_get_rows(ctx0, residual, inp_out_ids);
            }
        }
        if (use_hc) {
            inpL = deepseek32_hc_post(ctx0, cur, residual, post, comb, n_embd);
            residual = inpL;
            cur = deepseek32_hc_pre(ctx0, inpL,
                    model.layers[il].hc_ffn_fn, model.layers[il].hc_ffn_scale, model.layers[il].hc_ffn_base,
                    norm_rms_eps, hparams.dsv4_hc_eps, hparams.dsv4_hc_sinkhorn_iters,
                    n_embd, &post, &comb);
        } else {
            cur = ggml_add(ctx0, cur, residual);
        }
        ggml_tensor * ffn_inp = cur;
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            cur = build_ffn(cur,
                model.layers[il].ffn_up, NULL, model.layers[il].ffn_up_s,
                model.layers[il].ffn_gate, NULL, model.layers[il].ffn_gate_s,
                model.layers[il].ffn_down, NULL, model.layers[il].ffn_down_s,
                NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        } else {
            // MoE branch
            ggml_tensor * moe_out = build_moe_ffn(cur,
                model.layers[il].ffn_gate_inp,
                model.layers[il].ffn_up_exps,
                model.layers[il].ffn_gate_exps,
                model.layers[il].ffn_down_exps,
                model.layers[il].ffn_exp_probs_b,
                n_expert, n_expert_used,
                LLM_FFN_SILU, hparams.expert_weights_norm,
                hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il,
                nullptr,
                model.layers[il].ffn_gate_up_exps,
                model.layers[il].ffn_up_exps_s,
                model.layers[il].ffn_gate_exps_s,
                model.layers[il].ffn_down_exps_s);
            cb(moe_out, "ffn_moe_out", il);

            // FFN shared expert
            {
                ggml_tensor * ffn_shexp =
                    build_ffn(cur,
                        model.layers[il].ffn_up_shexp, NULL, model.layers[il].ffn_up_shexp_s,
                        model.layers[il].ffn_gate_shexp, NULL, model.layers[il].ffn_gate_shexp_s,
                        model.layers[il].ffn_down_shexp, NULL, model.layers[il].ffn_down_shexp_s,
                        NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(ffn_shexp, "ffn_shexp", il);

                cur = ggml_add(ctx0, moe_out, ffn_shexp);
                cb(cur, "ffn_out", il);
            }
        }
        if (use_hc) {
            inpL = deepseek32_hc_post(ctx0, cur, residual, post, comb, n_embd);
            inpL = build_cvec(inpL, il);
        } else {
            cur = ggml_add(ctx0, cur, ffn_inp);
            inpL = build_cvec(cur, il);
        }
        cb(inpL, "l_out", il);

        // input for next layer
        cur = inpL;
    }
    cur = inpL;

    if (use_hc) {
        cur = deepseek32_hc_mean(ctx0, cur);
    }

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);

    // post-norm hidden state feeds the NextN/MTP draft head
    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    if (crop_after_nextn(inp_out_ids)) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = ggml_mul_mat(ctx0, model.output, cur);

    cb(cur, "result_output", -1);
    res->t_logits = cur;
}

// LLM_GRAPH_TYPE_DECODER_MTP draft head for DeepSeek V3.2 (DEEPSEEK32).
// Semantics mirror the deepseek-family NextN/MTP layer:
//   enorm(embed) + hnorm(prev_hidden) -> concat(e, h) -> eh_proj ->
//   full deepseek32 decoder block (dense MLA attention + sigmoid-gated MoE FFN
//   with shared expert, exactly as the trunk deepseek2 graph builds it) ->
//   shared_head_norm (fallback output_norm) -> shared LM head.
// The DSA indexer is not used at runtime.
llama_model_deepseek32::graph_mtp::graph_mtp(const llama_model & model, const llm_graph_params & params)
    : llm_graph_context(params) {
    GGML_ASSERT(hparams.n_layer_nextn > 0 && "DEEPSEEK32 MTP requires n_layer_nextn > 0");
    GGML_ASSERT(hparams.n_layer_nextn == 1 && "DEEPSEEK32 MTP currently only supports a single MTP block");
    GGML_ASSERT(hparams.is_mla() && "DEEPSEEK32 MTP requires MLA");

    const int il = hparams.n_layer() + cparams.nextn_layer_offset;
    GGML_ASSERT(cparams.nextn_layer_offset >= 0 &&
                cparams.nextn_layer_offset < (int) hparams.n_layer_nextn &&
                "nextn_layer_offset out of range [0, n_layer_nextn)");
    const auto & layer = model.layers[il];

    GGML_ASSERT(layer.nextn.eh_proj && "MTP block missing nextn.eh_proj");
    GGML_ASSERT(layer.nextn.enorm   && "MTP block missing nextn.enorm");
    GGML_ASSERT(layer.nextn.hnorm   && "MTP block missing nextn.hnorm");
    GGML_ASSERT(layer.ffn_gate_inp  && "MTP block missing ffn_gate_inp");

    // note: these are the actual head sizes you get when treating as MHA or after "decompression" using wv_b for MLA
    const int64_t n_embd_head_k = hparams.n_embd_head_k_mla();

    const int64_t n_embd_head_qk_rope = hparams.n_rot();
    const int64_t n_embd_head_qk_nope = n_embd_head_k - n_embd_head_qk_rope;

    const uint32_t kv_lora_rank = hparams.n_lora_kv;

    // We have to pre-scale kq_scale and attn_factor to make the YaRN RoPE work correctly.
    // See the deepseek2 trunk graph for the detailed explanation - this must match it EXACTLY.
    GGML_ASSERT(ext_factor >= 0.0f);
    const float attn_factor_org = attn_factor * (1.0f + 0.1f * logf(1.0f / freq_scale));

    const float mscale   = attn_factor_org * (1.0f + 0.1f * hparams.rope_yarn_log_mul * logf(1.0f / freq_scale));
    const float kq_scale = 1.0f * mscale * mscale / sqrtf(float(n_embd_head_k));

    // TODO: extract in a common llm_graph_context::build_inp_embd_h()
    auto inp = std::make_unique<llm_graph_input_embd_h>(hparams.n_embd_inp(), hparams.n_embd);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_inp(), n_tokens);
    ggml_set_input(inp->embd);

    ggml_tensor * tok_embd;
    ASSERT_EMBD_OR_TOKEN(ubatch);
    if (ubatch.token) {
        ggml_tensor * tok_embd_w = layer.nextn.embed_tokens ? layer.nextn.embed_tokens : model.tok_embd;

        tok_embd = ggml_get_rows(ctx0, tok_embd_w, inp->tokens);
    } else {
        tok_embd = inp->embd;
    }
    cb(tok_embd, "mtp_tok_embd", il);

    inp->h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd, n_tokens);
    ggml_set_input(inp->h);
    ggml_set_name(inp->h, "mtp_h_input");

    ggml_tensor * h_embd = inp->h;

    res->add_input(std::move(inp));

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // MLA with the absorption optimization uses a K-only cache (V is a view of K)
    auto * inp_attn = build_attn_inp_k();

    ggml_tensor * h_norm = build_norm(h_embd, layer.nextn.hnorm, nullptr, LLM_NORM_RMS, il);
    cb(h_norm, "mtp_hnorm", il);

    ggml_tensor * e_norm = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
    cb(e_norm, "mtp_enorm", il);

    ggml_tensor * concat = ggml_concat(ctx0, e_norm, h_norm, /*dim=*/ 0);
    cb(concat, "mtp_concat", il);

    ggml_tensor * cur = build_lora_mm(layer.nextn.eh_proj, concat, layer.nextn.eh_proj_s);
    cb(cur, "mtp_eh_proj", il);

    const bool use_hc = hparams.dsv4_hc_mult > 0;
    const int64_t hc = hparams.dsv4_hc_mult;
    ggml_tensor * inpL = cur;
    if (use_hc) {
        inpL = ggml_reshape_3d(ctx0, inpL, n_embd, 1, n_tokens);
        inpL = ggml_repeat_4d(ctx0, inpL, n_embd, hc, n_tokens, 1);
    }
    ggml_tensor * residual = inpL;
    ggml_tensor * post = nullptr;
    ggml_tensor * comb = nullptr;
    if (use_hc) {
        cur = deepseek32_hc_pre(ctx0, inpL,
                layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base,
                norm_rms_eps, hparams.dsv4_hc_eps, hparams.dsv4_hc_sinkhorn_iters,
                n_embd, &post, &comb);
    }

    cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_attn_norm", il);

    // self-attention: dense MLA, same construction as the deepseek2 trunk graph
    {
        ggml_tensor * q = ggml_mul_mat(ctx0, layer.wq_a, cur);
        cb(q, "mtp_q", il);

        q = build_norm(q, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
        cb(q, "mtp_q", il);

        q = ggml_mul_mat(ctx0, layer.wq_b, q);
        cb(q, "mtp_q", il);

        // split into {n_embd_head_qk_nope, n_head, n_tokens}
        ggml_tensor * q_nope =
            ggml_view_3d(ctx0, q, n_embd_head_qk_nope, n_head, n_tokens, ggml_row_size(q->type, n_embd_head_k),
                         ggml_row_size(q->type, n_embd_head_k) * n_head, 0);
        cb(q_nope, "mtp_q_nope", il);

        // and {n_embd_head_qk_rope, n_head, n_tokens}
        ggml_tensor * q_pe = ggml_view_3d(
            ctx0, q, n_embd_head_qk_rope, n_head, n_tokens, ggml_row_size(q->type, n_embd_head_k),
            ggml_row_size(q->type, n_embd_head_k) * n_head, ggml_row_size(q->type, n_embd_head_qk_nope));
        cb(q_pe, "mtp_q_pe", il);

        ggml_tensor * kv_cmpr_pe = ggml_mul_mat(ctx0, layer.wkv_a_mqa, cur);
        cb(kv_cmpr_pe, "mtp_kv_cmpr_pe", il);

        // split into {kv_lora_rank, n_tokens}
        ggml_tensor * kv_cmpr =
            ggml_view_2d(ctx0, kv_cmpr_pe, kv_lora_rank, n_tokens,
                         ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope), 0);
        cb(kv_cmpr, "mtp_kv_cmpr", il);

        // and {n_embd_head_qk_rope, 1, n_tokens}
        ggml_tensor * k_pe = ggml_view_3d(ctx0, kv_cmpr_pe, n_embd_head_qk_rope, 1, n_tokens,
                                          ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                                          ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                                          ggml_row_size(kv_cmpr_pe->type, kv_lora_rank));
        cb(k_pe, "mtp_k_pe", il);

        q_pe = ggml_rope_ext(ctx0, q_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                             ext_factor, attn_factor, beta_fast, beta_slow);
        cb(q_pe, "mtp_q_pe", il);

        k_pe = ggml_rope_ext(ctx0, k_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                             ext_factor, attn_factor, beta_fast, beta_slow);
        cb(k_pe, "mtp_k_pe", il);

        kv_cmpr = build_norm(kv_cmpr, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
        cb(kv_cmpr, "mtp_kv_cmpr", il);

        // {n_embd_head_qk_nope, n_tokens, n_head}
        q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);
        cb(q_nope, "mtp_q_nope_perm", il);

        // {n_embd_head_qk_nope, kv_lora_rank, n_head} x {n_embd_head_qk_nope, n_tokens, n_head}
        ggml_tensor * q_nope_absorbed = ggml_mul_mat(ctx0, layer.wk_b, q_nope);
        cb(q_nope_absorbed, "mtp_q_nope_absorbed", il);

        // {kv_lora_rank, n_head, n_tokens}
        q_nope_absorbed = ggml_permute(ctx0, q_nope_absorbed, 0, 2, 1, 3);
        cb(q_nope_absorbed, "mtp_q_nope_absorbed_perm", il);

        // {n_embd_head_qk_rope + kv_lora_rank, n_head, n_tokens}
        // note: rope must go first for in-place context shifting in build_rope_shift()
        ggml_tensor * Qcur = ggml_concat(ctx0, q_nope_absorbed, q_pe, 0);
        cb(Qcur, "mtp_Qcur", il);

        kv_cmpr = ggml_reshape_3d(ctx0, kv_cmpr, kv_lora_rank, 1, n_tokens);
        cb(kv_cmpr, "mtp_kv_cmpr_reshape", il);

        // {n_embd_head_qk_rope + kv_lora_rank, 1, n_tokens}
        ggml_tensor * Kcur = ggml_concat(ctx0, kv_cmpr, k_pe, 0);
        cb(Kcur, "mtp_Kcur", il);

        // {kv_lora_rank, 1, n_tokens}
        ggml_tensor * Vcur = kv_cmpr;
        cb(Vcur, "mtp_Vcur", il);

        // note: MLA with the absorption optimization converts into MQA (ie: GQA with 1 group)
        cur = build_attn(inp_attn,
                layer.wo, NULL, layer.wo_s,
                Qcur, Kcur, Vcur, nullptr, nullptr, layer.wv_b, kq_scale, il);
        cb(cur, "mtp_attn_out", il);
    }

    if (use_hc) {
        inpL = deepseek32_hc_post(ctx0, cur, residual, post, comb, n_embd);
        residual = inpL;
        cur = deepseek32_hc_pre(ctx0, inpL,
                layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base,
                norm_rms_eps, hparams.dsv4_hc_eps, hparams.dsv4_hc_sinkhorn_iters,
                n_embd, &post, &comb);
    } else {
        cur = ggml_add(ctx0, cur, residual);
    }
    ggml_tensor * ffn_inp = cur;
    cb(ffn_inp, "mtp_ffn_inp", il);

    cur = build_norm(ffn_inp, layer.ffn_norm, NULL, LLM_NORM_RMS, il);
    cb(cur, "mtp_ffn_norm", il);

    // MoE FFN with shared expert - same construction as the deepseek2 trunk graph
    ggml_tensor * moe_out = build_moe_ffn(cur,
        layer.ffn_gate_inp,
        layer.ffn_up_exps,
        layer.ffn_gate_exps,
        layer.ffn_down_exps,
        layer.ffn_exp_probs_b,
        n_expert, n_expert_used,
        LLM_FFN_SILU, hparams.expert_weights_norm,
        hparams.expert_weights_scale,
        (llama_expert_gating_func_type) hparams.expert_gating_func,
        il,
        nullptr,
        layer.ffn_gate_up_exps,
        layer.ffn_up_exps_s,
        layer.ffn_gate_exps_s,
        layer.ffn_down_exps_s);
    cb(moe_out, "mtp_ffn_moe_out", il);

    // FFN shared expert
    ggml_tensor * ffn_shexp =
        build_ffn(cur,
            layer.ffn_up_shexp, NULL, layer.ffn_up_shexp_s,
            layer.ffn_gate_shexp, NULL, layer.ffn_gate_shexp_s,
            layer.ffn_down_shexp, NULL, layer.ffn_down_shexp_s,
            NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
    cb(ffn_shexp, "mtp_ffn_shexp", il);

    cur = ggml_add(ctx0, moe_out, ffn_shexp);
    cb(cur, "mtp_ffn_out", il);

    if (use_hc) {
        inpL = deepseek32_hc_post(ctx0, cur, residual, post, comb, n_embd);
        cur = deepseek32_hc_mean(ctx0, inpL);
    } else {
        cur = ggml_add(ctx0, cur, ffn_inp);
    }
    cb(cur, "mtp_post_ffn", il);

    // shared_head_norm applied after the decoder block, before the shared LM head.
    // The post-norm hidden state seeds the next MTP step.
    ggml_tensor * head_norm_w = layer.nextn.shared_head_norm
            ? layer.nextn.shared_head_norm
            : model.output_norm;
    GGML_ASSERT(head_norm_w && "DEEPSEEK32 MTP: missing both nextn.shared_head_norm and output_norm");
    cur = build_norm(cur, head_norm_w, nullptr, LLM_NORM_RMS, -1);

    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    cb(cur, "mtp_shared_head_norm", -1);

    ggml_tensor * head_w = layer.nextn.shared_head_head ? layer.nextn.shared_head_head : model.output;
    ggml_tensor * head_s = layer.nextn.shared_head_head ? layer.nextn.shared_head_head_s : model.output_s;
    GGML_ASSERT(head_w && "DEEPSEEK32 MTP: missing LM head (nextn.shared_head_head or model.output)");
    cur = build_lora_mm(head_w, cur, head_s);
    cb(cur, "result_output", -1);

    res->t_logits = cur;
}

