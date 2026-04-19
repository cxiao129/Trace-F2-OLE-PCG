#include "tabt_z2k_ole_api.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "common.h"
#include "dpf.h"
#include "fft.h"
#include "gr128_bench.h"
#include "gr128_trace_bench.h"
#include "modular_bench.h"
#include "utils.h"

struct TabtTraceProgram {
    size_t *pos;
    struct GR128 *coef;
};

struct TabtPrg {
    uint64_t s;
};

static uint64_t splitmix64_next(struct TabtPrg *prg) {
    uint64_t z = (prg->s += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static uint64_t seed_bytes(const uint8_t *data, size_t len, uint64_t domain) {
    uint64_t h = 0xcbf29ce484222325ULL ^ domain;
    for (size_t i = 0; i < len; ++i) {
        h ^= data[i];
        h *= 0x100000001b3ULL;
    }
    if (h == 0) {
        h = 0x6a09e667f3bcc909ULL ^ domain;
    }
    return h;
}

static size_t prg_index(struct TabtPrg *prg, size_t bound) {
    if (bound <= 1) {
        return 0;
    }
    return (size_t)(splitmix64_next(prg) % bound);
}

static struct GR128 prg_gr128(struct TabtPrg *prg) {
    struct GR128 out;
    out.c0 = (uint128_t)splitmix64_next(prg);
    out.c1 = (uint128_t)splitmix64_next(prg);
    if (out.c0 == 0 && out.c1 == 0) {
        out.c0 = 1;
    }
    return out;
}

static void add_gr128_inplace(struct GR128 *dst, const struct GR128 *src) {
    dst->c0 += src->c0;
    dst->c1 += src->c1;
}

static void sub_gr128_inplace(struct GR128 *dst, const struct GR128 *src) {
    dst->c0 -= src->c0;
    dst->c1 -= src->c1;
}

static struct GR128 sigma_gr128(const struct GR128 *x) {
    struct GR128 out;
    out.c0 = x->c0 - x->c1;
    out.c1 = -x->c1;
    return out;
}

static struct GR128 zeta_mul_gr128(const struct GR128 *x) {
    struct GR128 out;
    out.c0 = -x->c1;
    out.c1 = x->c0 - x->c1;
    return out;
}

static struct GR128 zeta2_mul_gr128(const struct GR128 *x) {
    struct GR128 out;
    out.c0 = x->c1 - x->c0;
    out.c1 = -x->c0;
    return out;
}

static void fft_recursive_gr128_z2k(struct GR128 *coeffs, const size_t num_vars, const size_t num_coeffs) {
    if (num_vars > 1) {
        fft_recursive_gr128_z2k(&coeffs[0], num_vars - 1, num_coeffs / 3);
        fft_recursive_gr128_z2k(&coeffs[num_coeffs], num_vars - 1, num_coeffs / 3);
        fft_recursive_gr128_z2k(&coeffs[2 * num_coeffs], num_vars - 1, num_coeffs / 3);
    }

    struct GR128 *coeffsL = &coeffs[0];
    struct GR128 *coeffsM = &coeffs[num_coeffs];
    struct GR128 *coeffsR = &coeffs[2 * num_coeffs];

    for (size_t j = 0; j < num_coeffs; ++j) {
        struct GR128 m_zeta = zeta_mul_gr128(&coeffsM[j]);
        struct GR128 r_zeta = zeta_mul_gr128(&coeffsR[j]);
        struct GR128 m_zeta2 = zeta2_mul_gr128(&coeffsM[j]);
        struct GR128 r_zeta2 = zeta2_mul_gr128(&coeffsR[j]);

        struct GR128 t0 = coeffsL[j];
        add_gr128_inplace(&t0, &coeffsM[j]);
        add_gr128_inplace(&t0, &coeffsR[j]);

        struct GR128 t1 = coeffsL[j];
        add_gr128_inplace(&t1, &m_zeta);
        add_gr128_inplace(&t1, &r_zeta2);

        struct GR128 t2 = coeffsL[j];
        add_gr128_inplace(&t2, &m_zeta2);
        add_gr128_inplace(&t2, &r_zeta);

        coeffsL[j] = t0;
        coeffsM[j] = t1;
        coeffsR[j] = t2;
    }
}

static void convert_gr128_to_FFT_z2k(const struct Param *param, struct GR128 *polys) {
    const size_t c = param->c;
    const size_t m = param->m;
    const size_t poly_size = param->poly_size;

    for (size_t i = 0; i < c * c * m; ++i) {
        struct GR128 *poly = &polys[i * poly_size];
        fft_recursive_gr128_z2k(poly, param->n, poly_size / 3);
    }
}

static size_t add_pos_mod3(size_t a, size_t b, size_t n) {
    uint8_t *ta = xcalloc(n, sizeof(uint8_t));
    uint8_t *tb = xcalloc(n, sizeof(uint8_t));
    uint8_t *tr = xcalloc(n, sizeof(uint8_t));
    int_to_trits(a, ta, n);
    int_to_trits(b, tb, n);
    for (size_t i = 0; i < n; ++i) {
        tr[i] = (uint8_t)((ta[i] + tb[i]) % 3);
    }
    size_t out = trits_to_int(tr, n);
    free(ta);
    free(tb);
    free(tr);
    return out;
}

static size_t sigma_pos_mod3(size_t a, size_t n) {
    uint8_t *ta = xcalloc(n, sizeof(uint8_t));
    int_to_trits(a, ta, n);
    for (size_t i = 0; i < n; ++i) {
        ta[i] = (uint8_t)((2 * ta[i]) % 3);
    }
    size_t out = trits_to_int(ta, n);
    free(ta);
    return out;
}

static void sample_public_from_seed(
    const struct Param *param,
    struct FFT_GR128_Trace_A *fft_a,
    const uint8_t *seed,
    size_t seed_len) {
    struct TabtPrg prg = {seed_bytes(seed, seed_len, 0x707075626c6963ULL)};
    const size_t poly_size = param->poly_size;
    const size_t c = param->c;
    for (size_t i = 1; i < c; ++i) {
        for (size_t j = 0; j < poly_size; ++j) {
            fft_a->fft_a[i][j] = prg_gr128(&prg);
        }
    }
    for (size_t j = 0; j < poly_size; ++j) {
        fft_a->fft_a[0][j].c0 = 1;
        fft_a->fft_a[0][j].c1 = 0;
    }
    for (size_t i = 0; i < c; ++i) {
        for (size_t j = 0; j < poly_size; ++j) {
            fft_a->fft_a_maps[i][j] = sigma_gr128(&fft_a->fft_a[i][j]);
        }
    }
    for (size_t i = 0; i < c; ++i) {
        for (size_t j = 0; j < c; ++j) {
            for (size_t k = 0; k < poly_size; ++k) {
                mult_gr128(&fft_a->fft_a[i][k],
                           &fft_a->fft_a[j][k],
                           &fft_a->fft_a_tensor_maps[i * c + j][k]);
                mult_gr128(&fft_a->fft_a[i][k],
                           &fft_a->fft_a_maps[j][k],
                           &fft_a->fft_a_tensor_maps[i * c + j][poly_size + k]);
            }
        }
    }
}

static void init_program(
    const struct Param *param,
    struct TabtTraceProgram *program,
    const uint8_t *seed,
    size_t seed_len,
    uint64_t domain) {
    const size_t total = param->c * param->t;
    program->pos = xcalloc(total, sizeof(size_t));
    program->coef = xcalloc(total, sizeof(struct GR128));
    struct TabtPrg prg = {seed_bytes(seed, seed_len, domain)};
    for (size_t i = 0; i < total; ++i) {
        program->pos[i] = prg_index(&prg, param->poly_size);
        program->coef[i] = prg_gr128(&prg);
    }
}

static void free_program(struct TabtTraceProgram *program) {
    free(program->pos);
    free(program->coef);
    program->pos = NULL;
    program->coef = NULL;
}

static void compute_trace_input(
    const struct Param *param,
    const struct FFT_GR128_Trace_A *pp,
    const struct TabtTraceProgram *program,
    uint64_t *out,
    uint64_t count,
    struct GR128 *fft_out) {
    const size_t c = param->c;
    const size_t t = param->t;
    const size_t poly_size = param->poly_size;
    struct GR128 *acc = xcalloc(poly_size, sizeof(struct GR128));
    struct GR128 *poly = xcalloc(poly_size, sizeof(struct GR128));
    struct GR128 tmp;

    for (size_t i = 0; i < c; ++i) {
        memset(poly, 0, poly_size * sizeof(struct GR128));
        for (size_t k = 0; k < t; ++k) {
            const size_t index = i * t + k;
            add_gr128_inplace(&poly[program->pos[index]], &program->coef[index]);
        }
        fft_recursive_gr128_z2k(poly, param->n, poly_size / 3);
        for (size_t j = 0; j < poly_size; ++j) {
            mult_gr128(&pp->fft_a[i][j], &poly[j], &tmp);
            add_gr128_inplace(&acc[j], &tmp);
        }
    }

    for (uint64_t i = 0; i < count; ++i) {
        uint128_t tr = 2 * acc[i].c0 - acc[i].c1;
        out[i] = (uint64_t)tr;
    }
    if (fft_out != NULL) {
        memcpy(fft_out, acc, poly_size * sizeof(struct GR128));
    }

    free(acc);
    free(poly);
}

static void trace_product_share(
    const struct Param *param,
    const struct FFT_GR128_Trace_A *pp,
    struct GR128 *share,
    uint64_t *out,
    uint64_t count) {
    const size_t c = param->c;
    const size_t m = param->m;
    const size_t poly_size = param->poly_size;
    struct GR128 *poly_buf = xcalloc(c * c * m * poly_size, sizeof(struct GR128));
    struct GR128 *z_poly = xcalloc(poly_size, sizeof(struct GR128));

    convert_gr128_to_FFT_z2k(param, share);
    multiply_gr128_FFT(param, pp->fft_a_tensor_maps, share, poly_buf);
    sum_gr128_FFT_polys(param, poly_buf, z_poly);

    for (uint64_t i = 0; i < count; ++i) {
        uint128_t tr = 2 * z_poly[i].c0 - z_poly[i].c1;
        out[i] = (uint64_t)tr;
    }

    free(poly_buf);
    free(z_poly);
}

static void dpf_raw_full_eval(
    struct DPFKey *key,
    uint128_t *cache_arg,
    uint128_t *output_arg,
    uint8_t *control_bits) {
    size_t size = key->size;
    const uint8_t *k = key->k;
    struct PRFKeys *prf_keys = key->prf_keys;
    uint128_t *cache = cache_arg;
    uint128_t *output = output_arg;

    if (size % 2 == 1) {
        uint128_t *tmp = cache;
        cache = output;
        output = tmp;
    }

    const size_t num_leaves = ipow(3, size);
    memcpy(&output[0], &k[0], 16);
    const uint128_t *sCW0 = (uint128_t *)&k[16];
    const uint128_t *sCW1 = (uint128_t *)&k[16 * size + 16];
    const uint128_t *sCW2 = (uint128_t *)&k[16 * 2 * size + 16];

    uint128_t *tmp;
    size_t idx0, idx1, idx2;
    uint8_t cb;
    size_t max_batch_size = ipow(3, 6);
    size_t batch, num_batches, batch_size, offset;
    size_t num_nodes = 1;

    for (uint8_t i = 0; i < size; ++i) {
        if (i < 6) {
            batch_size = num_nodes;
            num_batches = 1;
        } else {
            batch_size = max_batch_size;
            num_batches = num_nodes / max_batch_size;
        }

        offset = 0;
        for (batch = 0; batch < num_batches; ++batch) {
            PRFBatchEval(prf_keys->prf_key0, &output[offset], &cache[offset], batch_size);
            PRFBatchEval(prf_keys->prf_key1, &output[offset], &cache[num_nodes + offset], batch_size);
            PRFBatchEval(prf_keys->prf_key2, &output[offset], &cache[(num_nodes * 2) + offset], batch_size);
            idx0 = offset;
            idx1 = num_nodes + offset;
            idx2 = (num_nodes * 2) + offset;

            while (idx0 < offset + batch_size) {
                cb = (uint8_t)(output[idx0] & 1);
                cache[idx0] ^= (uint128_t)cb * sCW0[i];
                cache[idx1] ^= (uint128_t)cb * sCW1[i];
                cache[idx2] ^= (uint128_t)cb * sCW2[i];
                ++idx0;
                ++idx1;
                ++idx2;
            }
            offset += batch_size;
        }

        tmp = output;
        output = cache;
        cache = tmp;
        num_nodes *= 3;
    }

    const size_t msg_len = key->msg_len;
    ExtendOutput(prf_keys, output, cache, num_leaves, msg_len * num_leaves);
    for (size_t i = 0; i < num_leaves; ++i) {
        control_bits[i] = (uint8_t)(cache[i * msg_len] & 1);
    }
}

static void dpf_gen_additive_gr128(
    struct PRFKeys *prf_keys,
    size_t domain_bits,
    size_t index,
    const struct GR128 *beta,
    uint128_t *cache,
    uint128_t *raw_a,
    uint128_t *raw_b,
    uint8_t *cb_a,
    uint8_t *cb_b,
    struct DPFKey *k0,
    struct DPFKey *k1) {
    uint128_t zero_msg[2] = {0, 0};
    DPFGen(prf_keys, domain_bits, index, zero_msg, 2, k0, k1);
    dpf_raw_full_eval(k0, cache, raw_a, cb_a);
    dpf_raw_full_eval(k1, cache, raw_b, cb_b);

    uint128_t diff0 = raw_a[index * 2] - raw_b[index * 2];
    uint128_t diff1 = raw_a[index * 2 + 1] - raw_b[index * 2 + 1];
    int sign = (int)cb_a[index] - (int)cb_b[index];
    uint128_t cw[2];
    if (sign >= 0) {
        cw[0] = beta->c0 - diff0;
        cw[1] = beta->c1 - diff1;
    } else {
        cw[0] = diff0 - beta->c0;
        cw[1] = diff1 - beta->c1;
    }

    const size_t cw_offset = 16 * 3 * domain_bits + 16;
    memcpy(&k0->k[cw_offset], cw, sizeof(cw));
    memcpy(&k1->k[cw_offset], cw, sizeof(cw));
}

static void dpf_eval_additive_gr128_into(
    struct DPFKey *key,
    int party,
    uint128_t *cache,
    uint128_t *raw,
    uint8_t *control_bits,
    struct GR128 *out) {
    const size_t num_leaves = ipow(3, key->size);
    const uint128_t *cw = (const uint128_t *)&key->k[16 * 3 * key->size + 16];
    dpf_raw_full_eval(key, cache, raw, control_bits);
    for (size_t i = 0; i < num_leaves; ++i) {
        uint128_t c0 = raw[i * 2];
        uint128_t c1 = raw[i * 2 + 1];
        if (control_bits[i]) {
            c0 += cw[0];
            c1 += cw[1];
        }
        if (party == 0) {
            out[i].c0 += c0;
            out[i].c1 += c1;
        } else {
            out[i].c0 -= c0;
            out[i].c1 -= c1;
        }
    }
}

static void add_split_singleton(
    const struct GR128 *beta,
    struct TabtPrg *prg,
    struct GR128 *dst0,
    struct GR128 *dst1) {
    struct GR128 mask = prg_gr128(prg);
    struct GR128 rest = *beta;
    sub_gr128_inplace(&rest, &mask);
    add_gr128_inplace(dst0, &mask);
    add_gr128_inplace(dst1, &rest);
}

static void spfss_add_point(
    const struct Param *param,
    struct PRFKeys *prf_keys,
    const struct GR128 *beta,
    size_t pos,
    struct TabtPrg *singleton_prg,
    uint128_t *cache,
    uint128_t *raw_a,
    uint128_t *raw_b,
    uint8_t *cb_a,
    uint8_t *cb_b,
    struct GR128 *share_recv,
    struct GR128 *share_send) {
    const size_t block_size = param->dpf_block_size;
    const size_t domain_bits = param->dpf_domain_bits;

    if (beta->c0 == 0 && beta->c1 == 0) {
        return;
    }

    const size_t block = pos / block_size;
    const size_t alpha = pos % block_size;
    struct GR128 *dst0 = &share_recv[block * block_size];
    struct GR128 *dst1 = &share_send[block * block_size];

    if (domain_bits == 0) {
        add_split_singleton(beta, singleton_prg, &dst0[0], &dst1[0]);
        return;
    }

    struct DPFKey k0 = {0};
    struct DPFKey k1 = {0};
    dpf_gen_additive_gr128(prf_keys,
                           domain_bits,
                           alpha,
                           beta,
                           cache,
                           raw_a,
                           raw_b,
                           cb_a,
                           cb_b,
                           &k0,
                           &k1);
    dpf_eval_additive_gr128_into(&k0, 0, cache, raw_a, cb_a, dst0);
    dpf_eval_additive_gr128_into(&k1, 1, cache, raw_b, cb_b, dst1);
    free(k0.k);
    free(k1.k);
}

static void spfss_share_cross_terms(
    const struct Param *param,
    const struct TabtTraceProgram *recv,
    const struct TabtTraceProgram *send,
    const uint8_t *recv_seed,
    size_t recv_seed_len,
    const uint8_t *send_seed,
    size_t send_seed_len,
    struct GR128 *share_recv,
    struct GR128 *share_send) {
    const size_t c = param->c;
    const size_t t = param->t;
    const size_t m = param->m;
    const size_t poly_size = param->poly_size;
    const size_t block_size = param->dpf_block_size;
    uint128_t *cache = xcalloc(block_size * 2, sizeof(uint128_t));
    uint128_t *raw_a = xcalloc(block_size * 2, sizeof(uint128_t));
    uint128_t *raw_b = xcalloc(block_size * 2, sizeof(uint128_t));
    uint8_t *cb_a = xcalloc(block_size, sizeof(uint8_t));
    uint8_t *cb_b = xcalloc(block_size, sizeof(uint8_t));
    struct PRFKeys *prf_keys = xmalloc(sizeof(struct PRFKeys));
    PRFKeyGen(prf_keys);

    uint64_t seed_material[2];
    seed_material[0] = seed_bytes(recv_seed, recv_seed_len, 0x737066737330ULL);
    seed_material[1] = seed_bytes(send_seed, send_seed_len, 0x737066737331ULL);
    struct TabtPrg singleton_prg = {seed_material[0] ^ (seed_material[1] << 1)};

    for (size_t i = 0; i < c; ++i) {
        for (size_t j = 0; j < c; ++j) {
            for (size_t w = 0; w < m; ++w) {
                struct GR128 *dst0 = &share_recv[((i * c + j) * m + w) * poly_size];
                struct GR128 *dst1 = &share_send[((i * c + j) * m + w) * poly_size];
                for (size_t k = 0; k < t; ++k) {
                    const size_t ri = i * t + k;
                    for (size_t l = 0; l < t; ++l) {
                        const size_t si = j * t + l;
                        struct GR128 scoef = w == 0
                            ? send->coef[si]
                            : sigma_gr128(&send->coef[si]);
                        size_t spos = w == 0
                            ? send->pos[si]
                            : sigma_pos_mod3(send->pos[si], param->n);
                        struct GR128 coeff;
                        mult_gr128(&recv->coef[ri], &scoef, &coeff);
                        const size_t pos = add_pos_mod3(recv->pos[ri], spos, param->n);
                        spfss_add_point(param,
                                        prf_keys,
                                        &coeff,
                                        pos,
                                        &singleton_prg,
                                        cache,
                                        raw_a,
                                        raw_b,
                                        cb_a,
                                        cb_b,
                                        dst0,
                                        dst1);
                    }
                }
            }
        }
    }

    DestroyPRFKey(prf_keys);
    free(cache);
    free(raw_a);
    free(raw_b);
    free(cb_a);
    free(cb_b);
}

static int validate_params(size_t n, size_t c, size_t t, uint64_t count) {
    if (c != 5 || t != 27 || n < 6 || n > 16 || count == 0) {
        return 0;
    }
    size_t capacity = ipow(3, n);
    if (count > capacity) {
        return 0;
    }
    return (capacity % (t * t)) == 0;
}

static void init_tabt_gr128_params(struct Param *param, size_t n, size_t c, size_t t) {
    param->n = n;
    param->c = c;
    param->t = t;
    param->m = 2;
    param->poly_size = ipow(3, n);
    param->block_size = param->poly_size / t;
    param->dpf_domain_bits = n - 6;
    param->dpf_block_size = ipow(3, param->dpf_domain_bits);
    param->block_bits = n - 3;
}

int tabt_z2k_ole_generate(
    size_t n,
    size_t c,
    size_t t,
    uint64_t count,
    const uint8_t *pp_seed,
    size_t pp_seed_len,
    const uint8_t *recv_seed,
    size_t recv_seed_len,
    const uint8_t *send_seed,
    size_t send_seed_len,
    uint64_t *x_out,
    uint64_t *delta_out,
    uint64_t *q_out,
    uint64_t *t_out,
    int verify_relation,
    struct TabtZ2kOleStats *stats) {
    if (!validate_params(n, c, t, count) || pp_seed == NULL || recv_seed == NULL ||
        send_seed == NULL || x_out == NULL || delta_out == NULL || q_out == NULL ||
        t_out == NULL) {
        return -1;
    }

    struct Param *param = xcalloc(1, sizeof(struct Param));
    init_tabt_gr128_params(param, n, c, t);
    if (param->poly_size % (t * t) != 0 || param->dpf_block_size * t * t != param->poly_size) {
        free(param);
        return -2;
    }

    struct FFT_GR128_Trace_A *pp = xcalloc(1, sizeof(struct FFT_GR128_Trace_A));
    init_FFT_GR128_Trace_A(param, pp);
    sample_public_from_seed(param, pp, pp_seed, pp_seed_len);

    struct TabtTraceProgram recv_program = {0};
    struct TabtTraceProgram send_program = {0};
    init_program(param, &recv_program, recv_seed, recv_seed_len, 0x7265637670726f67ULL);
    init_program(param, &send_program, send_seed, send_seed_len, 0x73656e6470726f67ULL);

    struct GR128 *recv_fft = xcalloc(param->poly_size, sizeof(struct GR128));
    struct GR128 *send_fft = xcalloc(param->poly_size, sizeof(struct GR128));
    compute_trace_input(param, pp, &recv_program, x_out, count, recv_fft);
    compute_trace_input(param, pp, &send_program, delta_out, count, send_fft);

    const size_t product_words = c * c * param->m * param->poly_size;
    struct GR128 *share_recv = xcalloc(product_words, sizeof(struct GR128));
    struct GR128 *share_send = xcalloc(product_words, sizeof(struct GR128));
    spfss_share_cross_terms(param,
                            &recv_program,
                            &send_program,
                            recv_seed,
                            recv_seed_len,
                            send_seed,
                            send_seed_len,
                            share_recv,
                            share_send);
    trace_product_share(param, pp, share_recv, q_out, count);
    trace_product_share(param, pp, share_send, t_out, count);
    for (uint64_t i = 0; i < count; ++i) {
        t_out[i] = (uint64_t)(0 - t_out[i]);
    }
    free(share_recv);
    free(share_send);

    uint64_t checked = 0;
    uint64_t failures = 0;
    if (verify_relation) {
        checked = count;
        for (uint64_t i = 0; i < count; ++i) {
            if ((uint64_t)(q_out[i] - t_out[i]) != (uint64_t)(x_out[i] * delta_out[i])) {
                ++failures;
            }
        }
    }

    if (stats != NULL) {
        stats->n = n;
        stats->c = c;
        stats->t = t;
        stats->capacity = param->poly_size;
        stats->checked = checked;
        stats->failures = failures;
    }

    free(recv_fft);
    free(send_fft);
    free_program(&recv_program);
    free_program(&send_program);
    free_FFT_GR128_Trace_A(param, pp);
    free(param);

    return failures == 0 ? 0 : 1;
}
