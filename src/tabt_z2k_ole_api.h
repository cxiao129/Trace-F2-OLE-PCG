#ifndef TABT_Z2K_OLE_API_H
#define TABT_Z2K_OLE_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct TabtZ2kOleStats {
    size_t n;
    size_t c;
    size_t t;
    size_t capacity;
    uint64_t checked;
    uint64_t failures;
};

/*
 * Assemble the Trace-F2-OLE-PCG GR(2^64,2) generalized-trace pipeline into
 * one programmable Z_{2^64} OLE batch.
 *
 * The current local upstream code exposes benchmark entrypoints, not a
 * seed-carrying SeedOLE/ExpandOLE API. This function is deliberately a
 * functional assembly layer: it derives sparse Trace programs from caller
 * seeds, evaluates the real GR128 FFT/trace algebra, and returns both OLE
 * endpoints for local correctness validation/integration.
 */
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
    struct TabtZ2kOleStats *stats);

#ifdef __cplusplus
}
#endif

#endif
