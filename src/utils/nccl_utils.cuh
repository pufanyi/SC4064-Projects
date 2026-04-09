/*
 * NCCL Utilities
 * ==============
 * Error checking macro for NCCL calls.
 * Single source of truth — included by tensor_parallel.cu and bench_multi_gpu.cu.
 */

#pragma once

#include <nccl.h>

#include <cstdio>
#include <cstdlib>

#define NCCL_CHECK(cmd)                                                                            \
    do {                                                                                           \
        ncclResult_t r = cmd;                                                                      \
        if (r != ncclSuccess) {                                                                    \
            fprintf(stderr, "NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(r)); \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)
