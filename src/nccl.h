#pragma once

#include <stddef.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    ncclSuccess = 0,
    ncclInvalidArgument = 1,
    ncclUnhandledCudaError = 2,
    ncclSystemError = 3,
} ncclResult_t;

typedef enum {
    ncclFloat = 0,
} ncclDataType_t;

typedef enum {
    ncclSum = 0,
} ncclRedOp_t;

typedef struct ncclCommStub {
    int nranks;
    int rank;
    int group_id;
} *ncclComm_t;

const char *ncclGetErrorString(ncclResult_t result);
ncclResult_t ncclCommInitAll(ncclComm_t *comms, int ndev, const int *devlist);
ncclResult_t ncclCommDestroy(ncclComm_t comm);
ncclResult_t ncclAllGather(const void *sendbuff, void *recvbuff, size_t sendcount,
                           ncclDataType_t datatype, ncclComm_t comm,
                           cudaStream_t stream);
ncclResult_t ncclAllReduce(const void *sendbuff, void *recvbuff, size_t count,
                           ncclDataType_t datatype, ncclRedOp_t op,
                           ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclGroupStart(void);
ncclResult_t ncclGroupEnd(void);

#ifdef __cplusplus
}
#endif
