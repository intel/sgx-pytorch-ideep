#ifndef IDEEP_ENCLAVE_HPP
#define IDEEP_ENCLAVE_HPP

#include "sgx_urts.h"
#include "sgx_error.h"
#include "sgx_eid.h"
#define ENCLAVE_FILENAME "/home/hy/libenclave.signed.so"


typedef struct _sgx_errlist_t {
    sgx_status_t err;
    const char *msg;
    const char *sug;
} sgx_errlist_t;

static sgx_errlist_t sgx_errlist[] = {
    {
        SGX_ERROR_UNEXPECTED,
        "Unexpected error occurred.",
        NULL
    },
};

static int initialize_enclave(sgx_enclave_id_t* eid)
{
    sgx_status_t ret = SGX_ERROR_UNEXPECTED;

    // Debug Support: set 2nd parameter to 1
    ret = sgx_create_enclave(ENCLAVE_FILENAME, 1, NULL, NULL, eid, NULL);
    if (ret != SGX_SUCCESS) {
        printf("sgx_create_enclave failed %d.\n", ret);
        return -1;
    }
    else
        printf("sgx_create_enclave successfully.\n");

    return 0;
}


#endif

