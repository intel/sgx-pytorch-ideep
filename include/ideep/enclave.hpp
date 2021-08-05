#ifndef IDEEP_ENCLAVE_HPP
#define IDEEP_ENCLAVE_HPP

#include "sgx_urts.h"
#include "sgx_error.h"
#include "sgx_eid.h"
#include "Enclave_u.h"


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

    std::string enclave_path = ENCLAVE_PATH;
    printf("ENCLAVE_PATH is %s.\n", enclave_path.c_str());

    // Debug Support: set 2nd parameter to 1
    ret = sgx_create_enclave(enclave_path.c_str(), 1, NULL, NULL, eid, NULL);

    if (ret != SGX_SUCCESS) {
        printf("sgx_create_enclave failed %d.\n", ret);
        return -1;
    }
    else
        printf("sgx_create_enclave successfully.\n");

    return 0;
}


#endif

