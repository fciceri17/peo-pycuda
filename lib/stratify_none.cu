
__global__ void stratify_none_getD(double *numbering, long long int *roots, int *indptr, int *indices, int n, float c, long long int root, float *D)
{
    const int i = threadIdx.x;
    D[i] = 0;
    if(roots[i] != root) return;

    int d = 0;
    for(int j = indptr[i]; j < indptr[i+1]; j++){
        if(roots[indices[j]] == root){
            d += 1;
        }
    }

    if(d >= 3 / 5 * c){
        D[i] = 1;
    }
}

__global__ void stratify_none_getC_D(double *numbering, long long int *roots, int *indptr, int *indices, int n, float *D, long long int root, float *C_D)
{
    const int i = threadIdx.x;
    C_D[i] = 0;

    if(roots[i] != root) return;

    if(D[i] == 0) C_D[i] = 1;
}