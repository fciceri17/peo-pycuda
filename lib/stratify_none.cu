// Getter of the D set
// The D set is defined as the nodes in C that have more than 3/5*|C| neighbors in C
__global__ void stratify_none_getD(float *C, int *indptr, int *indices, int n, float c, float *D)
{
    const int i = threadIdx.x;
    D[i] = 0;
    if(C[i] == 0) return;

    int d = 0;
    for(int j = indptr[i]; j < indptr[i+1]; j++){
        if(C[indices[j]]){
            d += 1;
        }
    }

    if(d >= 3 / 5 * c){
        D[i] = 1;
    }
}