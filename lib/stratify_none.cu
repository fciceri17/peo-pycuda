
__global__ void stratify_none_getD(float *is_class_component, int *indptr, int *indices, int n, float c, float *D)
{
    const int i = threadIdx.x;
    D[i] = 0;
    if(is_class_component[i] == 0) return;

    int d = 0;
    for(int j = indptr[i]; j < indptr[i+1]; j++){
        if(is_class_component[indices[j]]){
            d += 1;
        }
    }

    if(d >= 3 / 5 * c){
        D[i] = 1;
    }
}

__global__ void stratify_none_getC_D(float *is_class_component, int *indptr, int *indices, int n, float *D, float *C_D)
{
    const int i = threadIdx.x;
    C_D[i] = 0;

    if(is_class_component[i] == 0) return;

    if(D[i] == 0) C_D[i] = 1;
}

__global__ void add_self(float *in_component, float *adjacencies, int n)
{
    const int i = threadIdx.x;
    if(in_component[i])
        adjacencies[i*n + n] = 1;

}

__global__ void logic_or(float *arr_a, float *arr_b, float *arr_dest)
{
    const int i = threadIdx.x;
    arr_dest[i] = arr_a[i] || arr_b[i];

}