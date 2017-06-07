
__global__ void compute_adjacent_nodes(int *indptr, int *indices, float *in_component, float *update_values, float *adjancencies, int n)
{
    const int i = threadIdx.x;
    if(update_values[i]==0)
        return;
    int offset = i*n;
    for(int j = indptr[i]; j < indptr[i+1]; j++)
        if(in_component[indices[j]] == 1)
            adjancencies[offset+indices[j]] = 1;

}

__global__ void logic_and(float *arr_a, float *arr_b, float *arr_dest)
{
    const int i = threadIdx.x;
    arr_dest[i] = arr_a[i] && arr_b[i];

}