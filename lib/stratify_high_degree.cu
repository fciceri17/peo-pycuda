// Computes adjacencies matrix in parallel
__global__ void compute_adjacent_nodes(int *indptr, int *indices, float *in_component, float *update_values, float *adjacencies, int n)
{
    const int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(i >= n) return;
    if(update_values[i] == 0)
        return;
    int offset = i*n;
    for(int j = indptr[i]; j < indptr[i+1]; j++)
        if(in_component[indices[j]] == 1)
            adjacencies[offset+indices[j]] = 1;

}