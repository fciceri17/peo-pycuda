#include <int128.cu>

// Parallel Prefix Sum (Mark Harris Cuda implementation)
__host__ __device__ void parallel_prefix(float *d_idata, float *d_odata, int num_elements)
{

    num_elements += 1;
    float** g_scanBlockSums;

    unsigned int blockSize = BLOCK_SIZE; // max size of the thread blocks
    unsigned int numElts = num_elements;

    int level = 0;

    do
    {
        unsigned int numBlocks =
            max(1, (int)ceil((float)numElts / (2.f * blockSize)));
        if (numBlocks > 1)
        {
            level++;
        }
        numElts = numBlocks;
    } while (numElts > 1);

    cudaMalloc((void**)&g_scanBlockSums, level * sizeof(float*));

    numElts = num_elements;
    level = 0;

    do
    {
        unsigned int numBlocks =
            max(1, (int)ceil((float)numElts / (2.f * blockSize)));
        if (numBlocks > 1)
        {
            cudaMalloc((void**) &g_scanBlockSums[level++],
                                      numBlocks * sizeof(float));
        }
        numElts = numBlocks;
    } while (numElts > 1);

    prescanArrayRecursive(d_odata, d_idata, num_elements, 0, g_scanBlockSums);
    cudaDeviceSynchronize();

}

// Computes the logic and between two arrays
__global__ void logic_and(float *arr_a, float *arr_b, float *arr_dest)
{
    const int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    arr_dest[i] = arr_a[i] && arr_b[i];

}

// Computes the logic or between two arrays
__global__ void logic_or(float *arr_a, float *arr_b, float *arr_dest)
{
    const int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    arr_dest[i] = arr_a[i] || arr_b[i];

}

// Sets all elements in an array to value val
__global__ void init_array(float *arr, float val)
{
    const int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    arr[i] = val;

}

// Creates an array with 1 if numbering[i] == n, 0 otherwise. If the sum of this array is >1, the component is non-singleton
__global__ void am_unique(my_uint128 *numbering, my_uint128 root_n, float *unique)
{
    const int i = threadIdx.x;
    if(numbering[i] == root_n) unique[i] = 1;
    else unique[i]=0;
}

// function to call on each node of the graph, sets its component root based on order in the array of indices and numbering
// this function is called until no further changes are made to the root array, creating all components by numbering
// a mask may be applied if components have to be searched in a restricted set of nodes
__global__ void split_classes(my_uint128 *numbering, int *indptr, int *indices, float *mask, float *roots, float *changes)
{
    const int i = threadIdx.x;
    if(mask[i] == 0){
        roots[i] = -1;
        return;
    }
    int min = roots[i];

    for(int j = indptr[i]; j < indptr[i+1]; j++){
        if(mask[indices[j]] == 1 && numbering[i] == numbering[indices[j]] && roots[indices[j]] < min){
            min = roots[indices[j]];
        }
    }

    if(min != roots[i]){
        roots[i] = min;
        changes[i] += 1;
    }
}

// loops the split_classes function to obtain the components, identifying them by their root, that is the element in the
// component with the smallest index in the array. Calls split_classes until no changes are made to the roots array
__host__ __device__ void get_class_components(my_uint128 *numbering, int *indptr, int *indices, float *mask, int n, float *roots)
{
    float *changes, *sum;

    cudaMalloc((void**)&changes, sizeof(float) * (n+1));
    cudaMalloc((void**)&sum, sizeof(float) * (n+1));

    do{
        init_array<<< 1, n >>>(changes, 0);
        cudaDeviceSynchronize();
        split_classes<<< 1, n >>>(numbering, indptr, indices, mask, roots, changes);
        cudaDeviceSynchronize();
        parallel_prefix(changes, sum, n);
        cudaDeviceSynchronize();
    }while(sum[n] > 0);

    cudaFree(changes);
    cudaFree(sum);
}

// writes the depth of the spanning tree depth of the current node if the current node hasn't been explored yet
// recursively called on node's neighbors, until all nodes have been explored
__global__ void spanning_tree_depth(int *indptr, int *indices, float *level, float *in_component, int *neighbors, int curr_level)
{
    const int i = threadIdx.x;
    int curr_node = neighbors[i];
    if(level[curr_node] > 0 || in_component[curr_node] == 0)
        return;
    level[curr_node] = curr_level;

    int j = indptr[curr_node];
    int num_neighbors = indptr[curr_node+1] - indptr[curr_node];
    if(num_neighbors > 0){
        __syncthreads();
        spanning_tree_depth<<< 1, num_neighbors >>>(indptr, indices, level, in_component, indices+j*sizeof(int), curr_level+1);
        cudaDeviceSynchronize();
    }
}

//outputs level of depth forming a spanning tree for a given root in component. The (level, node index) pair gives a unique
//depth ordering for each node in the component
__host__ __device__ void spanning_tree_numbering(int *indptr, int *indices, float *in_component, float *level, int root, int n)
{
    init_array<<< 1, n >>>(level, 0);
    cudaDeviceSynchronize();
    level[root] = 1;
    int j = indptr[root];
    int num_neighbors = indptr[root + 1] - indptr[root];
    spanning_tree_depth<<< 1, num_neighbors >>>(indptr, indices, level, in_component, indices+j*sizeof(int), 2);
    cudaDeviceSynchronize();

}
// computes the size of each component in an array with component roots
__global__ void compute_component_sizes(float *roots, float *sizes)
{
    const int i = threadIdx.x;
    sizes[(int)roots[i]] += 1;
}

// computes the set of richer neighbors of a component, and also determines whether or not the neighbor satisifes the
// high-degree criterion need in the stratify call
__global__ void richer_neighbors(my_uint128 *numbering, float *roots, int *indptr, int *indices, int root, float c, float *is_richer_neighbor, float *high_degree, float *neighbors_in_c)
{
    const int i = threadIdx.x;
    is_richer_neighbor[i] = 0;
    high_degree[i] = 0;
    neighbors_in_c[i] = 0;
    if(roots[i] == root) return;

    for(int j = indptr[i]; j < indptr[i+1]; j++){
        if(numbering[i] > numbering[indices[j]] && roots[indices[j]] == root){
            is_richer_neighbor[i] = 1;
            neighbors_in_c[i] += 1;
        }
    }
    if(neighbors_in_c[i] >= 2 / 5 * c){
        high_degree[i] = 1;
    }
}

// extracts only elements of a component into an array based on their root
__global__ void in_class(float *roots, int c, float *is_class_component)
{
    const int i = threadIdx.x;
    is_class_component[i] = 0;
    if(roots[i] == c) is_class_component[i] = 1;
}

// like above call, but sets value to root rather than 1
__global__ void in_class_special(float *roots, int c, float *is_class_component)
{
    const int i = threadIdx.x;
    is_class_component[i] = -1;
    if(roots[i] == c) is_class_component[i] = c;
}

// counts neighbors in component for each node in the component. If it has c - 1 neighbors, it is connected to all
// nodes in the component, and therefore makes it elegible to be a clique
__global__ void is_clique(float *in_component, int *indptr, int *indices, int n, float c, float *full_connected)
{
    const int i = threadIdx.x;
    full_connected[i] = 0;
    if(in_component[i] == 0) return;

    int d = 0;
    for(int j = indptr[i]; j < indptr[i+1]; j++){
        if(in_component[indices[j]] == 1){
            d += 1;
        }
    }

    if(d >= c-1){
        full_connected[i] = 1;
    }
}

// Creates a list of nonzero array indices from the parallel_prefix_sum of an array
__global__ void sum_array_to_list(float *sums, float *list)
{
    const int i = threadIdx.x;

    if(sums[i+1] == sums[i]) return;

    list[(int)sums[i+1] - 1] = i;
}

// incrementally increases the numbering for each node that is in the set of nodes to be incremented
__global__ void add_i(my_uint128 *numbering, float *D_sum, int *indptr, int *indices, int n)
{
    const int i = threadIdx.x;

    if(D_sum[i+1] == D_sum[i]) return;

    numbering[i] = add_my_uint128 (numbering[i], int_to_my_uint128(D_sum[i+1]));
}

// Returns elements in array a but not in array b
__global__ void difference(float *a, float *b, float *r)
{
    const int i = threadIdx.x;
    r[i] = a[i] * (1 - b[i]);
}

// finds the index of the first nonzero element in an array
__global__ void find_first(float *a, int *first)
{
    const int i = threadIdx.x;
    if(a[i+1] == 1 && a[i] == 0) *first = i;
}

// increases the numbering of the elements in other_array by amount delta
__global__ void inc_delta(my_uint128 *numbering, float *other_array, my_uint128 delta)
{
    const int i = threadIdx.x;
    if(other_array[i] == 1) numbering[i] = add_my_uint128 (numbering[i], delta);
}

// finds common neighbors between nodes f and s. Required for the stratify_none call
__global__ void find_common_neighbors(float *is_class_component, int *indptr, int *indices, int f, int s, float *r)
{
    const int i = threadIdx.x;
    r[i] = 0;
    if(is_class_component[i] == 0) return;

    int d = 0;
    for(int j = indptr[i]; j < indptr[i+1]; j++){
        if(indices[j] == f || indices[j] == s){
            d += 1;
        }
    }

    if(d == 2) r[i] = 1;
}

// Generic function for array printing
__host__ __device__ void print_array(float *a, int n)
{
    for(int i=0; i<n; i++)
        printf("%f ", a[i]);
    printf("\\n");
}