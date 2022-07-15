#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <cmath>

namespace {

    __global__ void my_interp_cuda_kernel(
            const torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> d_input,
            const torch::PackedTensorAccessor<float,1,torch::RestrictPtrTraits,size_t> d_interp_loc,
            torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> d_output,
            int direction
        ){
        // direction 0 for horizontal, 1 for vertical

        
        int blockId = blockIdx.x + blockIdx.y * gridDim.x;
        int idx = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
        
        const int bs = d_input.size(0);
        const int ls = d_input.size(1);
        const int cs = d_input.size(2);

        const int newcs = d_interp_loc.size(0);

        if (idx >= bs*ls*newcs){
            return;
        }
        int newcs_idx = idx % newcs;
        int ls_idx = (idx / newcs) % ls;
        int bs_idx = (idx / newcs) / ls;

        float current_loc = d_interp_loc[newcs_idx];
        d_output[bs_idx][ls_idx][newcs_idx][1-direction] = current_loc;

        int pos = -1;
        for (int i = cs - 1; i > 0; i-- ){
            if (d_input[bs_idx][ls_idx][i][direction] < 0 || d_input[bs_idx][ls_idx][i-1][direction] < 0){
                continue;
            }
            if (d_input[bs_idx][ls_idx][i][1-direction] < 0 || d_input[bs_idx][ls_idx][i-1][1-direction] < 0){
                continue;
            }
            if ( (d_input[bs_idx][ls_idx][i][1-direction] - current_loc) * (d_input[bs_idx][ls_idx][i-1][1-direction] - current_loc) <= 0){
                pos = i;
                break;
            }
        }
        if (pos == -1){ return; }

        float len = abs(d_input[bs_idx][ls_idx][pos][1-direction] - d_input[bs_idx][ls_idx][pos-1][1-direction]);
        float part1 = abs( d_input[bs_idx][ls_idx][pos][1-direction] - current_loc );
        // float part2 = abs( d_input[bs_idx][ls_idx][pos-1][1-direction] - current_loc );
        float factor1 = 1 - part1 / len;
        float factor2 = 1 - factor1;

        float value = d_input[bs_idx][ls_idx][pos][direction] * factor1 + d_input[bs_idx][ls_idx][pos-1][direction] * factor2;


        d_output[bs_idx][ls_idx][newcs_idx][direction] = value;

    }


}

torch::Tensor my_interp_cuda(
    torch::Tensor input, torch::Tensor interp_loc, int direction){
    // input is : num_batch, num_lane, num_cls_per_lane, 2
    // interp_loc: new_num_cls_per_lane

    const int bs = input.size(0);
    const int ls = input.size(1);
    const int cs = input.size(2);

    const int newcs = interp_loc.size(0);

    auto options =torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(torch::kCUDA).requires_grad(false);
    auto res = torch::zeros({bs, ls, newcs, 2}, options) -1;


    const int threads = 1024;
    const int blocks = (bs*ls*newcs + threads - 1) / threads;

    my_interp_cuda_kernel<<<blocks, threads>>>(
        input.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
        interp_loc.packed_accessor<float,1,torch::RestrictPtrTraits,size_t>(),
        res.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
        direction
    );
    // direction 0 for horizontal, 1 for vertical
    return res;
}