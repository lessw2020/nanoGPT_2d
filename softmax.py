# multiple softmax versions

import torch
import triton
import triton.language as tl

# eager softmax

@torch.jit.script
def softmax_eager(x):
    row_max = x.max(dim=1)[0]
    safe_x = x - row_max[:,None]
    numerator = torch.exp(safe_x)
    denom = numerator.sum(dim=1)
    sm_out = numerator/denom[:,None]
    return sm_out

@triton.jit
def _softmax_kernel(
    output_ptr,
    output_stride,
    input_ptr,
    input_stride,
    n_cols,
    block_size: tl.constexpr,

):
    # setup row access
    row_index = tl.program_id(0) 
    row_start_ptr = input_ptr  + ( row_index * input_stride)
    col_offsets = tl.arange(0,block_size)
    row_ptrs = row_start_ptr + col_offsets

    # load row into SRAM
    row = tl.load(row_ptrs, mask=col_offsets < n_cols, other=float("-inf"))

    # softmax
    safe_row = row - tl.max(row,axis=0)
    numerator = tl.exp(safe_row)
    denom = tl.sum(numerator, axis=0)
    sm_out = numerator / denom

    # write out to HBM
    out_start_ptr = output_ptr + ( row_index * output_stride)
    out_ptrs = out_start_ptr + col_offsets
    tl.store(out_ptrs, sm_out, mask = col_offsets < n_cols)



def softmax_triton(x: torch.Tensor)-> torch.Tensor:
    shape = x.shape
    print(f"{shape=}")
    batch, rows, cols = x.shape
    #assert x.dim() == 2, "only supporting 2d inputs for now"
    output = torch.empty_like(x)

    block_size = triton.next_power_of_2(cols)
    grid = (rows,)
    num_warps = 4
    if rows > 2048:
        num_warps=8
    if rows > 4096:
        num_warps=16
    
    _softmax_kernel[grid](
        output,
        output.stride(0),
        x,
        x.stride(0),
        cols,
        block_size,
        num_warps=num_warps
    )
    return output


if __name__== '__main__':
    sample = torch.tensor([[1,2,3,4,5], [5,4,3,2,1]], dtype = torch.float32, device="cuda")
    res = softmax_eager(sample)
    res_triton = softmax_triton(sample)
    print(f"{res_triton=}")




