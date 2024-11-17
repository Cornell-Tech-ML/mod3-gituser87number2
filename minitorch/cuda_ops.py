# type: ignore
# Currently pyright doesn't support numba.cuda

from typing import Callable, Optional, TypeVar, Any

import numba
from numba import cuda
from numba.cuda import jit as _jit
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

FakeCUDAKernel = Any

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

Fn = TypeVar("Fn")


def device_jit(fn: Fn, **kwargs) -> Fn:  # noqa: ANN003
    """Decorator to create device functions."""
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn: Fn, **kwargs) -> FakeCUDAKernel:  # noqa: ANN003
    """Decorator to create device functions."""
    return _jit(**kwargs)(fn)  # type: ignore


to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        cufn: Callable[[float], float] = device_jit(fn)
        f = tensor_map(cufn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Instantiate and run the cuda kernel.
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """Cude zip function"""
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_zip(cufn)

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """Cuda reduce function"""
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_reduce(cufn)

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            threadsperblock = 1024
            blockspergrid = out_a.size
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Cuda matrix multiply function"""
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        # One block per batch, extra rows, extra col
        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implement


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        in_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        if i < out_size:  # guard
            to_index(i, out_shape, out_index)  # Same code sequence as fast_ops.py

            broadcast_index(out_index, out_shape, in_shape, in_index)

            #o = index_to_position(out_index, out_strides)
            j = index_to_position(in_index, in_strides)

            out[i] = fn(in_storage[j])

        # TODO: Implement for Task 3.3.

    return cuda.jit()(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
    ----
        fn: function mappings two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        if i < out_size:  # guard
            to_index(i, out_shape, out_index)
            #o = index_to_position(out_index, out_strides)
            broadcast_index(
                out_index, out_shape, a_shape, a_index
            )  # same zip logic from fast_ops
            j = index_to_position(a_index, a_strides)
            broadcast_index(out_index, out_shape, b_shape, b_index)
            k = index_to_position(b_index, b_strides)
            out[i] = fn(a_storage[j], b_storage[k])

        # TODO: Implement for Task 3.3.

    return cuda.jit()(_zip)  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    """Practice sum kernel to prepare for reduce.

    Given an array of length $n$ and out of size $n // text{blockDIM}$
    it should sum up each blockDim values into an out cell.

    $[a_1, a_2, ..., a_{100}]$

    |

    $[a_1 +...+ a_{31}, a_{32} + ... + a_{64}, ... ,]$

    Note: Each block must do the sum using shared memory!

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        size (int):  length of a.

    """
    BLOCK_DIM = 32

    cache = cuda.shared.array(BLOCK_DIM, numba.float64)
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    pos = cuda.threadIdx.x

    if i < size:  # reg thread guard
        cache[pos] = a[i]  # copy global to shared memory
    else:
        cache[pos] = 0.0  # fill rest of cache with dummy 0s (floats)

    cuda.syncthreads()  # wait for all data pop to finish

    # Parallel reduction in shared memory
    stride = 1
    while stride < BLOCK_DIM:
        if pos % (2 * stride) == 0: # follows structure discussed in class, sums block to pos[0]
            cache[pos] += cache[pos + stride]
        stride *= 2
        cuda.syncthreads() # does all parallel paths and waits, ensures that all are complete before next root level in tree

    # Write result for block to global
    if pos == 0:  # pos = 0 for start of block
        out[cuda.blockIdx.x] = cache[0]  # writes block data from cache

    # TODO: Implement for Task 3.3.


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    """Sum practice function."""
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """CUDA higher-order tensor reduce function.

    Args:
    ----
        fn: reduction function maps two floats to float.

    Returns:
    -------
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
        BLOCK_DIM = 1024
        cache = cuda.shared.array(BLOCK_DIM, numba.float64)
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        out_pos = cuda.blockIdx.x
        pos = cuda.threadIdx.x

        if out_pos < out_size:
            cache[pos] = reduce_value  # initialize
            to_index(out_pos, out_shape, out_index)  

            out_index[reduce_dim] = out_index[reduce_dim] * BLOCK_DIM + pos #offset by block dim

            if out_index[reduce_dim] < a_shape[reduce_dim]:
                cache[pos] = a_storage[index_to_position(out_index, a_strides)]
                cuda.syncthreads()

                stride = 1
                while stride < BLOCK_DIM:
                    if pos % (stride * 2) == 0: #parallel binary tree
                        cache[pos] = fn(cache[pos], cache[pos + stride])
                        cuda.syncthreads()
                    stride *= 2

            # Send to global output
            if pos == 0:
                out[out_pos] = cache[0]  # after binary tree reduced output is in cache[0]

        # TODO: Implement for Task 3.3.

    return jit(_reduce)  # type: ignore


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    """Practice square MM kernel to prepare for matmul.

    Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Compute

    ```
     for i:
         for j:
              for k:
                  out[i, j] += a[i, k] * b[k, j]
    ```

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        b (Storage): storage for `b` tensor.
        size (int): size of the square

    """
    # BLOCK_DIM = 32
    # Included but not needed??? To-Do: Remove if not needed

    # Get thread indices
    i = cuda.threadIdx.x
    j = cuda.threadIdx.y

    # Shared memory for a and b matrices, 2 diff shared mem banks
    shared_a = cuda.shared.array((MAX_DIMS, MAX_DIMS), numba.float32)
    shared_b = cuda.shared.array((MAX_DIMS, MAX_DIMS), numba.float32)

    # Load data into shared memory if within bounds
    if i < size and j < size:  # 2d size guards
        shared_a[i, j] = a[i * size + j]
        shared_b[i, j] = b[i * size + j]

    # Ensure all threads have loaded data
    cuda.syncthreads()

    # Only compute if in guard bounds
    if i < size and j < size:
        # Compute dot product for this element
        sum = 0.0
        for k in range(size):
            sum += shared_a[i, k] * shared_b[k, j]

        # Write result to global memory
        out[i * size + j] = sum

    # TODO: Implement for Task 3.3.


jit_mm_practice = jit(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    """Matrix multiply practice function."""
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """CUDA tensor matrix multiply function.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as ::

    ```python
    assert a_shape[-1] == b_shape[-2]
    ```
    Returns:
        None : Fills in `out`
    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    # Batch dimension - fixed
    batch = cuda.blockIdx.z

    BLOCK_DIM = 32
    a_shared = cuda.shared.array(
        (BLOCK_DIM, BLOCK_DIM), numba.float64
    )  # initialize shared memory
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # The final position c[i, j]
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # Local position in the block
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    # Accumulator for out[i,j]
    result = 0.0

    # Local variables to reduce global reads
    a_rows = a_shape[-2]
    k_size = a_shape[-1]
    b_cols = b_shape[-1]

    # Traverse blocks over shared dimension (k) in (x, k) @ (k, y)
    for k_block in range(0, k_size, BLOCK_DIM):
        # Copy a into shared memory
        a_i = i
        a_j = k_block + ty
        if a_i < a_rows and a_j < k_size:
            a_shared[tx, ty] = a_storage[
                batch * a_batch_stride + a_i * a_strides[-2] + a_j * a_strides[-1]
            ]
        else:
            a_shared[tx, ty] = 0.0  # pad with 0s if out of bounds

        # Copy b into shared memory
        b_i = k_block + tx
        b_j = j
        if b_i < k_size and b_j < b_cols:
            b_shared[tx, ty] = b_storage[
                batch * b_batch_stride + b_i * b_strides[-2] + b_j * b_strides[-1]
            ]
        else:
            b_shared[tx, ty] = 0.0  # pad with 0s if out of bounds

        cuda.syncthreads()  # wait for all data to be loaded into shared memory

        # Compute dot product for position out[i, j] for this block
        for k in range(min(BLOCK_DIM, k_size - k_block)):
            result += a_shared[tx, k] * b_shared[k, ty]

        cuda.syncthreads()

    # Write into global memory
    if i < a_rows and j < b_cols:
        out[batch * out_strides[0] + i * out_strides[1] + j * out_strides[2]] = result

    # TODO: Implement for Task 3.4.


tensor_matrix_multiply = jit(_tensor_matrix_multiply)
