# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py



PARALLEL CHECK:




Parallel loop listing for  Function tensor_map.<locals>._map, C:\Users\gtown\MLE\Modules\mod3-gituser87number2\minitorch\fast_ops.py (164)
--------------------------------------------------------------------------------------|loop #ID
    def _map(                                                                         |
        out: Storage,                                                                 |
        out_shape: Shape,                                                             |
        out_strides: Strides,                                                         |
        in_storage: Storage,                                                          |
        in_shape: Shape,                                                              |
        in_strides: Strides,                                                          |
    ) -> None:                                                                        |
        if (                                                                          |
            (out_strides != in_strides).any()-----------------------------------------| #0
            or (out_shape != in_shape).any()------------------------------------------| #1
            or len(out_strides) != len(in_strides)  # If need for broadcasting        |
            or len(out_shape) == 0                                                    |
        ):                                                                            |
            for i in prange(len(out)):------------------------------------------------| #3
                out_index: Index = np.empty(MAX_DIMS, np.int32)                       |
                in_index: Index = np.empty(MAX_DIMS, np.int32)                        |
                to_index(i, out_shape, out_index)                                     |
                broadcast_index(out_index, out_shape, in_shape, in_index)             |
                o = index_to_position(out_index, out_strides)                         |
                j = index_to_position(in_index, in_strides)                           |
                out[o] = fn(in_storage[j])                                            |
        else:                                                                         |
            # When `out` and `in` are stride-aligned, avoid indexing                  |
            for i in prange(len(out)):------------------------------------------------| #2
                out[i] = fn(in_storage[i])  # short circuit when strides are equal    |
                                                                                      |
        # TODO: Implement for Task 3.1.                                               |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 4 parallel for-
loop(s) (originating from loops labelled: #0, #1, #3, #2).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
C:\Users\gtown\MLE\Modules\mod3-gituser87number2\minitorch\fast_ops.py (179) is
hoisted out of the parallel loop labelled #3 (it will be performed before the
loop is executed and reused inside the loop):
   Allocation:: out_index: Index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
C:\Users\gtown\MLE\Modules\mod3-gituser87number2\minitorch\fast_ops.py (180) is
hoisted out of the parallel loop labelled #3 (it will be performed before the
loop is executed and reused inside the loop):
   Allocation:: in_index: Index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip,
C:\Users\gtown\MLE\Modules\mod3-gituser87number2\minitorch\fast_ops.py (219)
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, C:\Users\gtown\MLE\Modules\mod3-gituser87number2\minitorch\fast_ops.py (219)
-------------------------------------------------------------------------------|loop #ID
    def _zip(                                                                  |
        out: Storage,                                                          |
        out_shape: Shape,                                                      |
        out_strides: Strides,                                                  |
        a_storage: Storage,                                                    |
        a_shape: Shape,                                                        |
        a_strides: Strides,                                                    |
        b_storage: Storage,                                                    |
        b_shape: Shape,                                                        |
        b_strides: Strides,                                                    |
    ) -> None:                                                                 |
        if (                                                                   |
            len(out_strides) != len(a_strides)                                 |
            or len(out_strides) != len(b_strides)                              |
            or (out_strides != a_strides).any()--------------------------------| #4
            or (out_strides != b_strides).any()  # If need for broadcasting----| #5
            or (out_shape != a_shape).any()------------------------------------| #6
            or (out_shape != b_shape).any()------------------------------------| #7
        ):                                                                     |
            for i in prange(len(out)):-----------------------------------------| #9
                out_index: Index = np.empty(MAX_DIMS, np.int32)                |
                a_index: Index = np.empty(MAX_DIMS, np.int32)                  |
                b_index: Index = np.empty(MAX_DIMS, np.int32)                  |
                to_index(i, out_shape, out_index)                              |
                o = index_to_position(out_index, out_strides)                  |
                broadcast_index(out_index, out_shape, a_shape, a_index)        |
                j = index_to_position(a_index, a_strides)                      |
                broadcast_index(out_index, out_shape, b_shape, b_index)        |
                k = index_to_position(b_index, b_strides)                      |
                out[o] = fn(a_storage[j], b_storage[k])                        |
        else:                                                                  |
            # When out, a, b are stride-aligned, avoid indexing                |
            for i in prange(len(out)):-----------------------------------------| #8
                out[i] = fn(                                                   |
                    a_storage[i], b_storage[i]                                 |
                )  # short circuit when strides are equal                      |
                                                                               |
        # TODO: Implement for Task 3.1.                                        |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 6 parallel for-
loop(s) (originating from loops labelled: #4, #5, #6, #7, #9, #8).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
C:\Users\gtown\MLE\Modules\mod3-gituser87number2\minitorch\fast_ops.py (239) is
hoisted out of the parallel loop labelled #9 (it will be performed before the
loop is executed and reused inside the loop):
   Allocation:: out_index: Index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
C:\Users\gtown\MLE\Modules\mod3-gituser87number2\minitorch\fast_ops.py (240) is
hoisted out of the parallel loop labelled #9 (it will be performed before the
loop is executed and reused inside the loop):
   Allocation:: a_index: Index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
C:\Users\gtown\MLE\Modules\mod3-gituser87number2\minitorch\fast_ops.py (241) is
hoisted out of the parallel loop labelled #9 (it will be performed before the
loop is executed and reused inside the loop):
   Allocation:: b_index: Index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce,
C:\Users\gtown\MLE\Modules\mod3-gituser87number2\minitorch\fast_ops.py (282)
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, C:\Users\gtown\MLE\Modules\mod3-gituser87number2\minitorch\fast_ops.py (282)
-------------------------------------------------------------|loop #ID
    def _reduce(                                             |
        out: Storage,                                        |
        out_shape: Shape,                                    |
        out_strides: Strides,                                |
        a_storage: Storage,                                  |
        a_shape: Shape,                                      |
        a_strides: Strides,                                  |
        reduce_dim: int,                                     |
    ) -> None:                                               |
        for i in prange(len(out)):---------------------------| #10
            out_index = np.empty(MAX_DIMS, np.int32)         |
            size = a_shape[reduce_dim]  # the reduce size    |
            to_index(i, out_shape, out_index)                |
            o = index_to_position(out_index, out_strides)    |
            j = index_to_position(out_index, a_strides)      |
            a = out[o]                                       |
            step = a_strides[reduce_dim]                     |
            for _ in range(size):                            |
                a = fn(a, a_storage[j])                      |
                j += step                                    |
            out[o] = a                                       |
                                                             |
        # TODO: Implement for Task 3.1.                      |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #10).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
C:\Users\gtown\MLE\Modules\mod3-gituser87number2\minitorch\fast_ops.py (292) is
hoisted out of the parallel loop labelled #10 (it will be performed before the
loop is executed and reused inside the loop):
   Allocation:: out_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY

================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply,
C:\Users\gtown\MLE\Modules\mod3-gituser87number2\minitorch\fast_ops.py (309)
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, C:\Users\gtown\MLE\Modules\mod3-gituser87number2\minitorch\fast_ops.py (309)
-----------------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                                   |
    out: Storage,                                                                              |
    out_shape: Shape,                                                                          |
    out_strides: Strides,                                                                      |
    a_storage: Storage,                                                                        |
    a_shape: Shape,                                                                            |
    a_strides: Strides,                                                                        |
    b_storage: Storage,                                                                        |
    b_shape: Shape,                                                                            |
    b_strides: Strides,                                                                        |
) -> None:                                                                                     |
    """NUMBA tensor matrix multiply function.                                                  |
                                                                                               |
    Should work for any tensor shapes that broadcast as long as                                |
                                                                                               |
    ```                                                                                        |
    assert a_shape[-1] == b_shape[-2]                                                          |
    ```                                                                                        |
                                                                                               |
    Optimizations:                                                                             |
                                                                                               |
    * Outer loop in parallel                                                                   |
    * No index buffers or function calls                                                       |
    * Inner loop should have no global writes, 1 multiply.                                     |
                                                                                               |
                                                                                               |
    Args:                                                                                      |
    ----                                                                                       |
        out (Storage): storage for `out` tensor                                                |
        out_shape (Shape): shape for `out` tensor                                              |
        out_strides (Strides): strides for `out` tensor                                        |
        a_storage (Storage): storage for `a` tensor                                            |
        a_shape (Shape): shape for `a` tensor                                                  |
        a_strides (Strides): strides for `a` tensor                                            |
        b_storage (Storage): storage for `b` tensor                                            |
        b_shape (Shape): shape for `b` tensor                                                  |
        b_strides (Strides): strides for `b` tensor                                            |
                                                                                               |
    Returns:                                                                                   |
    -------                                                                                    |
        None : Fills in `out`                                                                  |
                                                                                               |
    """                                                                                        |
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0  # provided                         |
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                                     |
                                                                                               |
    # Parallel loop through batches and matrix dimensions                                      |
    for batch in prange(out_shape[0]):  # Batch calc dimension, parallel-----------------------| #11
        for row in range(out_shape[1]):  # Rows, parallel                                      |
            for col in range(out_shape[2]):  # Columns, parallel                               |
                # Position in A and B for this batch, row, and column                          |
                                                                                               |
                a_pos = batch * a_batch_stride + row * a_strides[1]                            |
                # Position in matrix A                                                         |
                                                                                               |
                b_pos = batch * b_batch_stride + col * b_strides[2]                            |
                # Position in matrix B                                                         |
                                                                                               |
                # Batch shift used to slide to proper batch before selecting row and column    |
                # batch * x_batch_stride #                                                     |
                                                                                               |
                # Matrix multiplication for (A,B) position                                     |
                result = 0.0                                                                   |
                for _ in range(a_shape[2]):  # Inner product size                              |
                    result += a_storage[a_pos] * b_storage[b_pos]                              |
                    a_pos += a_strides[2]  # Move along row A, loop shift                      |
                    b_pos += b_strides[1]  # Move down column B, loop shift                    |
                                                                                               |
                # Store result in output tensor                                                |
                out_pos = (                                                                    |
                    batch * out_strides[0] + row * out_strides[1] + col * out_strides[2]       |
                )                                                                              |
                                                                                               |
                out[out_pos] = result                                                          |
                                                                                               |
    # TODO: Implement for Task 3.2.                                                            |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #11).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None


















GPU Split: 9m50.352s

Epoch  0  loss  7.323205704875873 correct 32
Epoch  10  loss  6.118753015662735 correct 36
Epoch  20  loss  4.738324459597917 correct 36
Epoch  30  loss  4.834154169163034 correct 46
Epoch  40  loss  3.834651316510346 correct 45
Epoch  50  loss  3.5499430723261267 correct 45
Epoch  60  loss  3.2117575171059842 correct 48
Epoch  70  loss  3.038699117293942 correct 49
Epoch  80  loss  3.868245716147908 correct 47
Epoch  90  loss  1.1737545827502984 correct 45
Epoch  100  loss  2.389566293224634 correct 47
Epoch  110  loss  1.9337902930985198 correct 45
Epoch  120  loss  3.328557549365134 correct 44
Epoch  130  loss  1.9086863502541707 correct 42
Epoch  140  loss  3.3647714968133755 correct 47
Epoch  150  loss  2.5688252655270585 correct 48
Epoch  160  loss  1.1558588222616923 correct 47
Epoch  170  loss  2.307879114327939 correct 48
Epoch  180  loss  0.3355852651074087 correct 47
Epoch  190  loss  1.2483154905780764 correct 47
Epoch  200  loss  2.3698112781194 correct 49
Epoch  210  loss  2.823223576674635 correct 46
Epoch  220  loss  1.2455376041478992 correct 49
Epoch  230  loss  0.7138262830530546 correct 49
Epoch  240  loss  0.1573853134575099 correct 48
Epoch  250  loss  0.4950519089319722 correct 48
Epoch  260  loss  0.18089741073599222 correct 49
Epoch  270  loss  1.3778912809056694 correct 48
Epoch  280  loss  2.831587383752258 correct 46
Epoch  290  loss  0.832461607227724 correct 47
Epoch  300  loss  0.9727058201801176 correct 49
Epoch  310  loss  3.639675616359032 correct 42
Epoch  320  loss  1.075445609460953 correct 49
Epoch  330  loss  2.7430423841616567 correct 47
Epoch  340  loss  0.830590468828901 correct 49
Epoch  350  loss  0.5037589819539816 correct 49
Epoch  360  loss  1.4780136286110916 correct 50
Epoch  370  loss  0.6214464673831179 correct 49
Epoch  380  loss  0.5306942414091794 correct 50
Epoch  390  loss  1.5455865584073665 correct 49
Epoch  400  loss  0.4381954956269034 correct 48
Epoch  410  loss  0.8229203533940941 correct 47
Epoch  420  loss  0.5637161440342296 correct 48
Epoch  430  loss  2.0031882116019237 correct 47
Epoch  440  loss  1.0632973097981495 correct 49
Epoch  450  loss  1.7030354342834857 correct 50
Epoch  460  loss  1.9258848294789512 correct 49
Epoch  470  loss  0.5821171810626986 correct 50
Epoch  480  loss  1.5423451037576692 correct 49
Epoch  490  loss  0.5256362130192228 correct 49


CPU Split: 1m23.264s


GPU XOR:

CPU XOR:


