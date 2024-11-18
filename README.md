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


PARALLEL CHECK 3.1/3.2:

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
















GPU Simple:
Total Time: 10:03
Average Time per Epoch: 1.21 sec

Epoch  0  loss  6.206625946464619 correct 44
Epoch  10  loss  2.5471892349982355 correct 49
Epoch  20  loss  0.9805035950893547 correct 49
Epoch  30  loss  0.7974613940056613 correct 49
Epoch  40  loss  0.3691987243098882 correct 49
Epoch  50  loss  0.36554205667072415 correct 49
Epoch  60  loss  0.6020547537800744 correct 49
Epoch  70  loss  1.1224913312437055 correct 50
Epoch  80  loss  0.4850686736822879 correct 50
Epoch  90  loss  0.2644996163552247 correct 50
Epoch  100  loss  0.0959074604820056 correct 50
Epoch  110  loss  0.20013605504324944 correct 50
Epoch  120  loss  0.5574511854190164 correct 50
Epoch  130  loss  0.007593282495780328 correct 50
Epoch  140  loss  0.11406508716664966 correct 50
Epoch  150  loss  0.846240569975847 correct 50
Epoch  160  loss  0.17345749215654369 correct 50
Epoch  170  loss  0.17909261682740074 correct 50
Epoch  180  loss  0.08658197936152316 correct 50
Epoch  190  loss  0.05335153479114112 correct 50
Epoch  200  loss  0.06743143789477872 correct 50
Epoch  210  loss  0.0449052682603517 correct 50
Epoch  220  loss  0.07580699131139779 correct 50
Epoch  230  loss  0.5520032095113726 correct 50
Epoch  240  loss  0.08748110940109485 correct 50
Epoch  250  loss  0.34978436347978364 correct 50
Epoch  260  loss  0.0558065675796175 correct 50
Epoch  270  loss  0.08899037448342165 correct 50
Epoch  280  loss  0.08768458466195761 correct 50
Epoch  290  loss  0.04772733938695023 correct 50
Epoch  300  loss  0.347792052773273 correct 50
Epoch  310  loss  0.04823904907499882 correct 50
Epoch  320  loss  0.030939826781681355 correct 50
Epoch  330  loss  0.027230715683040474 correct 50
Epoch  340  loss  0.035549479203656136 correct 50
Epoch  350  loss  0.2199263247750174 correct 50
Epoch  360  loss  0.24831018218095335 correct 50
Epoch  370  loss  0.0675026769268789 correct 50
Epoch  380  loss  0.23573546727520878 correct 50
Epoch  390  loss  0.05386218323086132 correct 50
Epoch  400  loss  0.21746047715088 correct 50
Epoch  410  loss  0.26014484295919377 correct 50
Epoch  420  loss  0.3509762174543138 correct 50
Epoch  430  loss  0.33228927334852776 correct 50
Epoch  440  loss  0.2279372864224336 correct 50
Epoch  450  loss  0.02160957887816996 correct 50
Epoch  460  loss  0.008547694707064197 correct 50
Epoch  470  loss  0.2818857780337252 correct 50
Epoch  480  loss  0.016212493780688914 correct 50
Epoch  490  loss  0.23029511686696272 correct 50



CPU Simple:
Total Time: 1:21
Average Time per Epoch: 0.162 sec

Epoch  0  loss  5.457756623595779 correct 41
Epoch  10  loss  1.5769509606620138 correct 46
Epoch  20  loss  1.33575702329639 correct 49
Epoch  30  loss  1.7792728468176344 correct 49
Epoch  40  loss  0.7524973040478757 correct 49
Epoch  50  loss  0.9218150278915367 correct 49
Epoch  60  loss  1.362438585839049 correct 48
Epoch  70  loss  0.44849883303847005 correct 50
Epoch  80  loss  0.579156190856539 correct 50
Epoch  90  loss  0.39087320615634635 correct 49
Epoch  100  loss  0.14462499337268034 correct 49
Epoch  110  loss  1.7380452303586456 correct 49
Epoch  120  loss  0.5362885208891575 correct 50
Epoch  130  loss  1.0955929690355632 correct 50
Epoch  140  loss  0.9020862264011086 correct 49
Epoch  150  loss  1.0222664067392053 correct 49
Epoch  160  loss  0.8424740507708717 correct 49
Epoch  170  loss  1.6324946616889904 correct 49
Epoch  180  loss  0.03698598681167755 correct 49
Epoch  190  loss  0.039127500356217954 correct 50
Epoch  200  loss  0.0050601159893076675 correct 50
Epoch  210  loss  0.06165971073631526 correct 50
Epoch  220  loss  0.09610425589493117 correct 49
Epoch  230  loss  0.9810297688393261 correct 49
Epoch  240  loss  1.021275092671503 correct 50
Epoch  250  loss  0.15936258473117174 correct 49
Epoch  260  loss  0.018702045762799456 correct 49
Epoch  270  loss  0.007736087737055133 correct 49
Epoch  280  loss  0.3345755734367651 correct 50
Epoch  290  loss  0.983283693256608 correct 49
Epoch  300  loss  1.1597691036605038 correct 50
Epoch  310  loss  1.0978483476881593 correct 49
Epoch  320  loss  0.04759126807216162 correct 50
Epoch  330  loss  0.3511715403043334 correct 50
Epoch  340  loss  0.7758513896375556 correct 49
Epoch  350  loss  0.6228444534719991 correct 49
Epoch  360  loss  0.9676994210260096 correct 49
Epoch  370  loss  0.00443964965606258 correct 49
Epoch  380  loss  0.12876780239062305 correct 49
Epoch  390  loss  0.4206355106216959 correct 49
Epoch  400  loss  0.01678505579010819 correct 49
Epoch  410  loss  0.01094523981344149 correct 49
Epoch  420  loss  0.02703824309648684 correct 49
Epoch  430  loss  0.6406082959486078 correct 49
Epoch  440  loss  0.008736404818773715 correct 50
Epoch  450  loss  0.11223244884059712 correct 50
Epoch  460  loss  0.31536118234869576 correct 49
Epoch  470  loss  1.080590518803945 correct 49
Epoch  480  loss  0.049557891628728694 correct 49
Epoch  490  loss  0.0023822819322159737 correct 50


GPU Split:
Total Time: 9:50
Average Time per Epoch: 1.18 sec

Epoch  0  loss  6.3492145163456435 correct 29
Epoch  10  loss  4.9128917325032795 correct 38
Epoch  20  loss  2.9716909642073626 correct 40
Epoch  30  loss  5.7087379343901485 correct 45
Epoch  40  loss  3.6624737899538378 correct 50
Epoch  50  loss  2.0507646846145287 correct 49
Epoch  60  loss  1.9142091054385613 correct 49
Epoch  70  loss  2.1640800436621377 correct 50
Epoch  80  loss  1.8562402322069442 correct 49
Epoch  90  loss  1.7373831384447114 correct 48
Epoch  100  loss  0.5604941768044749 correct 49
Epoch  110  loss  1.1446981719213354 correct 50
Epoch  120  loss  1.187536480280806 correct 50
Epoch  130  loss  0.3490963374378955 correct 50
Epoch  140  loss  1.238231040482828 correct 50
Epoch  150  loss  0.5020107515971097 correct 50
Epoch  160  loss  1.4165235329947563 correct 50
Epoch  170  loss  0.29288622373303624 correct 50
Epoch  180  loss  0.6653277224163293 correct 50
Epoch  190  loss  1.2621808750235322 correct 50
Epoch  200  loss  0.37962654454583533 correct 50
Epoch  210  loss  0.6283282463015353 correct 50
Epoch  220  loss  1.1837458864445674 correct 50
Epoch  230  loss  0.794579120925186 correct 50
Epoch  240  loss  0.25058366303110613 correct 50
Epoch  250  loss  0.6047793899356654 correct 50
Epoch  260  loss  0.20722238439118074 correct 50
Epoch  270  loss  0.7004177899594257 correct 50
Epoch  280  loss  0.7197378809353299 correct 50
Epoch  290  loss  0.38110421238187386 correct 50
Epoch  300  loss  0.7438368460104493 correct 50
Epoch  310  loss  0.31759137704809104 correct 50
Epoch  320  loss  0.2771526342949847 correct 50
Epoch  330  loss  0.1340422903701834 correct 50
Epoch  340  loss  0.34992274553292746 correct 50
Epoch  350  loss  0.5910053501357467 correct 50
Epoch  360  loss  0.17030424669317587 correct 50
Epoch  370  loss  0.04634469177195964 correct 50
Epoch  380  loss  0.280717907505403 correct 50
Epoch  390  loss  0.09960152635577746 correct 50
Epoch  400  loss  0.288914660951842 correct 50
Epoch  410  loss  0.12112024460342316 correct 50
Epoch  420  loss  0.44358416450006016 correct 50
Epoch  430  loss  0.6752380020031066 correct 50
Epoch  440  loss  0.4877040696898176 correct 50
Epoch  450  loss  0.35213689090571587 correct 50
Epoch  460  loss  0.006100907199265469 correct 50
Epoch  470  loss  0.11721977848107913 correct 50
Epoch  480  loss  0.5015473769688756 correct 50
Epoch  490  loss  0.2365734064111548 correct 50



CPU Split:
Total Time: 1:21
Average Time per Epoch: 0.162 seconds

Epoch  0  loss  8.603783450660258 correct 21
Epoch  10  loss  6.582210584336402 correct 43
Epoch  20  loss  5.117317691326395 correct 39
Epoch  30  loss  4.918029292714646 correct 36
Epoch  40  loss  5.273702705222643 correct 47
Epoch  50  loss  5.344184747275652 correct 42
Epoch  60  loss  3.7466651343232114 correct 49
Epoch  70  loss  2.9326692297785826 correct 48
Epoch  80  loss  1.3666300741517259 correct 43
Epoch  90  loss  2.3437091431434434 correct 48
Epoch  100  loss  3.3840039777412634 correct 46
Epoch  110  loss  3.4476571909957614 correct 45
Epoch  120  loss  1.2878393943364823 correct 50
Epoch  130  loss  0.6184274577792238 correct 46
Epoch  140  loss  1.9043097100019017 correct 50
Epoch  150  loss  1.5869474505806627 correct 49
Epoch  160  loss  1.1982380691072083 correct 50
Epoch  170  loss  1.6119952141367528 correct 47
Epoch  180  loss  1.7326225918939928 correct 50
Epoch  190  loss  2.3556022132053056 correct 50
Epoch  200  loss  0.950770364636129 correct 50
Epoch  210  loss  1.4466972445786799 correct 49
Epoch  220  loss  0.8722539463613275 correct 50
Epoch  230  loss  0.8239641257790574 correct 50
Epoch  240  loss  0.8354499862016364 correct 50
Epoch  250  loss  0.6694830172353881 correct 49
Epoch  260  loss  0.2636037240468979 correct 50
Epoch  270  loss  2.2213796225448665 correct 45
Epoch  280  loss  0.5689389574212625 correct 50
Epoch  290  loss  0.5327095675974475 correct 50
Epoch  300  loss  0.45749162519718745 correct 50
Epoch  310  loss  0.18819629763256351 correct 50
Epoch  320  loss  0.8913475076304866 correct 50
Epoch  330  loss  0.6211880449129281 correct 50
Epoch  340  loss  0.5328914951770912 correct 50
Epoch  350  loss  0.6454887311087313 correct 50
Epoch  360  loss  0.4039665802209586 correct 50
Epoch  370  loss  0.3411971504215726 correct 50
Epoch  380  loss  0.886008942777027 correct 50
Epoch  390  loss  0.4736643878474843 correct 50
Epoch  400  loss  0.28108092177370814 correct 50
Epoch  410  loss  0.25832682216543496 correct 50
Epoch  420  loss  0.20465246462804426 correct 50
Epoch  430  loss  0.23085658841849727 correct 50
Epoch  440  loss  0.27118587277498796 correct 50
Epoch  450  loss  0.38745912831750084 correct 50
Epoch  460  loss  0.48877120452894607 correct 50
Epoch  470  loss  0.25023999271342806 correct 50
Epoch  480  loss  0.21526845223543853 correct 50
Epoch  490  loss  0.31114850112763087 correct 50


GPU XOR:
Total Time: 9:53
Average Time per Epoch: 1.19sec

Epoch  0  loss  5.310942698200925 correct 39
Epoch  10  loss  3.6460861524755837 correct 44
Epoch  20  loss  2.955300341789256 correct 46
Epoch  30  loss  2.530215703906295 correct 46
Epoch  40  loss  3.2617401209321835 correct 46
Epoch  50  loss  2.117811020345897 correct 47
Epoch  60  loss  1.813675391123928 correct 48
Epoch  70  loss  1.638727723379182 correct 49
Epoch  80  loss  1.7948104973796428 correct 47
Epoch  90  loss  2.456939747567019 correct 49
Epoch  100  loss  1.1412740582195355 correct 47
Epoch  110  loss  3.226520669650705 correct 48
Epoch  120  loss  2.381651831735512 correct 47
Epoch  130  loss  1.6578587465059087 correct 47
Epoch  140  loss  2.7257135168989186 correct 47
Epoch  150  loss  1.7271762010383633 correct 49
Epoch  160  loss  0.7996296717218202 correct 49
Epoch  170  loss  0.48025545755172866 correct 49
Epoch  180  loss  1.1472060752211994 correct 49
Epoch  190  loss  1.5073917663419716 correct 49
Epoch  200  loss  0.5373892553287956 correct 49
Epoch  210  loss  1.271925107423326 correct 50
Epoch  220  loss  0.8100923223671256 correct 49
Epoch  230  loss  0.3764705447755022 correct 50
Epoch  240  loss  0.6053959204408529 correct 50
Epoch  250  loss  0.5445318404420296 correct 49
Epoch  260  loss  1.244618569485965 correct 50
Epoch  270  loss  0.24571861584860955 correct 50
Epoch  280  loss  1.6328877051519552 correct 50
Epoch  290  loss  0.3731640610633212 correct 50
Epoch  300  loss  1.3095810360648097 correct 49
Epoch  310  loss  0.10931794329125319 correct 49
Epoch  320  loss  1.21347798471712 correct 50
Epoch  330  loss  0.636650077971557 correct 49
Epoch  340  loss  0.6830146959192388 correct 49
Epoch  350  loss  0.20555211589352332 correct 50
Epoch  360  loss  0.33479594758319986 correct 50
Epoch  370  loss  0.12134909613428624 correct 50
Epoch  380  loss  0.17423307118578804 correct 50
Epoch  390  loss  0.043471722715609874 correct 50
Epoch  400  loss  1.0533128397293303 correct 50
Epoch  410  loss  0.18962973390071203 correct 49
Epoch  420  loss  0.31023937621709197 correct 50
Epoch  430  loss  0.4016042477388878 correct 50
Epoch  440  loss  0.9968534934649803 correct 50
Epoch  450  loss  0.09155483321094117 correct 50
Epoch  460  loss  0.6511519137668546 correct 50
Epoch  470  loss  0.2296236318382353 correct 50
Epoch  480  loss  0.4601543131874097 correct 50
Epoch  490  loss  0.1379352702312036 correct 50


CPU XOR:
Total Time: 1:22
Average Time per Epoch: 0.164 sec

Epoch  0  loss  5.480224947481973 correct 27
Epoch  10  loss  4.075770910070962 correct 43
Epoch  20  loss  3.9789966789506117 correct 44
Epoch  30  loss  5.082128634548825 correct 40
Epoch  40  loss  4.252674005992152 correct 45
Epoch  50  loss  3.6931930532339026 correct 47
Epoch  60  loss  3.188739649965321 correct 46
Epoch  70  loss  2.960532421388165 correct 47
Epoch  80  loss  1.7840805351975997 correct 48
Epoch  90  loss  2.2363775466838645 correct 48
Epoch  100  loss  1.9653338681661259 correct 48
Epoch  110  loss  1.4774261035451983 correct 48
Epoch  120  loss  1.5413682801635111 correct 48
Epoch  130  loss  1.5187482145646252 correct 46
Epoch  140  loss  2.3050055841438795 correct 48
Epoch  150  loss  2.7808768666598995 correct 47
Epoch  160  loss  2.589177113540803 correct 50
Epoch  170  loss  0.4905936668005946 correct 49
Epoch  180  loss  1.9458220046446766 correct 50
Epoch  190  loss  2.525798372903358 correct 48
Epoch  200  loss  1.8183333690679901 correct 50
Epoch  210  loss  2.3198478177280726 correct 48
Epoch  220  loss  1.7336764109554985 correct 49
Epoch  230  loss  0.1385100353887874 correct 48
Epoch  240  loss  0.983552536320708 correct 48
Epoch  250  loss  1.2163073528526505 correct 49
Epoch  260  loss  0.35229911871930475 correct 48
Epoch  270  loss  1.6830091268582819 correct 48
Epoch  280  loss  1.830809889035003 correct 48
Epoch  290  loss  1.1193535975041806 correct 49
Epoch  300  loss  1.9071264456592631 correct 48
Epoch  310  loss  2.136752498359173 correct 48
Epoch  320  loss  1.8240377237641003 correct 47
Epoch  330  loss  1.3973546581348744 correct 50
Epoch  340  loss  0.5488027126373839 correct 50
Epoch  350  loss  0.5414608659618909 correct 50
Epoch  360  loss  0.799008441251847 correct 48
Epoch  370  loss  2.746976693739139 correct 48
Epoch  380  loss  1.127451912366987 correct 49
Epoch  390  loss  0.9100819788577719 correct 50
Epoch  400  loss  1.9273744806336244 correct 48
Epoch  410  loss  0.21794111690619392 correct 48
Epoch  420  loss  1.9253328945622445 correct 48
Epoch  430  loss  0.42591629064331776 correct 50
Epoch  440  loss  0.45284265193635403 correct 50
Epoch  450  loss  1.1331107137994518 correct 49
Epoch  460  loss  1.3761278856450345 correct 50
Epoch  470  loss  0.7706448351016437 correct 50
Epoch  480  loss  0.9666146432257136 correct 49
Epoch  490  loss  0.29034075026024714 correct 49



Large Model - Simple with Hidden = 200

GPU
Total Time: 10:30
Average Time: 1.26 seconds

Epoch  0  loss  8.92757287846681 correct 21
Epoch  10  loss  1.98591482433413 correct 48
Epoch  20  loss  0.9289483260910978 correct 50
Epoch  30  loss  0.3702102396337607 correct 49
Epoch  40  loss  0.2668080821182341 correct 49
Epoch  50  loss  0.10016245818104877 correct 48
Epoch  60  loss  0.23050762670398353 correct 48
Epoch  70  loss  0.2004714539458176 correct 48
Epoch  80  loss  1.0731993493086982 correct 50
Epoch  90  loss  0.11493712251144571 correct 50
Epoch  100  loss  0.07074356063001876 correct 49
Epoch  110  loss  0.16324848002979 correct 48
Epoch  120  loss  0.9847754875007527 correct 50
Epoch  130  loss  0.022718671214725282 correct 48
Epoch  140  loss  0.1969892507774778 correct 50
Epoch  150  loss  0.7235611522900026 correct 50
Epoch  160  loss  0.08324591565417304 correct 50
Epoch  170  loss  0.024862246454172315 correct 48
Epoch  180  loss  0.4948052043171781 correct 50
Epoch  190  loss  0.02305846319953367 correct 49
Epoch  200  loss  0.02923724651334461 correct 50
Epoch  210  loss  0.02482675596580356 correct 50
Epoch  220  loss  0.07975139623399793 correct 50
Epoch  230  loss  0.23942217679362696 correct 50
Epoch  240  loss  0.3701452658201963 correct 50
Epoch  250  loss  0.3208979544916436 correct 50
Epoch  260  loss  0.020732644907154953 correct 50
Epoch  270  loss  0.539663436351629 correct 49
Epoch  280  loss  0.5655967154693741 correct 50
Epoch  290  loss  0.5025701398643183 correct 50
Epoch  300  loss  1.1488404970356916 correct 50
Epoch  310  loss  9.136010282381289e-05 correct 50
Epoch  320  loss  0.3309977551561974 correct 50
Epoch  330  loss  0.2957108855539082 correct 50
Epoch  340  loss  0.17246257094093334 correct 50
Epoch  350  loss  0.20189671672950715 correct 50
Epoch  360  loss  0.014130871163978693 correct 50
Epoch  370  loss  0.0007447571319091972 correct 50
Epoch  380  loss  0.344420267057002 correct 50
Epoch  390  loss  0.08692330499807344 correct 50
Epoch  400  loss  0.3881688049982792 correct 50
Epoch  410  loss  0.09086840352915244 correct 50
Epoch  420  loss  0.23419478005093014 correct 50
Epoch  430  loss  0.724721169957777 correct 50
Epoch  440  loss  0.22851589300371067 correct 50
Epoch  450  loss  0.08520879331290852 correct 50
Epoch  460  loss  0.21892634500104857 correct 50
Epoch  470  loss  0.4013166687098459 correct 50
Epoch  480  loss  0.18817657674835075 correct 50
Epoch  490  loss  0.25341991336249436 correct 50


CPU
Total Time: 2:23
Average Time: 0.286 sec

Epoch  0  loss  2.5253076045920917 correct 38
Epoch  10  loss  6.465895625854253 correct 44
Epoch  20  loss  1.124342713136247 correct 50
Epoch  30  loss  0.7086661356105151 correct 48
Epoch  40  loss  0.7807999750780793 correct 50
Epoch  50  loss  1.9865610139728493 correct 50
Epoch  60  loss  0.7721924715176467 correct 50
Epoch  70  loss  1.390038516598796 correct 47
Epoch  80  loss  0.08813912712173239 correct 50
Epoch  90  loss  1.757084392100541 correct 48
Epoch  100  loss  0.4213883478554402 correct 50
Epoch  110  loss  0.3426231922086758 correct 50
Epoch  120  loss  0.5121300981140936 correct 50
Epoch  130  loss  0.6914411499817387 correct 50
Epoch  140  loss  0.4373464617054605 correct 50
Epoch  150  loss  0.16316481341156747 correct 50
Epoch  160  loss  0.5059174036694797 correct 50
Epoch  170  loss  0.5429632078877404 correct 50
Epoch  180  loss  0.9524272260981751 correct 49
Epoch  190  loss  0.48259932282740975 correct 50
Epoch  200  loss  0.13160545681447722 correct 50
Epoch  210  loss  0.2690955305784865 correct 50
Epoch  220  loss  0.17687574876564688 correct 50
Epoch  230  loss  0.01631895523391126 correct 50
Epoch  240  loss  0.2210055203303227 correct 50
Epoch  250  loss  0.026592664811590536 correct 50
Epoch  260  loss  0.16117059390000033 correct 50
Epoch  270  loss  0.4040534960113813 correct 50
Epoch  280  loss  0.14587485927945293 correct 50
Epoch  290  loss  0.5508087671922914 correct 50
Epoch  300  loss  0.7112075328487139 correct 50
Epoch  310  loss  0.3925836504585622 correct 50
Epoch  320  loss  0.15870099273264063 correct 50
Epoch  330  loss  1.0222177815999565 correct 49
Epoch  340  loss  0.2922916494234822 correct 50
Epoch  350  loss  0.006292766999986413 correct 50
Epoch  360  loss  0.329409728695678 correct 50
Epoch  370  loss  0.011779911248492078 correct 49
Epoch  380  loss  0.16392781820390004 correct 50
Epoch  390  loss  0.004424805057230488 correct 50
Epoch  400  loss  0.2968971112333085 correct 50
Epoch  410  loss  0.0017752235211095065 correct 50
Epoch  420  loss  0.3049326418875311 correct 50
Epoch  430  loss  0.1385124524268146 correct 50
Epoch  440  loss  0.46280743023976145 correct 50
Epoch  450  loss  0.05350048132315493 correct 50
Epoch  460  loss  0.5122467880704027 correct 50
Epoch  470  loss  0.18805768572905823 correct 50
Epoch  480  loss  0.32125031422855554 correct 50
Epoch  490  loss  0.2817282683405374 correct 50




