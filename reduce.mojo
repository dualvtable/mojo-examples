 Simple reduction operation on a large array of values to produce a single result
# Reductions and scans are common algorithm patterns in parallel computing.

from Benchmark import Benchmark
from DType import DType
from Object import object
from Pointer import DTypePointer
from SIMD import Float32, Float64
from Random import rand
from IO import print
from Range import range
from Time import now
from Reductions import sum
from Buffer import Buffer

# Simple array struct
struct ArrayInput:
    var data: DTypePointer[DType.float32]

    fn __init__(inout self, size: Int):
        self.data = DTypePointer[DType.float32].alloc(size);
        rand(self.data, size)

    fn __del__(owned self):
        self.data.free()

    @always_inline
    fn __getitem__(self, x: Int) -> Float32:
        return self.data.load(x)

# Use the https://en.wikipedia.org/wiki/Kahan_summation_algorithm
# Simple summation of the array elements
fn reduce_sum_naive(data: ArrayInput, size: Int) -> Float32:
    var sum = data[0]
    var c : Float32 = 0.0
    for i in range(size):
        var y = data[i] - c
        var t = sum + y
        c = (t - sum) - y
        sum = t
    return sum

fn benchmark_naive_reduce_sum(size: Int) -> Float32:
    print("Computing reduction sum for array num elements: ", size)
    var A = ArrayInput(size)
    # Prevent DCE
    var mySum: Float32 = 0.0

    @always_inline
    @parameter
    fn test_fn():
        try:
            _ = reduce_sum_naive(A, size)
        except:
            pass

    let bench_time = Float64(Benchmark().run[test_fn]())
    return mySum

fn benchmark_stdlib_reduce_sum(size: Int) -> Float32:
    # Allocate a Buffer and then use the Mojo stdlib Reduction class
    # TODO: Use globals
    #alias numElem = size
    alias numElem = 1 << 30
    # Can use either stack allocation or heap
    # see stackalloc
    # var A = Buffer[numElem, DType.float32].stack_allocation()
    # see heapalloc
    var B = DTypePointer[DType.float32].alloc(numElem)
    var A = Buffer[numElem, DType.float32](B)

    # initialize buffer
    for i in range(numElem):
        A[i] = Float32(i)

    # Prevent DCE
    var mySum : Float32 = 0.0
    print("Computing reduction sum for array num elements: ", size)

    @always_inline
    @parameter
    fn test_fn():
        try:
            mySum = sum[numElem, DType.float32](A)
        except:
            pass

    let bench_time = Float64(Benchmark().run[test_fn]())
    return mySum


fn main():
    var size = 1 << 30
    var eval_begin : Float64 = now()
    var sum = benchmark_naive_reduce_sum(size)
    var eval_end : Float64 = now()
    var execution_time = Float64((eval_end - eval_begin)) / 1e6
    print("Completed naive reduction sum: ", sum, " in ", execution_time, "ms")

    eval_begin = now()
    sum = benchmark_stdlib_reduce_sum(size)
    eval_end = now()
    execution_time = Float64((eval_end - eval_begin)) / 1e6
    print("Completed stdlib reduction sum: ", sum, " in ", execution_time, "ms")
