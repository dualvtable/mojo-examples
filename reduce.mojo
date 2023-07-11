# Simple reduction operation on a large array of values to produce a single result
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
fn reduce_cpu(data: ArrayInput, size: Int) -> Float32:
    var sum = data[0]
    var c : Float32 = 0.0
    for i in range(size):
        var y = data[i] - c
        var t = sum + y
        c = (t - sum) - y
        sum = t
    return sum

fn benchmark_reduce(size: Int):
    var A = ArrayInput(size)

    var eval_begin : Float64 = now()

    @always_inline
    @parameter
    fn test_fn():
        try:
            _ = reduce_cpu(A, size)
        except:
            pass

    let bench_time = Float64(Benchmark().run[test_fn]())
    var eval_end : Float64 = now()
    let execution_time = Float64((eval_end - eval_begin)) / 1e6
    print("Completed in ", execution_time, "ms")


fn main():
    benchmark_reduce(1 << 32)
