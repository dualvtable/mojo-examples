# Simple reduction operation on a large array of values to produce a single result
# Reductions and scans are common algorithm patterns in parallel computing.

from Benchmark import Benchmark
from DType import DType
from Object import object

struct ArrayInput:
    var data: DTypePointer[DType.float32]

    fn __init__(inout self, size: Int):
        self.data = DTypePointer[DType.float32].alloc(size);
        rand(self.data, size)

# Use the https://en.wikipedia.org/wiki/Kahan_summation_algorithm
fn reduce_cpu(data: ArrayInput, size: Int) -> DType.float32:
    var sum = data[0]
    var c = 0.0
    for i in range(size):
        var y = data[i] - c
        var t = sum + y
        c = (t - sum) - y
        sum = t
    return sum

fn benchmark_reduce(size: Int):
    var A = ArrayInput(size)

    @always_inline
    @parameter
    fn test_fn():
        try:
            _ = reduce_cpu(A, size)
        except:
            pass

    let secs = Float64(Benchmark().run[test_fn]()) / 1_000_000_000
    print("Completed in ", secs)


fn main():
    benchmark_reduce(1 << 10)





    
