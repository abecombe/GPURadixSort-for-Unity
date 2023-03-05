# GPU Radix Sort for Unity

**GPU Radix Sort (& Prefix Scan) using Compute Shader**

**Based on [Fast 4-way parallel radix sorting on GPUs](https://vgc.poly.edu/~csilva/papers/cgf.pdf)**

**The key type used for sorting is limited to `uint`.**

**No restrictions on input data type or size.**

## Algorithmic complexity
GPURadixSort has **`O(n * s * w)`** complexity  
```text
n : number of data
s : size of data struct
w : number of bits to sort
```

## Usage
### Init
***C# code***
```csharp
RadixSort<uint2> radixSort = new();
```
***CustomDefinition.hlsl***
```text
#define DATA_TYPE uint2  // input data struct
#define GET_KEY(s) s.x   // how to get the key-values used for sorting
```
**`uint2` is an example of a data type & you can change it.**  
**Note that the larger the data struct size, the longer it takes to sort.**

### Sort
```csharp
radixSort.Sort(GraphicsBuffer DataBuffer, uint MaxValue);
```
* **DataBuffer**  
  * input data buffer to be sorted

* **MaxValue**  
  * maximum key-value  
  * **since this variable directly related to the algorithmic complexity, passing this argument will reduce the cost of sorting.**

### Dispose
```csharp
void OnDestroy() {
  radixSort.ReleaseBuffers();
}
```

## References
* **[Fast 4-way parallel radix sorting on GPUs](https://vgc.poly.edu/~csilva/papers/cgf.pdf)**  
* **[Chapter 39. Parallel Prefix Sum (Scan) with CUDA](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda)**  
* **[GPU Radix Sort](https://github.com/mark-poscablo/gpu-radix-sort)**
