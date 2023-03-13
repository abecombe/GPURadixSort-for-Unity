using System.Runtime.InteropServices;
using UnityEngine;

namespace Abecombe.GPUBufferOperators
{
    public class GPUFiltering<T>
    {
        private static readonly int _numGroupThreads = 128;
        private static readonly int _numElementsPerGroup = _numGroupThreads;

        private static readonly int _max_dispatch_size = 65535;

        protected ComputeShader _filteringCS;
        private int _kernelRadixSortLocal;
        private int _kernelGlobalShuffle;

        private readonly GPUPrefixScan _prefixScan = new();

        // buffer to store the locally sorted input data
        // size: number of data
        private GraphicsBuffer _tempBuffer;
        // buffer to store the number of true elements within locally sorted groups
        // size: number of groups
        private GraphicsBuffer _groupSumBuffer;
        // buffer to store the global prefix sums of true elements within locally sorted groups
        // size: number of groups
        private GraphicsBuffer _globalPrefixSumBuffer;

        private bool _inited = false;

        protected virtual void LoadComputeShader()
        {
            _filteringCS = Resources.Load<ComputeShader>("FilteringCS");
        }

        private void Init()
        {
            if (!_filteringCS) LoadComputeShader();
            _kernelRadixSortLocal = _filteringCS.FindKernel("RadixSortLocal");
            _kernelGlobalShuffle = _filteringCS.FindKernel("GlobalShuffle");

            _inited = true;
        }

        // Gather elements that meet certain condition to the front of the buffer

        // dataBuffer
        // : data<T> buffer to be filtered
        // returnNumTrueElements
        // : whether this function should return the number of true elements (= elements that meet certain condition)
        // return value
        // : the number of true elements (only when returnNumTrueElements is true)
        public uint Filter(GraphicsBuffer dataBuffer, bool returnNumTrueElements = false)
        {
            return Filter(dataBuffer, null, 0, returnNumTrueElements);
        }

        // dataBuffer
        // : data<T> buffer to be filtered
        // numBuffer
        // : data<uint> buffer to store the number of true elements
        // bufferOffset
        // : index of the element in the numBuffer to store the number of true elements
        // returnNumTrueElements
        // : whether this function should return the number of true elements
        // return value
        // : the number of true elements (only when returnNumTrueElements is true)
        public uint Filter(GraphicsBuffer dataBuffer, GraphicsBuffer numBuffer, uint bufferOffset, bool returnNumTrueElements = false)
        {
            if (!_inited) Init();

            var cs = _filteringCS;
            var k_local = _kernelRadixSortLocal;
            var k_shuffle = _kernelGlobalShuffle;

            int numElements = dataBuffer.count;
            int numGroups = (numElements + _numElementsPerGroup - 1) / _numElementsPerGroup;

            CheckBufferSizeChanged(numElements, numGroups);

            cs.SetInt("num_elements", numElements);
            cs.SetInt("num_groups", numGroups);

            // sort input data locally and output the number of true elements within groups
            cs.SetBuffer(k_local, "data_in_buffer", dataBuffer);
            cs.SetBuffer(k_local, "data_out_buffer", _tempBuffer);
            cs.SetBuffer(k_local, "group_sum_buffer", _groupSumBuffer);
            cs.SetBuffer(k_local, "global_prefix_sum_buffer", _globalPrefixSumBuffer);
            for (int i = 0; i < numGroups; i += _max_dispatch_size)
            {
                cs.SetInt("group_offset", i);
                cs.Dispatch(k_local, Mathf.Min(numGroups - i, _max_dispatch_size), 1, 1);
            }

            // prefix scan global group sum data
            uint numTrueElements;
            if (numBuffer != null)
                numTrueElements = _prefixScan.Scan(_globalPrefixSumBuffer, numBuffer, bufferOffset, returnNumTrueElements);
            else
                numTrueElements = _prefixScan.Scan(_globalPrefixSumBuffer, returnNumTrueElements);

            // copy input data to final position in global memory
            cs.SetBuffer(k_shuffle, "data_in_buffer", _tempBuffer);
            cs.SetBuffer(k_shuffle, "data_out_buffer", dataBuffer);
            cs.SetBuffer(k_shuffle, "group_sum_buffer", _groupSumBuffer);
            cs.SetBuffer(k_shuffle, "global_prefix_sum_buffer", _globalPrefixSumBuffer);
            for (int i = 0; i < numGroups; i += _max_dispatch_size)
            {
                cs.SetInt("group_offset", i);
                cs.Dispatch(k_shuffle, Mathf.Min(numGroups - i, _max_dispatch_size), 1, 1);
            }

            return numTrueElements;
        }

        private void CheckBufferSizeChanged(int numElements, int numGroups)
        {
            if (_tempBuffer == null || _tempBuffer.count != numElements)
            {
                _tempBuffer?.Release();
                _tempBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, numElements, Marshal.SizeOf(typeof(T)));
            }
            if (_groupSumBuffer == null || _groupSumBuffer.count != numGroups)
            {
                _groupSumBuffer?.Release();
                _groupSumBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, numGroups, sizeof(uint));
            }
            if (_globalPrefixSumBuffer == null || _globalPrefixSumBuffer.count != numGroups)
            {
                _globalPrefixSumBuffer?.Release();
                _globalPrefixSumBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, numGroups, sizeof(uint));
            }
        }

        public void ReleaseBuffers()
        {
            _tempBuffer?.Release();
            _groupSumBuffer?.Release();
            _globalPrefixSumBuffer?.Release();

            _prefixScan?.ReleaseBuffers();
        }
    }
}