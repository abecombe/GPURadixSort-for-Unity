using Abecombe.GPUBufferOperator;
using System.Runtime.InteropServices;
using Unity.Mathematics;
using UnityEngine;

using Random = UnityEngine.Random;

public class RadixSortSample : MonoBehaviour
{
    [SerializeField]
    private int _numData = 100;
    [SerializeField]
    private uint _randomValueMax = 100;
    [SerializeField]
    private int _randomSeed = 0;

    private readonly GPURadixSort<uint2> _radixSort = new();

    private GraphicsBuffer _dataBuffer;
    private GraphicsBuffer _tempBuffer;

    private ComputeShader _copyCS;
    private int _kernel;

    private int DispatchSize => (_numData + 1023) / 1024;

    private void Start()
    {
        _dataBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _numData, Marshal.SizeOf(typeof(uint2)));
        _tempBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _numData, Marshal.SizeOf(typeof(uint2)));

        uint2[] dataArr = new uint2[_numData];

        Random.InitState(_randomSeed);
        for (uint i = 0; i < _numData; i++)
        {
            uint value = (uint)Random.Range(0, (int)_randomValueMax + 1);
            dataArr[i] = new uint2(value, i);
        }
        _tempBuffer.SetData(dataArr);

        _copyCS = Resources.Load<ComputeShader>("CopyCS");
        _kernel = _copyCS.FindKernel("CopySortBuffer");

        _copyCS.SetBuffer(_kernel, "sort_data_buffer", _dataBuffer);
        _copyCS.SetBuffer(_kernel, "sort_temp_buffer", _tempBuffer);
        _copyCS.SetInt("num_elements", _numData);

        _copyCS.Dispatch(_kernel, DispatchSize, 1, 1);

        _radixSort.Sort(_dataBuffer, _randomValueMax);

        _dataBuffer.GetData(dataArr);
        for (int i = 0; i < _numData - 1; i++)
        {
            if (dataArr[i + 1].x < dataArr[i].x)
            {
                Debug.LogError("Sorting Failure");
                break;
            }
        }
    }

    private void Update()
    {
        _copyCS.Dispatch(_kernel, DispatchSize, 1, 1);

        _radixSort.Sort(_dataBuffer, _randomValueMax);
    }

    private void OnDestroy()
    {
        _radixSort?.ReleaseBuffers();

        _dataBuffer?.Release();
        _tempBuffer?.Release();
    }
}
