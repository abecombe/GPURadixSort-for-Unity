using System.Runtime.InteropServices;
using Unity.Mathematics;
using UnityEngine;

using Random = UnityEngine.Random;

public enum TestCase
{
    RadixSort,
    PrefixScan
}

public class SortSample : MonoBehaviour
{
    [SerializeField]
    private int _numData = 100;
    [SerializeField]
    private uint _randomValueMax = 100;
    [SerializeField]
    private int _randomSeed = 0;
    [SerializeField]
    private TestCase _testCase = TestCase.RadixSort;

    private GraphicsBuffer _dataBuffer;
    private GraphicsBuffer _tempBuffer;

    private GPURadixSort<uint2> _radixSort = new();
    private GPUPrefixScan _prefixScan = new();

    private ComputeShader _copyCS;
    private int _kernel;

    private int _dispatchSize => (_numData + 1023) / 1024;

    private void Start()
    {
        switch (_testCase)
        {
            case TestCase.RadixSort:
                InitSort();
                break;
            case TestCase.PrefixScan:
                InitScan();
                break;
        }
    }

    private void Update()
    {
        switch (_testCase)
        {
            case TestCase.RadixSort:
                Sort();
                break;
            case TestCase.PrefixScan:
                Scan();
                break;
        }
    }

    private void InitSort()
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

        _copyCS.Dispatch(_kernel, _dispatchSize, 1, 1);

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

    private void Sort()
    {
        _copyCS.Dispatch(_kernel, _dispatchSize, 1, 1);

        _radixSort.Sort(_dataBuffer, _randomValueMax);
    }

    private void InitScan()
    {
        _dataBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _numData, sizeof(uint));
        _tempBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _numData, sizeof(uint));

        uint[] dataArr1 = new uint[_numData];
        uint[] dataArr2 = new uint[_numData];

        Random.InitState(_randomSeed);
        uint sum1 = 0;
        for (uint i = 0; i < _numData; i++)
        {
            uint value = (uint)Random.Range(0, (int)_randomValueMax + 1);
            sum1 += value;
            dataArr1[i] = value;
        }
        _tempBuffer.SetData(dataArr1);

        _copyCS = Resources.Load<ComputeShader>("CopyCS");
        _kernel = _copyCS.FindKernel("CopyScanBuffer");

        _copyCS.SetBuffer(_kernel, "scan_data_buffer", _dataBuffer);
        _copyCS.SetBuffer(_kernel, "scan_temp_buffer", _tempBuffer);
        _copyCS.SetInt("num_elements", _numData);

        _copyCS.Dispatch(_kernel, _dispatchSize, 1, 1);

        uint sum2 = _prefixScan.Scan(_dataBuffer, true);

        _dataBuffer.GetData(dataArr2);
        for (int i = 0; i < _numData - 1; i++)
        {
            if (dataArr2[i + 1] - dataArr2[i] != dataArr1[i])
            {
                Debug.LogError("Scanning Failure");
                break;
            }
        }

        if (sum1 != sum2)
        {
            Debug.LogError("Scanning Failure");
        }
    }

    private void Scan()
    {
        _copyCS.Dispatch(_kernel, _dispatchSize, 1, 1);

        _prefixScan.Scan(_dataBuffer);
    }

    private void OnDestroy()
    {
        _radixSort?.ReleaseBuffers();
        _prefixScan?.ReleaseBuffers();
        _dataBuffer?.Release();
        _tempBuffer?.Release();
    }
}
