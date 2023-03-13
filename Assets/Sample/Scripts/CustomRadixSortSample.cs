using Abecombe.GPUBufferOperators;
using System.Runtime.InteropServices;
using UnityEngine;

using Random = UnityEngine.Random;

public struct CustomStruct
{
    public uint key;
    public uint id;
    public float dummy1;
    public float dummy2;

    public CustomStruct(uint key, uint id, float dummy1, float dummy2)
    {
        this.key = key;
        this.id = id;
        this.dummy1 = dummy1;
        this.dummy2 = dummy2;
    }
}

public class CustomRadixSort : GPURadixSort<CustomStruct>
{
    protected override void LoadComputeShader()
    {
        _radixSortCS = Resources.Load<ComputeShader>("CustomRadixSortCS");
    }
}

public class CustomRadixSortSample : MonoBehaviour
{
    [SerializeField]
    private int _numData = 100;
    [SerializeField]
    private uint _randomValueMax = 100;
    [SerializeField]
    private int _randomSeed = 0;

    private readonly CustomRadixSort _radixSort = new();

    private GraphicsBuffer _dataBuffer;
    private GraphicsBuffer _tempBuffer;

    private ComputeShader _copyCS;
    private int _kernel;

    private int DispatchSize => (_numData + 1023) / 1024;

    private void Start()
    {
        _dataBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _numData, Marshal.SizeOf(typeof(CustomStruct)));
        _tempBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _numData, Marshal.SizeOf(typeof(CustomStruct)));

        CustomStruct[] dataArr = new CustomStruct[_numData];

        Random.InitState(_randomSeed);
        for (uint i = 0; i < _numData; i++)
        {
            uint value = (uint)Random.Range(0, (int)_randomValueMax + 1);
            dataArr[i] = new CustomStruct(value, i, 10f, 20f);
        }
        _tempBuffer.SetData(dataArr);

        _copyCS = Resources.Load<ComputeShader>("CopyCS");
        _kernel = _copyCS.FindKernel("CopyCustomSortBuffer");

        _copyCS.SetBuffer(_kernel, "custom_sort_data_buffer", _dataBuffer);
        _copyCS.SetBuffer(_kernel, "custom_sort_temp_buffer", _tempBuffer);
        _copyCS.SetInt("num_elements", _numData);

        _copyCS.Dispatch(_kernel, DispatchSize, 1, 1);

        _radixSort.Sort(_dataBuffer, _randomValueMax);

        _dataBuffer.GetData(dataArr);
        for (int i = 0; i < _numData - 1; i++)
        {
            if (dataArr[i + 1].key < dataArr[i].key)
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
