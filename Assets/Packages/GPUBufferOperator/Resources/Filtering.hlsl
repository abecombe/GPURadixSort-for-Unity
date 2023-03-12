#ifndef _CS_FILTERING_HLSL_
#define _CS_FILTERING_HLSL_

#define NUM_GROUP_THREADS 128

StructuredBuffer<DATA_TYPE> data_in_buffer;
RWStructuredBuffer<DATA_TYPE> data_out_buffer;

RWStructuredBuffer<uint> group_sum_buffer;
RWStructuredBuffer<uint> global_prefix_sum_buffer;

int num_elements;
int num_groups;
int group_offset;

static const int num_elements_per_group = NUM_GROUP_THREADS;
static const int log_num_elements_per_group = log2(num_elements_per_group);
static const int num_elements_per_group_1 = num_elements_per_group - 1;

static const int s_data_len = num_elements_per_group;
static const int s_scan_len = num_elements_per_group;

groupshared DATA_TYPE s_data[s_data_len];
groupshared uint s_scan[s_scan_len];

// sort input data locally and output the number of true elements within groups
[numthreads(NUM_GROUP_THREADS, 1, 1)]
void RadixSortLocal(int group_thread_id : SV_GroupThreadID, int group_id : SV_GroupID)
{
    int global_id = num_elements_per_group * (group_id + group_offset) + group_thread_id;
    
    // extract key
    DATA_TYPE data = (DATA_TYPE)0;
    bool key = false;
    if (global_id < num_elements)
    {
        data = data_in_buffer[global_id];
        key = GET_KEY(data);
    }

    // build scan data
    s_scan[group_thread_id] = key;
    GroupMemoryBarrierWithGroupSync();

    // Hillis-Steele scan
    uint sum = 0;
    int partner;
    [unroll(log_num_elements_per_group)]
    for (int offset = 1;; offset <<= 1)
    {
        sum = s_scan[group_thread_id];
        partner = group_thread_id - offset;
        if (partner >= 0)
        {
            sum += s_scan[partner];
        }
        GroupMemoryBarrierWithGroupSync();
        s_scan[group_thread_id] = sum;
        GroupMemoryBarrierWithGroupSync();
    }
    
    uint sum_true = s_scan[num_elements_per_group_1];
    
    if (group_thread_id == 0)
    {
        // copy the number of true elements to global memory
        group_sum_buffer[group_id + group_offset] = sum_true;
        global_prefix_sum_buffer[group_id + group_offset] = sum_true;
    }

    // sort the input data locally
    int new_id = 0;
    if (group_thread_id > 0)
    {
        new_id = s_scan[group_thread_id - 1];
    }
    if (key)
    {
        s_data[new_id] = data;
    }
    GroupMemoryBarrierWithGroupSync();

    // copy sorted input data to global memory
    if ((uint)group_thread_id < sum_true)
    {
        data_out_buffer[global_id] = s_data[group_thread_id];
    }
}

// copy input data to final position in global memory
[numthreads(NUM_GROUP_THREADS, 1, 1)]
void GlobalShuffle(int group_thread_id : SV_GroupThreadID, int group_id : SV_GroupID)
{
    int global_id = num_elements_per_group * (group_id + group_offset) + group_thread_id;

    if ((uint)group_thread_id < group_sum_buffer[group_id + group_offset])
    {
        int new_id = group_thread_id + global_prefix_sum_buffer[group_id + group_offset];
        // copy data to the final destination
        data_out_buffer[new_id] = data_in_buffer[global_id];
    }
}


#endif /* _CS_FILTERING_HLSL_ */