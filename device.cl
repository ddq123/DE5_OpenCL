//__attribute((num_compute_units(2)))  49  112
__kernel 
__attribute((reqd_work_group_size(112,10,1)))
//__attribute((num_simd_work_items(4)))
void BP_Test_1(	
							__global char *restrict input, 
							__global float *restrict weight1_train, 
							__global float *restrict b1_train,
							__global float *restrict output1
							)

{
	//local storage of input,weight and bias
	__local char  local_input[112];
	__local float local_weight[10][112];
	//__local float local_bias[100];
	
	// Block index
    int block_x = get_group_id(0);//0-7
    int block_y = get_group_id(1);//0-9
	
	// Local ID index (offset within a block)
	int x = get_local_id(0);//0-112 size of one input
	int y = get_local_id(1);//0-9 size of neural
	
	int global_x = get_global_id(0);//0-783 size of one input
	int global_y = get_global_id(1);//0-99 size of neural
	
	// Compute loop bounds
    int a_start = 0;
    int a_end   = 784;
    int b_start = block_y * 10 * 784;
	
	float sigma = 0;
	
	for(int i = 0; i < 10000; i++){
		for(int a = a_start, b = b_start; a < a_end; a += 112, b += 112){
			//load input to local memory 
			local_input[x] = input[i * 784 + a + x];
			local_weight[y][x] = weight1_train[b + y * 784 + x];
			barrier(CLK_LOCAL_MEM_FENCE);

			//
			#pragma unroll
			for(int j = 0; j < 112; j++){
				sigma += local_input[j] * local_weight[y][j];
			}
		}
		
		sigma = sigma + b1_train[block_y * 10 + y];
		barrier(CLK_LOCAL_MEM_FENCE);
		
		output1[i * get_global_size(1) + global_y] = 1.0 / (1.0 + exp(-sigma));
	}
		
}	

//__attribute__((num_compute_units(2)))
__kernel void BP_Test_2(	
							__global float4 *restrict input, 
							__global float4 *restrict weight2_train,
							__global float *restrict b2_train,
							__global float *restrict output2
							 )

{
	float D_local[10];
	unsigned cycle = get_global_id(0);     //0-10000
	unsigned k = get_global_id(1);//0-10
	
	#pragma unroll
	for (unsigned i = 0; i < 10; i++) 
	{
		D_local[i] = b2_train[i];
	}

	input   += cycle*25;
	output2 += cycle*10;
	{
		float sigma = 0;
		for (unsigned j = 0; j < 25; j++)       //   100
		{
			sigma += input[j].s0 * weight2_train[k*25+j].s0;
			sigma += input[j].s1 * weight2_train[k*25+j].s1;
			sigma += input[j].s2 * weight2_train[k*25+j].s2;
			sigma += input[j].s3 * weight2_train[k*25+j].s3;
		}
		float x = sigma + D_local[k];
		output2[k] = x;
	}
}
