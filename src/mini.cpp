 
 // Includes
 #include  <stdio.h>
 #include  <stdlib.h>
 #include  <iostream>
 #include  <cstring>
 #include  <math.h>
 #include  <time.h>
 #include  <string.h>
 #include  <limits.h>
 
 #include <cuda.h>
 #include <cuda_runtime.h>
 
 using namespace std;
 
 #include <omp.h>
 
 #include "utils/CTimer.h"
 #include "utils/CTimerCpu.h"
 #include "frame/CFrame.h"
 #include "awgn_channel/Chanel_AWGN_SIMD.h"
 #include "ber_analyzer/ErrorAnalyzer.h"
 #include "terminal/CTerminal.h"
 
 #include "decoder/CGPU_Decoder_MS_SIMD.h"
 #include "matrix/constantes_gpu.h"
 
 int    QUICK_STOP           =  false;
 bool   BER_SIMULATION_LIMIT =  false;
 double BIT_ERROR_LIMIT      =  1e-7;
 
 
 ////////////////////////////////////////////////////////////////////////////////////
 void show_info();

 #define MAX_THREADS 4
 ////////////////////////////////////////////////////////////////////////////////////
 
 int main(int argc, char* argv[])
 {
	 int p;
	 srand( 0 );
	 printf("LDPC DECODER - Flooding scheduled decoder\n");
	 printf("GENERATED : %s - %s\n", __DATE__, __TIME__);
	 
	 double Eb_N0;
	 double MinSignalToNoise  = 0.50;
	 double MaxSignalToNoise  = 1;
	 double PasSignalToNoise  = 0.10;
	 int    NUMBER_ITERATIONS  = 10;
	 int    STOP_TIMER_SECOND  = -1;
	 bool   QPSK_CHANNEL       = false;
	 bool   Es_N0              = false; // FALSE => MODE Eb_N0
	 int    NB_THREAD_ON_GPU   = 1024;
	 int    FRAME_ERROR_LIMIT  =  200;
	 
	 char  defDecoder[] = "MS";
	 const char* type = defDecoder;
	
	 int NUM_ACTIVE_THREADS = 1;
	 omp_set_num_threads(NUM_ACTIVE_THREADS);
	 
	 cudaSetDevice(0);
	 cudaDeviceSynchronize();
	 
	 for (p=1; p<argc; p++) 
	 {
		 if( strcmp(argv[p], "-min") == 0 ){
			 MinSignalToNoise = atof( argv[p+1] );
			 p += 1;
			 
		 }else if( strcmp(argv[p], "-max") == 0 ){
			MaxSignalToNoise = atof( argv[p+1] );
			 p += 1;
			 
		 }else if( strcmp(argv[p], "-pas") == 0 ){
			 PasSignalToNoise = atof( argv[p+1] );
			 p += 1;
			 
		 }else if( strcmp(argv[p], "-timer") == 0 ){
			 STOP_TIMER_SECOND = atoi( argv[p+1] );
			 p += 1;
			 
		 }else if( strcmp(argv[p], "-iter") == 0 ){
			 NUMBER_ITERATIONS = atoi( argv[p+1] );
			 p += 1;
			 
		 }else if( strcmp(argv[p], "-fer") == 0 ){
			 FRAME_ERROR_LIMIT = atoi( argv[p+1] );
			 p += 1;
			 
		 }else if( strcmp(argv[p], "-qef") == 0 ){
			 BER_SIMULATION_LIMIT =  true;
			 BIT_ERROR_LIMIT      = ( atof( argv[p+1] ) );
			 p += 1;
			 
		 }else if( strcmp(argv[p], "-bpsk") == 0 ){
			 QPSK_CHANNEL = false;
			 
		 }else if( strcmp(argv[p], "-qpsk") == 0 ){
			 QPSK_CHANNEL = true;
			 
		 }else if( strcmp(argv[p], "-Eb/N0") == 0 ){
			 Es_N0 = false;
			 
		 }else if( strcmp(argv[p], "-Es/N0") == 0 ){
			 Es_N0 = true;
			 
		 }else if( strcmp(argv[p], "-n") == 0 ){
			 NB_THREAD_ON_GPU = atoi( argv[p+1] );
			 p += 1;
			 
		 }
		 else if (strcmp(argv[p], "-thread") == 0)
		 {
			 int nThreads = atoi(argv[p + 1]);
			 if (nThreads > 4) 
			 {
				 printf(" Number of thread can not be higher than 4 => Using 4 threads.");
				 NUM_ACTIVE_THREADS = 4;
			 } 
			 else if (nThreads < 1) 
			 {
				 printf("Number of thread can be lower than 1 => Using 1 thread.");
				 NUM_ACTIVE_THREADS = 1;
			 } 
			 else 
			 {
				 NUM_ACTIVE_THREADS = nThreads;
			 }
			 omp_set_num_threads(NUM_ACTIVE_THREADS);
			 p += 1;
		 }
		 else if( strcmp(argv[p], "-info") == 0 )
		 {
			 show_info();
			 exit( 0 );
			 
		 }
		 else
		 {
			 printf("(EE) Unknown argument (%d) => [%s]\n", p, argv[p]);
			 exit(0);
		 }
	 }
	 
	 double performance = (double)(NmoinsK)/(double)(_N);
	 printf("Code LDPC (N, K)     : (%d,%d)\n", _N, _K);
	 printf("Performance of Code  : %.3f\n", performance);
	 printf("# ITERATIONs of CODE : %d\n", NUMBER_ITERATIONS);
	 printf("FER LIMIT FOR SIMU   : %d\n", FRAME_ERROR_LIMIT);
	 printf("SIMULATION  RANGE    : [%.2f, %.2f], STEP = %.2f\n", MinSignalToNoise, MaxSignalToNoise, PasSignalToNoise);
	 printf("MODE EVALUATION      : %s\n", ((Es_N0)?"Es/N0":"Eb/N0") );
	 printf("MIN-SUM ALGORITHM    : %s\n", type );
	 printf("FAST STOP MODE       : %d\n", QUICK_STOP);
	 
	 CTimer simu_timer(true);
	
	 CFrame* simu_data[MAX_THREADS];
	for(int i=0;i<MAX_THREADS;i++)
        simu_data[i] = new CFrame(_N, _K, NB_THREAD_ON_GPU);
	 
	 Chanel_AWGN_SIMD* noise[MAX_THREADS];
	 for(int i=0;i<MAX_THREADS;i++)
		noise[i] = new Chanel_AWGN_SIMD(simu_data[i], 4, QPSK_CHANNEL, Es_N0);
	 
	 CGPUDecoder* decoder[MAX_THREADS];
	 for(int i=0;i<MAX_THREADS;i++)
		decoder[i] = new CGPU_Decoder_MS_SIMD( NB_THREAD_ON_GPU, _N, _K, _M );
	 
	 Eb_N0 = MinSignalToNoise;
	 long int fdecoding = 0;
	 
	 ErrorAnalyzer* errCounter;
	 
	 long etime = 0;
	 ErrorAnalyzer errCounters(simu_data[0], FRAME_ERROR_LIMIT, false, false);
	 if(STOP_TIMER_SECOND == -1)
		 while (Eb_N0 <=MaxSignalToNoise)
		 {
			 noise[0]->configure(Eb_N0);
			 
			 CTimer temps_ecoule(true);
			 CTimer term_refresh(true);
			 
			 errCounter = new ErrorAnalyzer(simu_data[0], FRAME_ERROR_LIMIT, true, true);
			 
			 CTerminal terminal(&errCounters, &temps_ecoule, Eb_N0);

			 noise[0]->generate();
			 
			 errCounter->store_enc_bits();

			CTimer essai(true);
			decoder[0]->decode( simu_data[0]->get_t_noise_data(), simu_data[0]->get_t_decode_data(), NUMBER_ITERATIONS, false );
			etime += essai.get_time_ms();
			noise[0]->generate();  
			errCounter->generate();
			fdecoding += 1;
			 
			 errCounters.reset_internals();
			 errCounters.accumulate( errCounter);
			 

			 terminal.final_report();
			 
			 Eb_N0 = Eb_N0 + PasSignalToNoise;
		 }
		 
		////////////////////////////////////////////////////////////////////////////////
		//
		//
		// SECOND EVALUATION OF THE THROUGHPUT WITHOUT ENCODED FRAME REGENERATION
		//
		//
		if( STOP_TIMER_SECOND != -1 )
		{
			int exec = 0;
			const int t_eval = STOP_TIMER_SECOND;
			
			
			//
			// ONE THREAD MODE
			//
			if (NUM_ACTIVE_THREADS == 1) 
			{
				CTimerCpu t_Timer1(true);
				while (t_Timer1.get_time_sec() < t_eval) 
				{
					for (int qq = 0; qq < 20; qq++) 
					{
						decoder[0]->decode(simu_data[0]->get_t_noise_data(), simu_data[0]->get_t_decode_data(), NUMBER_ITERATIONS, true);
						exec += 1;
					}
				}
				t_Timer1.stop();
				float debit = _N * ((exec * NB_THREAD_ON_GPU ) / ((float) t_Timer1.get_time_sec()));
				debit /= 1000000.0f;
				printf("(PERF1) LDPC decoder air throughput = %1.6f Mbps\n", debit);
			}
		//
		// TWO THREAD MODE
		//
			if (NUM_ACTIVE_THREADS == 2) 
			{
				exec = 0;
				omp_set_num_threads(2);
				CTimerCpu t_Timer2(true);

				while (t_Timer2.get_time_sec() < t_eval) 
				{
					const int looper = 20;
					#pragma omp parallel sections
					{
						#pragma omp section
						{
							for (int qq = 0; qq < looper; qq++)
								decoder[0]->decode(simu_data[0]->get_t_noise_data(), simu_data[0]->get_t_decode_data(), NUMBER_ITERATIONS, true);
						}
						#pragma omp section
						{
							for (int qq = 0; qq < looper; qq++)
								decoder[1]->decode(simu_data[1]->get_t_noise_data(), simu_data[1]->get_t_decode_data(), NUMBER_ITERATIONS, true);
						}
					}
					exec += 2 * looper;
				}
				t_Timer2.stop();

				float debit = _N * ((exec * NB_THREAD_ON_GPU) / ((float) t_Timer2.get_time_sec()));
				debit /= 1000000.0f;
				printf("(PERF2) LDPC decoder air throughput = %1.3f Mbps\n", debit);
		}

		//
		// THREE THREAD MODE
		//
		if (NUM_ACTIVE_THREADS == 3) 
		{
			exec = 0;
			omp_set_num_threads(3);
			CTimerCpu t_Timer3(true);

			while (t_Timer3.get_time_sec() < t_eval) 
			{
				const int looper = 20;
				#pragma omp parallel sections
				{
					#pragma omp section
					{
						for (int qq = 0; qq < looper; qq++)
							decoder[0]->decode(simu_data[0]->get_t_noise_data(), simu_data[0]->get_t_decode_data(), NUMBER_ITERATIONS, true);
					}
					#pragma omp section
					{
						for (int qq = 0; qq < looper; qq++)
							decoder[1]->decode(simu_data[1]->get_t_noise_data(), simu_data[1]->get_t_decode_data(), NUMBER_ITERATIONS, true);
					}
					#pragma omp section
					{
						for (int qq = 0; qq < looper; qq++)
							decoder[2]->decode(simu_data[2]->get_t_noise_data(), simu_data[2]->get_t_decode_data(), NUMBER_ITERATIONS, true);
					}
				}
				exec += 3 * looper;
			}
			t_Timer3.stop();

			float debit = _N * ((exec * NB_THREAD_ON_GPU) / ((float) t_Timer3.get_time_sec()));
			debit /= 1000000.0f;
			printf("(PERF4) LDPC decoder air throughput = %1.3f Mbps\n", debit);
		}
			
			
		//
        // FOUR THREAD MODE
        //
        if (NUM_ACTIVE_THREADS == 4) 
		{
            exec = 0;
            omp_set_num_threads(4);
            CTimerCpu t_Timer4(true);

            while (t_Timer4.get_time_sec() < t_eval) 
			{
                const int looper = 20;
                #pragma omp parallel sections
                {
                    #pragma omp section
                    {
                        for (int qq = 0; qq < looper; qq++)
                            decoder[0]->decode(simu_data[0]->get_t_noise_data(), simu_data[0]->get_t_decode_data(), NUMBER_ITERATIONS, true);
                    }
                    #pragma omp section
                    {
                        for (int qq = 0; qq < looper; qq++)
                            decoder[1]->decode(simu_data[1]->get_t_noise_data(), simu_data[1]->get_t_decode_data(), NUMBER_ITERATIONS, true);
                    }
                    #pragma omp section
                        {
                        for (int qq = 0; qq < looper; qq++)
                            decoder[2]->decode(simu_data[2]->get_t_noise_data(), simu_data[2]->get_t_decode_data(), NUMBER_ITERATIONS, true);
                    }
                    #pragma omp section
                    {
                        for (int qq = 0; qq < looper; qq++)
                            decoder[3]->decode(simu_data[3]->get_t_noise_data(), simu_data[3]->get_t_decode_data(), NUMBER_ITERATIONS, true);
                    }
                }
                exec += 4 * looper;
            }
            t_Timer4.stop();

            float debit = _N * ((exec * NB_THREAD_ON_GPU) / ((float) t_Timer4.get_time_sec()));
            debit /= 1000000.0f;
            printf("(PERF4) LDPC decoder air throughput = %1.3f Mbps\n", debit);
        }
	}

return 0;
 }
 
 
void show_info()
{
	 struct cudaDeviceProp devProp;
	 cudaGetDeviceProperties(&devProp, 0);
	 printf("Number of Multi-Processor    : %d\n", devProp.multiProcessorCount);
	 printf("+ totalGlobalMem             : %ld Mo\n", (devProp.totalGlobalMem/1024/1024));
	 printf("+ sharedMemPerBlock          : %ld Ko\n", (devProp.sharedMemPerBlock/1024));
	 printf("+ regsPerBlock               : %d\n", (int)devProp.regsPerBlock);
	 printf("+ warpSize                   : %d\n", (int)devProp.warpSize);
	 printf("+ memoryBusWidth             : %d\n", (int)devProp.memoryBusWidth);
	 printf("+ memoryClockRate            : %d\n", (int)devProp.memoryClockRate);
	 fflush(stdout);
}
