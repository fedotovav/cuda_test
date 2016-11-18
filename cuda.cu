#include "base.h"

int get_cuda_cores_num()
{
   int dev_id = 0;

   cudaDeviceProp dev_prop;
   cudaError_t error = cudaGetDevice(&dev_id);
   error = cudaGetDeviceProperties(&dev_prop, dev_id);

   int cores = 0;
   int mp = dev_prop.multiProcessorCount;

   switch (dev_prop.major)
   {
      case 2: // Fermi
         if (dev_prop.minor == 1)
            cores = mp * 48;
         else
            cores = mp * 32;

         break;

      case 3: // Kepler
         cores = mp * 192;
         break;

      case 5: // Maxwell
         cores = mp * 128;
         break;

      default:
         cout << "Unknown device type" << endl;
         break;
   }
   
   return cores;
}

__global__ void warm_up_kernel( double * a, double b )
{
   a[0] = b;
}

void warm_up()
{
   double * dev, host;

   double b = 26;

   cudaError_t err;

   err = cudaMalloc((void **)&dev, sizeof(double));
   err = cudaMemcpy(dev, &host, sizeof(double), cudaMemcpyHostToDevice);

   warm_up_kernel<<< dim3(1), dim3(1) >>>(dev, b);

   cudaDeviceSynchronize();

   err = cudaMemcpy(&host, dev, sizeof(double), cudaMemcpyDeviceToHost);

   cudaFree(dev);
}

template<typename T>
__global__ void calc_kernel( unsigned int op_num, T * res, T number )
{
   res[0] = 10e-10;

   for (unsigned int i = 0; i < op_num; ++i)
      res[0] *= number;
}

template<typename T>
__global__ void calc_kernel_parallel( unsigned int op_num, T * res, T number, int op_num_per_thread )
{
   int id = blockIdx.x * blockDim.x + threadIdx.x;

   if (id < op_num)
      for (unsigned int i = 0; i < op_num_per_thread; ++i)
         res[0] *= number;
}

template<typename T>
__global__ void calc_kernel_parallel_no_optimize( unsigned int op_num, T * res, T number, unsigned int rand_idx, int op_num_per_thread )
{
   int id = blockIdx.x * blockDim.x + threadIdx.x;

   if (id < op_num)
      for (unsigned int i = 0; i < op_num_per_thread; ++i)
         res[rand_idx] *= number;
}

template< typename T >
double dev_mem_alloc( T * vec, size_t size, T ** vec_dev )
{
   cudaEvent_t startEvent, stopEvent;

   cudaEventCreate(&startEvent);
   cudaEventCreate(&stopEvent);

   cudaEventRecord(startEvent, 0);

   cudaError error = cudaMalloc((void **)vec_dev, sizeof(T) * size);
   if (error != cudaSuccess) printf("CUDA error: %s, line(%d)\n", cudaGetErrorString(error), __LINE__);

   error = cudaMemcpy(*vec_dev, vec, sizeof(T) * size, cudaMemcpyHostToDevice);
   if (error != cudaSuccess) printf("CUDA error: %s, line(%d)\n", cudaGetErrorString(error), __LINE__);

   cudaEventRecord(stopEvent, 0);
   cudaEventSynchronize(stopEvent);

   float duration;
   cudaEventElapsedTime(&duration, startEvent, stopEvent);

   cudaEventDestroy(startEvent);
   cudaEventDestroy(stopEvent);

   return duration;
}

template<typename T>
double get_solution_from_device( T * vec, size_t size, T ** vec_dev )
{
   cudaError_t error;

   cudaEvent_t startEvent, stopEvent;

   cudaEventCreate(&startEvent);
   cudaEventCreate(&stopEvent);

   cudaEventRecord(startEvent, 0);

   error = cudaMemcpy(vec, *vec_dev, sizeof(T) * size, cudaMemcpyDeviceToHost);
   if (error != cudaSuccess) printf("CUDA error: %s, line(%d)\n", cudaGetErrorString(error), __LINE__);

   cudaFree(*vec_dev);

   cudaEventRecord(stopEvent, 0);
   cudaEventSynchronize(stopEvent);

   float duration;
   cudaEventElapsedTime(&duration, startEvent, stopEvent);

   cudaEventDestroy(startEvent);
   cudaEventDestroy(stopEvent);

   return duration;
}

template< typename T >
double dev_partial_mem_alloc( T * vec, size_t size, T ** vec_dev )
{
   cudaEvent_t startEvent, stopEvent;

   cudaEventCreate(&startEvent);
   cudaEventCreate(&stopEvent);

   cudaEventRecord(startEvent, 0);

   cudaError error = cudaMalloc((void **)vec_dev, sizeof(T) * size);
   if (error != cudaSuccess) printf("CUDA error: %s, line(%d)\n", cudaGetErrorString(error), __LINE__);

   error = cudaMemcpy(*vec_dev, vec, sizeof(T) * size, cudaMemcpyHostToDevice);
   if (error != cudaSuccess) printf("CUDA error: %s, line(%d)\n", cudaGetErrorString(error), __LINE__);

   cudaEventRecord(stopEvent, 0);
   cudaEventSynchronize(stopEvent);

   float duration;
   cudaEventElapsedTime(&duration, startEvent, stopEvent);

   cudaEventDestroy(startEvent);
   cudaEventDestroy(stopEvent);

   return duration;
}

template<typename T>
double get_partial_solution_from_device( T * vec, size_t size, T ** vec_dev )
{
   time_res_t time;

   cudaError_t error;

   cudaEvent_t startEvent, stopEvent;

   cudaEventCreate(&startEvent);
   cudaEventCreate(&stopEvent);

   cudaEventRecord(startEvent, 0);

   error = cudaMemcpy(vec, *vec_dev, sizeof(T) * size, cudaMemcpyDeviceToHost);
   if (error != cudaSuccess) printf("CUDA error: %s, line(%d)\n", cudaGetErrorString(error), __LINE__);

   cudaFree(*vec_dev);

   cudaEventRecord(stopEvent, 0);
   cudaEventSynchronize(stopEvent);

   float duration;
   cudaEventElapsedTime(&duration, startEvent, stopEvent);

   cudaEventDestroy(startEvent);
   cudaEventDestroy(stopEvent);

   return duration;
}

template<typename T>
double partial_mem_test( T * vec, size_t from, size_t to )
{
   warm_up();

   time_res_t time;

   T * vec_dev;

   double duration = 0;

   time.measure_start();

   size_t size = to - from;

   T * data = new T[size];

   for (size_t i = 0; i < size; ++i)
      data[i] = vec[i + from];

   time.gpu_mem_partial_time_ = time.measure_finish();

   duration += dev_partial_mem_alloc(vec, size, &vec_dev);

   duration += get_partial_solution_from_device(vec, size, &vec_dev);

   time.measure_start();

   for (size_t i = 0; i < size; ++i)
      vec[i + from] = data[i];

   delete[] data;

   time.gpu_mem_partial_time_ += time.measure_finish() + duration;

   return time.gpu_mem_partial_time_;
}

template<typename T>
double mem_test( T * vec, size_t size )
{
   warm_up();

   double time = 0;

   T * vec_dev;

   time += dev_mem_alloc(vec, size, &vec_dev);

   time += get_solution_from_device(vec, size, &vec_dev);

   return time;
}

template<typename T>
double calc( T * vec, size_t op_num )
{
   warm_up();

   time_res_t time;

   cudaError_t error;

   cudaEvent_t startEvent, stopEvent;

   float duration;

   T * dev, host;

   T b = 10e-10;

   error = cudaMalloc((void **)&dev, sizeof(double));
   error = cudaMemcpy(dev, &host, sizeof(double), cudaMemcpyHostToDevice);

   cudaEventCreate(&startEvent);
   cudaEventCreate(&stopEvent);

   cudaEventRecord(startEvent, 0);

   error = cudaPeekAtLastError();
   if (error != cudaSuccess) printf("CUDA error: %s, line(%d)\n", cudaGetErrorString(error), __LINE__);

   calc_kernel<T><<<1, 1>>>(op_num, dev, b);

   error = cudaPeekAtLastError();
   if (error != cudaSuccess) printf("CUDA error: %s, line(%d)\n", cudaGetErrorString(error), __LINE__);

   cudaDeviceSynchronize();

   cudaEventRecord(stopEvent, 0);
   cudaEventSynchronize(stopEvent);

   cudaEventElapsedTime(&duration, startEvent, stopEvent);

   cudaFree(dev);

   cudaEventDestroy(startEvent);
   cudaEventDestroy(stopEvent);

   return duration;
}

template<typename T>
double calc_parallel( T * vec, size_t op_num, size_t threads_num )
{
   if (!threads_num)
      return 0;

   cudaEvent_t startEvent, stopEvent;

   time_res_t time;

   T * dev, host;

   T b = 10e-10;

   cudaError_t error;

   error = cudaMalloc((void **)&dev, sizeof(double));
   error = cudaMemcpy(dev, &host, sizeof(double), cudaMemcpyHostToDevice);

   float duration;

   int block_dim = (threads_num > 1024) ? 1024 : threads_num;
   int grid_dim = (int)ceil(threads_num / 1024);
   int op_num_per_thread = op_num / threads_num;

   if (!grid_dim)
      grid_dim = 1;

   cudaEventCreate(&startEvent);
   cudaEventCreate(&stopEvent);

   cudaEventRecord(startEvent, 0);

   error = cudaPeekAtLastError();
   if (error != cudaSuccess) printf("CUDA error: %s, line(%d)\n", cudaGetErrorString(error), __LINE__);

   calc_kernel_parallel<T><<<grid_dim, block_dim>>>(op_num, dev, b, op_num_per_thread);

   error = cudaPeekAtLastError();
   if (error != cudaSuccess) printf("CUDA error: %s, line(%d)\n", cudaGetErrorString(error), __LINE__);

   cudaDeviceSynchronize();

   cudaEventRecord(stopEvent, 0);
   cudaEventSynchronize(stopEvent);

   cudaEventElapsedTime(&duration, startEvent, stopEvent);

   cudaEventDestroy(startEvent);
   cudaEventDestroy(stopEvent);

   return duration;
}

template<typename T>
double calc_parallel_no_optimize( T * vec, size_t size, size_t op_num, size_t threads_num )
{
   if (!threads_num)
      return 0;

   cudaEvent_t startEvent, stopEvent;

   time_res_t time;

   T b = 10e-10;

   cudaError_t error;

   T * vec_dev;

   unsigned int rand_idx = (size - 1) * (double)rand() / (double)RAND_MAX;

   dev_mem_alloc(vec, size, &vec_dev);

   float duration;

   int block_dim = (threads_num > 1024) ? 1024 : threads_num;
   int grid_dim = (int)ceil(threads_num / 1024);
   int op_num_per_thread = op_num / threads_num;

   if (!grid_dim)
      grid_dim = 1;

   cudaEventCreate(&startEvent);
   cudaEventCreate(&stopEvent);

   cudaEventRecord(startEvent, 0);

   error = cudaPeekAtLastError();
   if (error != cudaSuccess) printf("CUDA error: %s, line(%d)\n", cudaGetErrorString(error), __LINE__);

   calc_kernel_parallel_no_optimize<T><<<grid_dim, block_dim>>>(op_num, vec_dev, b, rand_idx, op_num_per_thread);

   error = cudaPeekAtLastError();
   if (error != cudaSuccess) printf("CUDA error: %s, line(%d)\n", cudaGetErrorString(error), __LINE__);

   cudaDeviceSynchronize();

   cudaEventRecord(stopEvent, 0);
   cudaEventSynchronize(stopEvent);

   cudaEventElapsedTime(&duration, startEvent, stopEvent);

   cudaEventDestroy(startEvent);
   cudaEventDestroy(stopEvent);

   get_solution_from_device(vec, size, &vec_dev);

   return duration;
}

template< typename T >
double measure_mem_time( T * vec, size_t size )
{
   int devID = 0;

   cudaError_t error;
   cudaDeviceProp deviceProp;
   error = cudaGetDevice(&devID);

   if (error != cudaSuccess)
   {
      printf("cudaGetDevice returned error code %d, line(%d)\n", error, __LINE__);
   }

   error = cudaGetDeviceProperties(&deviceProp, devID);

   if (deviceProp.computeMode == cudaComputeModeProhibited)
   {
      fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
      return 0;
   }

   if (error != cudaSuccess) printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);

   double duration = mem_test(vec, size);

   cudaDeviceReset();

   error = cudaPeekAtLastError();
   if (error != cudaSuccess) printf("CUDA error code (%d) at line(%d): %s\n", error, __LINE__, cudaGetErrorString(error));

   return duration;
}

template< typename T >
double measure_partial_mem_time( T * vec, size_t from, size_t to )
{
   int devID = 0;

   cudaError_t error;
   cudaDeviceProp deviceProp;
   error = cudaGetDevice(&devID);

   if (error != cudaSuccess)
   {
      printf("cudaGetDevice returned error code %d, line(%d)\n", error, __LINE__);
   }

   error = cudaGetDeviceProperties(&deviceProp, devID);

   if (deviceProp.computeMode == cudaComputeModeProhibited)
   {
      fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
      return 0;
   }

   if (error != cudaSuccess) printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);

   double duration = partial_mem_test(vec, from, to);

   cudaDeviceReset();

   error = cudaPeekAtLastError();
   if (error != cudaSuccess) printf("CUDA error code (%d) at line(%d): %s\n", error, __LINE__, cudaGetErrorString(error));

   return duration;
}

template< typename T >
double measure_calc_time( T * vec, size_t op_num )
{
   int devID = 0;

   cudaError_t error;
   cudaDeviceProp deviceProp;
   error = cudaGetDevice(&devID);

   if (error != cudaSuccess)
   {
      printf("cudaGetDevice returned error code %d, line(%d)\n", error, __LINE__);
   }

   error = cudaGetDeviceProperties(&deviceProp, devID);

   if (deviceProp.computeMode == cudaComputeModeProhibited)
   {
      fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
      return 0;
   }

   if (error != cudaSuccess) printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);

   double duration = calc<T>(vec, op_num);

   cudaDeviceReset();

   error = cudaPeekAtLastError();
   if (error != cudaSuccess) printf("CUDA error code (%d) at line(%d): %s\n", error, __LINE__, cudaGetErrorString(error));

   return duration;
}

template< typename T >
double measure_parallel_calc_time( T * vec, size_t op_num, size_t thread_num )
{
   int devID = 0;

   cudaError_t error;
   cudaDeviceProp deviceProp;
   error = cudaGetDevice(&devID);

   if (error != cudaSuccess)
   {
      printf("cudaGetDevice returned error code %d, line(%d)\n", error, __LINE__);
   }

   error = cudaGetDeviceProperties(&deviceProp, devID);

   if (deviceProp.computeMode == cudaComputeModeProhibited)
   {
      fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
      return 0;
   }

   if (error != cudaSuccess) printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);

   double duration = calc_parallel<T>(vec, op_num, thread_num);

   cudaDeviceReset();

   error = cudaPeekAtLastError();
   if (error != cudaSuccess) printf("CUDA error code (%d) at line(%d): %s\n", error, __LINE__, cudaGetErrorString(error));

   return duration;
}

template< typename T >
double measure_parallel_calc_no_optimize_time( T * vec, size_t size, size_t op_num, size_t thread_num )
{
   int devID = 0;

   cudaError_t error;
   cudaDeviceProp deviceProp;
   error = cudaGetDevice(&devID);

   if (error != cudaSuccess)
   {
      printf("cudaGetDevice returned error code %d, line(%d)\n", error, __LINE__);
   }

   error = cudaGetDeviceProperties(&deviceProp, devID);

   if (deviceProp.computeMode == cudaComputeModeProhibited)
   {
      fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
      return 0;
   }

   if (error != cudaSuccess) printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);

   double duration = calc_parallel_no_optimize<T>(vec, size, op_num, thread_num);

   cudaDeviceReset();

   error = cudaPeekAtLastError();
   if (error != cudaSuccess) printf("CUDA error code (%d) at line(%d): %s\n", error, __LINE__, cudaGetErrorString(error));

   return duration;
}

double measure_mem_gpu( int * vec, size_t size )
{
   return measure_mem_time(vec, size);
}

double measure_mem_gpu( float * vec, size_t size )
{
   return measure_mem_time(vec, size);
}

double measure_mem_gpu( double * vec, size_t size )
{
   return measure_mem_time(vec, size);
}

double measure_mem_gpu( unsigned int * vec, size_t size )
{
   return measure_mem_time(vec, size);
}

double measure_partial_mem_gpu( int * vec, size_t from, size_t to )
{
   return measure_partial_mem_time(vec, from, to);
}

double measure_partial_mem_gpu( float * vec, size_t from, size_t to )
{
   return measure_partial_mem_time(vec, from, to);
}

double measure_partial_mem_gpu( double * vec, size_t from, size_t to )
{
   return measure_partial_mem_time(vec, from, to);
}

double measure_partial_mem_gpu( unsigned int * vec, size_t from, size_t to )
{
   return measure_partial_mem_time(vec, from, to);
}

double measure_calc_time_gpu( int * vec, size_t op_num )
{
   return measure_calc_time(vec, op_num);
}

double measure_calc_time_gpu( float * vec, size_t op_num )
{
   return measure_calc_time(vec, op_num);
}

double measure_calc_time_gpu( double * vec, size_t op_num )
{
   return measure_calc_time(vec, op_num);
}

double measure_calc_time_gpu( unsigned int * vec, size_t op_num )
{
   return measure_calc_time(vec, op_num);
}

double measure_parallel_calc_time_gpu( int * vec, size_t op_num, size_t thread_num )
{
   return measure_parallel_calc_time(vec, op_num, thread_num);
}

double measure_parallel_calc_time_gpu( float * vec, size_t op_num, size_t thread_num )
{
   return measure_parallel_calc_time(vec, op_num, thread_num);
}

double measure_parallel_calc_time_gpu( double * vec, size_t op_num, size_t thread_num )
{
   return measure_parallel_calc_time(vec, op_num, thread_num);
}

double measure_parallel_calc_time_gpu( unsigned int * vec, size_t op_num, size_t thread_num )
{
   return measure_parallel_calc_time(vec, op_num, thread_num);
}

double measure_parallel_calc_no_optimize_time_gpu( int * vec, size_t size, size_t op_num, size_t thread_num )
{
   return measure_parallel_calc_no_optimize_time(vec, size, op_num, thread_num);
}

double measure_parallel_calc_no_optimize_time_gpu( float * vec, size_t size, size_t op_num, size_t thread_num )
{
   return measure_parallel_calc_no_optimize_time(vec, size, op_num, thread_num);
}

double measure_parallel_calc_no_optimize_time_gpu( double * vec, size_t size, size_t op_num, size_t thread_num )
{
   return measure_parallel_calc_no_optimize_time(vec, size, op_num, thread_num);
}

double measure_parallel_calc_no_optimize_time_gpu( unsigned int * vec, size_t size, size_t op_num, size_t thread_num )
{
   return measure_parallel_calc_no_optimize_time(vec, size, op_num, thread_num);
}
