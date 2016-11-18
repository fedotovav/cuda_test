#include "base.h"

using namespace std;

template< typename T >
double measure_cpu( T * arr, size_t size, size_t op_num )
{
   size_t   idx1 = rand() % size
          , idx2 = rand() % size
          , idx3 = rand() % size;
   
   time_res_t time;
   
   time.measure_start();

   for (size_t i = 0; i < op_num; ++i)
      arr[idx3] = arr[idx1] * arr[idx2];
   
   time.cpu_comp_time_ = time.measure_finish();
   
   return time.cpu_comp_time_;
}

template< typename T >
double measure_cpu_parallel( T * arr, size_t size, size_t op_num, size_t threads_num )
{
   size_t   idx1 = rand() % size
          , idx2 = rand() % size
          , idx3 = rand() % size;
   
   time_res_t time;
   
   time.measure_start();

   omp_set_num_threads(threads_num);
   
   #pragma omp parallel for
   for (size_t i = 0; i < op_num; ++i)
      arr[idx3] = arr[idx1] * arr[idx2];
   
   time.cpu_comp_time_ = time.measure_finish();
   
   return time.cpu_comp_time_;
}

int main( int argc, char ** argv )
{
   const size_t   test_mem_max_size = 100 // in mb
                , measure_num = 50
                , precision = 30;

   const size_t   test_elem_max_num = test_mem_max_size * (1048576 / sizeof(double))
                , op_num = test_elem_max_num
                , gpu_part_step = op_num / (measure_num - 1)
                , gpu_part_mem_step = test_elem_max_num / (measure_num - 1);
   
   cout << "START TEST" << endl << endl;
   
   double * data = new double[test_elem_max_num];
   
   fill_array(test_elem_max_num, &data);
   
   vector<double> gpu_part_arr, cpu_calc_time_arr, cpu_parallel_calc_time_arr, gpu_clac_time_arr, gpu_parallel_calc_time_arr, gpu_parallel_calc_no_optimize_time_arr;
   
   size_t   cpu_threads_max_num = omp_get_max_threads()
          , cuda_core_num = get_cuda_cores_num();
   
   cout << "CPU cores number: " << cpu_threads_max_num << endl;
   cout << "GPU cores number: " << cuda_core_num << endl;
   
   cout << "--measure calculation performance--" << endl;
   
   for (size_t gpu_part = 0; gpu_part <= op_num; gpu_part += gpu_part_step)
   {
      double   cpu_calc_time = 0
             , cpu_parallel_calc_time = 0
             , gpu_clac_time = 0
             , gpu_parallel_calc_time = 0
             , gpu_parallel_calc_no_optimize_time = 0;
      
      size_t cpu_part = op_num - gpu_part;
      
      for (size_t i = 0; i < precision; ++i)
      {
         cpu_calc_time                      += measure_cpu(data, test_elem_max_num, cpu_part);
         cpu_parallel_calc_time             += measure_cpu_parallel(data, test_elem_max_num, cpu_part, cpu_threads_max_num);
         gpu_clac_time                      += measure_calc_time_gpu(data, gpu_part);
         gpu_parallel_calc_time             += measure_parallel_calc_time_gpu(data, gpu_part, gpu_part);
         gpu_parallel_calc_no_optimize_time += measure_parallel_calc_no_optimize_time_gpu(data, test_elem_max_num, gpu_part, gpu_part);
      }
      
      cpu_calc_time                      /= (double)precision;
      cpu_parallel_calc_time             /= (double)precision;
      gpu_clac_time                      /= (double)precision;
      gpu_parallel_calc_time             /= (double)precision;
      gpu_parallel_calc_no_optimize_time /= (double)precision;
      
      cpu_calc_time_arr.push_back(cpu_calc_time);
      cpu_parallel_calc_time_arr.push_back(cpu_parallel_calc_time);
      gpu_clac_time_arr.push_back(gpu_clac_time);
      gpu_parallel_calc_time_arr.push_back(gpu_parallel_calc_time);
      gpu_parallel_calc_no_optimize_time_arr.push_back(gpu_parallel_calc_no_optimize_time);
      
      gpu_part_arr.push_back((double)gpu_part / (double)op_num);
      
      cout << (double)gpu_part / (double)op_num << " " << cpu_calc_time << " " << cpu_parallel_calc_time << " " << gpu_clac_time << " " << gpu_parallel_calc_time
           << " " << gpu_parallel_calc_no_optimize_time << endl;
   }
   
   vector<double> gpu_mem_time_arr, data_size_arr;
   
   cout << "--measure memory operations performance--" << endl;
   
   for (size_t gpu_part = 0; gpu_part <= test_elem_max_num; gpu_part += gpu_part_mem_step)
   {
      double   gpu_mem_time = 0
             , data_size;
      
      for (size_t i = 0; i < precision; ++i)
         gpu_mem_time += measure_mem_gpu(data, gpu_part);
      
      gpu_mem_time /= (double)precision;
      
      data_size = 1048576 * (double)test_mem_max_size * ((double)gpu_part / (double)test_elem_max_num);
      
      gpu_mem_time_arr.push_back(gpu_mem_time);
      data_size_arr.push_back(data_size);
      
      cout << data_size << " " << gpu_mem_time << endl;
   }
   
   cout << "--measure CPU parallelization efficiency--" << endl;
   
   ofstream cpu_parallel_efficiency_file("cpu_parallel_efficiency.plt");
   
   cpu_parallel_efficiency_file << "Variables = \"threads_num\", \"efficiency\", \"time\", \"parallel_time\"" << endl;

   double cpu_calc_time = 0;

   for (size_t i = 0; i < precision; ++i)
      cpu_calc_time += measure_cpu_parallel(data, test_elem_max_num, op_num, 1);

   cpu_calc_time /= (double)precision;

   for (size_t cur_threads_num = 1; cur_threads_num < cpu_threads_max_num + 1; ++cur_threads_num)
   {
      double cpu_parallel_calc_time = 0;
      
      for (size_t i = 0; i < precision; ++i)
         cpu_parallel_calc_time += measure_cpu_parallel(data, test_elem_max_num, op_num, cur_threads_num);

      cpu_parallel_calc_time /= (double)precision;
         
      cpu_parallel_efficiency_file << cur_threads_num << " " << cpu_calc_time / ((double)cur_threads_num * cpu_parallel_calc_time) << " " << cpu_calc_time << " " << cpu_parallel_calc_time << endl;

      cout << cur_threads_num << " " << cpu_calc_time / ((double)cur_threads_num * cpu_parallel_calc_time) << " " << cpu_calc_time << " " << cpu_parallel_calc_time << endl;
   }   

   cout << "--measure GPU parallelization efficiency--" << endl;
   
   ofstream gpu_parallel_efficiency_file("gpu_parallel_efficiency.plt");
   
   gpu_parallel_efficiency_file << "Variables = \"threads_num\", \"efficiency\", \"no_optimize_efficiency\", \"time\", \"parallel_time\", \"no_optimize_parallel_time\"" << endl;
   
   double gpu_calc_time = 0;
   
   for (size_t i = 0; i < precision; ++i)
      gpu_calc_time += measure_calc_time_gpu(data, op_num);
   
   gpu_calc_time /= (double)precision;

   for (size_t cur_threads_num = 1; cur_threads_num < cuda_core_num; cur_threads_num += (cuda_core_num / 10))
   {
      double   gpu_parallel_calc_time = 0
             , gpu_parallel_calc_no_optimize_time = 0;
      
      for (size_t i = 0; i < precision; ++i)
      {
         gpu_parallel_calc_time             += measure_parallel_calc_time_gpu(data, op_num, cur_threads_num);
         gpu_parallel_calc_no_optimize_time += measure_parallel_calc_no_optimize_time_gpu(data, test_elem_max_num, op_num, cur_threads_num);
      }

      gpu_parallel_calc_time /= (double)precision;
      gpu_parallel_calc_no_optimize_time /= (double)precision;
         
      gpu_parallel_efficiency_file << cur_threads_num << " " << gpu_calc_time / ((double)cur_threads_num * gpu_parallel_calc_time)
                                   << " " << gpu_calc_time / ((double)cur_threads_num * gpu_parallel_calc_no_optimize_time)
                                   << " " << gpu_calc_time << " " << gpu_parallel_calc_time << " " << gpu_parallel_calc_no_optimize_time << endl;

      cout << cur_threads_num << " " << gpu_calc_time / ((double)cur_threads_num * gpu_parallel_calc_time)
           << " " << gpu_calc_time / ((double)cur_threads_num * gpu_parallel_calc_no_optimize_time)
           << " " << gpu_calc_time << " " << gpu_parallel_calc_time << " " << gpu_parallel_calc_no_optimize_time << endl;
   }   

   cout << "--export resutl--" << endl;

   ofstream calc_res("calc_result.plt");
   
   calc_res << "Variables = \"gpu_part\", \"cpu_calc_time\", \"gpu_clac_time\", \"cpu_parallel_calc_time\", \"gpu_parallel_calc_time\", \"gpu_parallel_calc_no_optimize_time\"" << endl;
   
   for (size_t i = 0; i < gpu_part_arr.size(); ++i)
      calc_res << gpu_part_arr[i] << " " << cpu_calc_time_arr[i] << " " << gpu_clac_time_arr[i] << " " << cpu_parallel_calc_time_arr[i]
               << " " << gpu_parallel_calc_time_arr[i] << " " << gpu_parallel_calc_no_optimize_time_arr[i] << endl;
   
   ofstream mem_res("mem_res.plt");
   
   mem_res << "Variables = \"data_size\", \"gpu_mem_time\"" << endl;
   
   for (size_t i = 0; i < gpu_part_arr.size(); ++i)
      mem_res << data_size_arr[i] << " " << gpu_mem_time_arr[i] << endl;

   postprocessing(gpu_part_arr, cpu_calc_time_arr, cpu_parallel_calc_time_arr, gpu_clac_time_arr, gpu_parallel_calc_time_arr, gpu_parallel_calc_no_optimize_time_arr, gpu_mem_time_arr, data_size_arr, op_num);
   
   delete[] data;
   
   return 0;
}
