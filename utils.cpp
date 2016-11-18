#include <omp.h>

#include "base.h"

template< typename T >
void fill_array_core( size_t size, T ** arr )
{
   size_t threads_num = omp_get_max_threads();
   
   #pragma omp parallel for num_threads(threads_num) shared(arr)
   for (size_t i = 0; i < size; ++i)
      (*arr)[i] = (size - 1) * (double)rand() / (double)RAND_MAX;
}

void fill_array( size_t size, int ** arr )
{
   fill_array_core(size, arr);
}

void fill_array( size_t size, float ** arr )
{
   fill_array_core(size, arr);
}

void fill_array( size_t size, double ** arr )
{
   fill_array_core(size, arr);
}

void fill_array( size_t size, unsigned int ** arr )
{
   fill_array_core(size, arr);
}

void perform_time( string file_name_preffix, double transfer_calc_ratio, double gpu_mem_time, vector<double> & time, vector<double> const & cpu_calc_time_arr, vector<double> const & gpu_clac_time_arr, vector<double> const & gpu_part_arr, vector<double> & new_gpu_part_arr )
{
   stringstream file_name;

   file_name << file_name_preffix << "_" << transfer_calc_ratio << ".plt";

   ofstream out(file_name.str().c_str());

   out << "Variables = \"gpu_part\", \"time\"" << endl;

   size_t measure_num = cpu_calc_time_arr.size();
   
   double   cur_time = 0
          , cur_cpu_time = 0
          , cur_gpu_time = 0
          , prev_cpu_time = 0
          , prev_gpu_time = 0
          , cur_sign = 0
          , prev_sign = 0;
   
   for (size_t i = 0; i < measure_num; ++i)
   {
      cur_cpu_time = cpu_calc_time_arr[i];
      cur_gpu_time = gpu_clac_time_arr[i] + gpu_mem_time * gpu_part_arr[i];
      
      cur_sign = cur_cpu_time - cur_gpu_time;
      
      if ((cur_sign * prev_sign < 0) && (i > 0))
      {
         double gpu_part_diff = gpu_part_arr[i] - gpu_part_arr[i - 1];
         
         cur_time = ((prev_gpu_time / (cur_gpu_time - prev_gpu_time)) - (prev_cpu_time / (cur_cpu_time - prev_cpu_time))) / (1. / (cur_gpu_time - prev_gpu_time) - 1. / (cur_cpu_time - prev_cpu_time));

         time.push_back(cur_time);
         
         double gpu_part = gpu_part_arr[i - 1] + (cur_time - prev_gpu_time) * gpu_part_diff / (cur_gpu_time - prev_gpu_time);
         
         new_gpu_part_arr.push_back(gpu_part);

         out << gpu_part << " " << cur_time << endl;
      }

      cur_time = max(cur_cpu_time, cur_gpu_time);
      
      time.push_back(cur_time);
      new_gpu_part_arr.push_back(gpu_part_arr[i]);
      
      prev_cpu_time = cur_cpu_time;
      prev_gpu_time = cur_gpu_time;
      prev_sign = cur_sign;

      out << gpu_part_arr[i] << " " << cur_time << endl;
   }
}

void perform_axel( string file_name_preffix, double transfer_calc_ratio, vector<double> const & time, vector<double> const & gpu_part_arr, vector<double> & axel )
{
   stringstream file_name;

   file_name << file_name_preffix << "_" << transfer_calc_ratio << ".plt";

   ofstream out(file_name.str().c_str());

   out << "Variables = \"gpu_part\", \"axel\"" << endl;
   
   size_t measure_num = gpu_part_arr.size();

   for (size_t i = 0; i < measure_num; ++i)
   {
      axel.push_back(time[0] / time[i]);

      out << gpu_part_arr[i] << " " << axel[i] << endl;
   }
}

void postprocessing( vector<double> & gpu_part_arr, vector<double> & cpu_calc_time_arr, vector<double> & cpu_parallel_calc_time_arr
                    ,vector<double> & gpu_clac_time_arr, vector<double> & gpu_parallel_calc_time_arr, vector<double> & gpu_parallel_calc_no_optimize_time_arr
                    ,vector<double> & gpu_mem_time_arr, vector<double> & data_size_arr, size_t op_num )
{
   size_t measure_num = gpu_part_arr.size();
   
   vector<double>   axel_by_transfer_calc_ratio(data_size_arr.size())
                  , transfer_calc_ratio_arr(data_size_arr.size())
                  , parallel_axel_by_transfer_calc_ratio(data_size_arr.size())
                  , no_opt_parallel_axel_by_transfer_calc_ratio(data_size_arr.size())
                  , global_gpu_part(data_size_arr.size())
                  , global_parallel_gpu_part(data_size_arr.size())
                  , no_opt_global_parallel_gpu_part(data_size_arr.size());
   
   for (size_t i = 0; i < data_size_arr.size(); ++i)
   {
      double transfer_calc_ratio = (double)data_size_arr[i] / (double)op_num;
      
      vector<double> time, time_gpu_part_arr;
      
      perform_time("trash/time", transfer_calc_ratio, gpu_mem_time_arr[i], time, cpu_calc_time_arr, gpu_clac_time_arr, gpu_part_arr, time_gpu_part_arr);
      
      vector<double> parallel_time, parallel_time_gpu_part_arr;

      perform_time("trash/parallel_time", transfer_calc_ratio, gpu_mem_time_arr[i], parallel_time
                            ,cpu_parallel_calc_time_arr, gpu_parallel_calc_time_arr, gpu_part_arr, parallel_time_gpu_part_arr);

      vector<double> no_optimize_parallel_time, no_parallel_time_gpu_part_arr;
      
      perform_time("trash/no_parallel_time", transfer_calc_ratio, gpu_mem_time_arr[i], no_optimize_parallel_time
                   ,cpu_parallel_calc_time_arr, gpu_parallel_calc_no_optimize_time_arr, gpu_part_arr, no_parallel_time_gpu_part_arr);
      
      vector<double> axel;
      
      perform_axel("trash/axel", transfer_calc_ratio, time, time_gpu_part_arr, axel);
            
      vector<double> parallel_axel;
      
      perform_axel("trash/parallel_axel", transfer_calc_ratio, parallel_time, parallel_time_gpu_part_arr, parallel_axel);

      vector<double> no_optimize_parallel_axel;
      
      perform_axel("trash/no_parallel_axel", transfer_calc_ratio, no_optimize_parallel_time, no_parallel_time_gpu_part_arr, no_optimize_parallel_axel);

      double axel_max = axel[0];
      
      for (size_t j = 0; j < time_gpu_part_arr.size(); ++j)
         if (axel_max < axel[j])
         {
            axel_max = axel[j];
            global_gpu_part[i] = time_gpu_part_arr[j];
         }
      
      axel_by_transfer_calc_ratio[i] = axel_max;

      double parallel_axel_max = parallel_axel[0];
      
      for (size_t j = 0; j < parallel_time_gpu_part_arr.size(); ++j)
         if (parallel_axel_max < parallel_axel[j])
         {
            parallel_axel_max = parallel_axel[j];
            global_parallel_gpu_part[i] = parallel_time_gpu_part_arr[j];
         }

      parallel_axel_by_transfer_calc_ratio[i] = parallel_axel_max;

      double no_opt_parallel_axel_max = no_optimize_parallel_axel[0];

      for (size_t j = 0; j < no_parallel_time_gpu_part_arr.size(); ++j)
         if (no_opt_parallel_axel_max < no_optimize_parallel_axel[j])
         {
            no_opt_parallel_axel_max = no_optimize_parallel_axel[j];
            no_opt_global_parallel_gpu_part[i] = no_parallel_time_gpu_part_arr[j];
         }

      no_opt_parallel_axel_by_transfer_calc_ratio[i] = no_opt_parallel_axel_max;
      
      transfer_calc_ratio_arr[i] = transfer_calc_ratio;
   }
   
   ofstream axel_out("global_axel.plt");

   axel_out << "Variables = \"transfer_calc_ratio\", \"axel\"" << endl;

   for (size_t i = 0; i < data_size_arr.size(); ++i)
      axel_out << transfer_calc_ratio_arr[i] << " " << axel_by_transfer_calc_ratio[i] << endl;

   ofstream parallel_axel_out("global_parallel_axel.plt");

   parallel_axel_out << "Variables = \"transfer_calc_ratio\", \"axel\", \"no_opt_axel\"" << endl;

   for (size_t i = 0; i < data_size_arr.size(); ++i)
      parallel_axel_out << transfer_calc_ratio_arr[i] << " " << parallel_axel_by_transfer_calc_ratio[i] << " " << no_opt_parallel_axel_by_transfer_calc_ratio[i] << endl;

   ofstream gpu_part_out("gpu_part_axel.plt");

   gpu_part_out << "Variables = \"transfer_calc_ratio\", \"gpu_part\"" << endl;

   for (size_t i = 0; i < data_size_arr.size(); ++i)
      gpu_part_out << transfer_calc_ratio_arr[i] << " " << global_gpu_part[i] << endl;

   ofstream parallel_gpu_part_out("parallel_gpu_part_axel.plt");

   parallel_gpu_part_out << "Variables = \"transfer_calc_ratio\", \"gpu_part\", \"no_opt_gpu_part\"" << endl;

   for (size_t i = 0; i < data_size_arr.size(); ++i)
      parallel_gpu_part_out << transfer_calc_ratio_arr[i] << " " << global_parallel_gpu_part[i] << " " << no_opt_global_parallel_gpu_part[i] << endl;
}
