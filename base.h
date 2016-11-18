#pragma once

#include <iostream>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <cmath>
#include <sstream>

#include <stdio.h>

#include <omp.h>

#include <cuda.h>

using namespace std;

class time_res_t
{
public:
   time_res_t() :
        gpu_comp_time_(0)
      , gpu_mem_time_ (0)
      , cpu_comp_time_(0)
      , gpu_comp_parallel_time_(0)
      , gpu_mem_partial_time_(0)
   {}
   
   void measure_start ()
   {
      time_start_ = omp_get_wtime();
   }
   
   double measure_finish()
   {
      time_finish_ = omp_get_wtime();

      return (time_finish_ - time_start_) * 1000.;
   }
   
   void clear()
   {
      gpu_comp_time_ = 0;
      gpu_mem_time_ = 0;
      gpu_mem_partial_time_ = 0;
      cpu_comp_time_ = 0;
      gpu_comp_parallel_time_ = 0;
   }
   
   double loop()
   {
      time_finish_ = omp_get_wtime();

      double duration = time_finish_ - time_start_;
      
      time_start_ = omp_get_wtime();

      return duration;
   }
   
   time_res_t & operator+=( const time_res_t & time_res )
   {
      gpu_comp_time_ += time_res.gpu_comp_time_;
      gpu_mem_time_  += time_res.gpu_mem_time_;
      cpu_comp_time_ += time_res.cpu_comp_time_;
      gpu_comp_parallel_time_ += time_res.gpu_comp_parallel_time_;
      gpu_mem_partial_time_ += time_res.gpu_mem_partial_time_;
      
      return *this;
   }

   time_res_t & operator/=( size_t val )
   {
      gpu_comp_time_ /= (double)val;
      gpu_mem_time_  /= (double)val;
      cpu_comp_time_ /= (double)val;
      gpu_comp_parallel_time_ /= (double)val;
      gpu_mem_partial_time_ /= (double)val;
      
      return *this;
   }
   
   double   gpu_comp_time_
          , gpu_comp_parallel_time_
          , gpu_mem_time_
          , gpu_mem_partial_time_
          , cpu_comp_time_;
   
private:
   double time_start_, time_finish_;
};

//////////////////////////////
// externs
//////////////////////////////

extern void fill_array( size_t size, int ** arr );
extern void fill_array( size_t size, float ** arr );
extern void fill_array( size_t size, double ** arr );
extern void fill_array( size_t size, unsigned int ** arr );

extern void postprocessing( vector<double> & gpu_part_arr, vector<double> & cpu_calc_time_arr, vector<double> & cpu_parallel_calc_time_arr
                           ,vector<double> & gpu_clac_time_arr, vector<double> & gpu_parallel_calc_time_arr, vector<double> & gpu_parallel_calc_no_optimize_time_arr
                           ,vector<double> & gpu_mem_time_arr
                           ,vector<double> & data_size_arr, size_t op_num );

extern double measure_mem_gpu( int * vec, size_t size );
extern double measure_mem_gpu( float * vec, size_t size );
extern double measure_mem_gpu( double * vec, size_t size );
extern double measure_mem_gpu( unsigned int * vec, size_t size );

extern double measure_partial_mem_gpu( int * vec, size_t size, size_t from, size_t to );
extern double measure_partial_mem_gpu( float * vec, size_t size, size_t from, size_t to );
extern double measure_partial_mem_gpu( double * vec, size_t size, size_t from, size_t to );
extern double measure_partial_mem_gpu( unsigned int * vec, size_t size, size_t from, size_t to );

extern double measure_calc_time_gpu( int * vec, size_t op_num );
extern double measure_calc_time_gpu( float * vec, size_t op_num );
extern double measure_calc_time_gpu( double * vec, size_t op_num );
extern double measure_calc_time_gpu( unsigned int * vec, size_t op_num );

extern double measure_parallel_calc_time_gpu( int * vec, size_t op_num, size_t thread_num );
extern double measure_parallel_calc_time_gpu( float * vec, size_t op_num, size_t thread_num );
extern double measure_parallel_calc_time_gpu( double * vec, size_t op_num, size_t thread_num );
extern double measure_parallel_calc_time_gpu( unsigned int * vec, size_t op_num, size_t thread_num );

extern double measure_parallel_calc_no_optimize_time_gpu( int * vec, size_t size, size_t op_num, size_t thread_num );
extern double measure_parallel_calc_no_optimize_time_gpu( float * vec, size_t size, size_t op_num, size_t thread_num );
extern double measure_parallel_calc_no_optimize_time_gpu( double * vec, size_t size, size_t op_num, size_t thread_num );
extern double measure_parallel_calc_no_optimize_time_gpu( unsigned int * vec, size_t size, size_t op_num, size_t thread_num );

extern int get_cuda_cores_num();