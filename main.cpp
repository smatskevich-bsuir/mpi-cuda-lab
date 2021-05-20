#include <mpi.h>
#include <iostream> 
#include <sstream>
#include <chrono>

#include "include/file_queue.hpp"
#include "include/image_processing.cuh"

typedef std::chrono::time_point<std::chrono::steady_clock> _time_t;

const char* TASK_DIRECTORY = "/cluster/task";

int main (int argc, char* argv[])
{
    int rank;
    int cluster_size;
    _time_t program_start = std::chrono::steady_clock::now();
    MPI_Init (&argc, &argv);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &cluster_size); 

    char proc_name[MPI_MAX_PROCESSOR_NAME];
    int len;
    MPI_Get_processor_name(proc_name,&len);

    std::stringstream info;
    std::vector<std::string> files = get_task_files(rank, cluster_size, TASK_DIRECTORY, {".pgm"});
    info << proc_name << " files:";
    for(auto file: files)
        info  << " " << file;
    info << std::endl;
    std::cout << info.str();

    info.str(std::string());
    info << proc_name << " report: ";
    for(auto file: files)
    {
        _time_t file_start = std::chrono::steady_clock::now();   
        prewittGPU(file);
        _time_t file_end = std::chrono::steady_clock::now(); \
        float time = std::chrono::duration<float>(file_end - file_start).count() * 1000;
        info << file << "(" << time << "ms) ";
    }

    _time_t program_end = std::chrono::steady_clock::now();
	float time = std::chrono::duration<float>(program_end - program_start).count() * 1000;
    info << "full time: " << time << "ms" << std::endl;
    std::cout << info.str();

    MPI_Finalize();
    return 0;
}