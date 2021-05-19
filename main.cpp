#include <mpi.h>
#include <iostream> 
#include <sstream>

#include "include/file_queue.hpp"
#include "include/image_processing.cuh"

const char* TASK_DIRECTORY = "/cluster/task";

int main (int argc, char* argv[])
{
    int rank;
    int cluster_size;
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

    for(auto file: files)
        prewittGPU(file);

    MPI_Finalize();
    return 0;
}