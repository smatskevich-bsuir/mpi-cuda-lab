#include <dirent.h>	
#include <sys/stat.h>
#include <queue>
#include <algorithm>
#include <filesystem>

#include "../include/file_queue.hpp"

struct file_t {
    std::string name;
    size_t size;
};

auto file_comparator = [](file_t a, file_t b) { return a.size < b.size; };
typedef std::priority_queue<file_t, std::vector<file_t>, decltype(file_comparator)> file_heap_t;

struct node_t {
    size_t idx;
    size_t used_size;
};

auto node_comparator = [](node_t a, node_t b) { return a.used_size > b.used_size; };
typedef std::priority_queue<node_t, std::vector<node_t>, decltype(node_comparator)> node_heap_t;

file_heap_t list_files(const std::string& directory, const std::unordered_set<std::string>& allowed_extensions) 
{
    file_heap_t files(file_comparator);

    for (const auto & file : std::filesystem::directory_iterator(directory))
    {
        std::string path = file.path();
        size_t dot = path.rfind(".");
        if(dot != std::string::npos && allowed_extensions.count(path.substr(dot)))
        {
            struct stat st;
            stat(path.c_str(), &st);
            files.push({path, (size_t)st.st_size});
        }
    }

    return files;
}

std::vector<std::string> get_task_files(const size_t rank, const size_t cluster_size, const std::string& directory, const std::unordered_set<std::string>& allowed_extensions)
{
    file_heap_t files = list_files(directory, allowed_extensions);
    std::vector<std::string> result;

    node_heap_t nodes(node_comparator);

    for(size_t i = 0; i < cluster_size; i++)
        nodes.push({i, 0});

    while(!files.empty())
    {
        file_t file = files.top();
        files.pop();

        node_t node = nodes.top();
        nodes.pop();

        if(node.idx == rank)
            result.push_back(file.name);
        
        node.used_size += file.size;
        nodes.push(node);
    }

    return result;
}