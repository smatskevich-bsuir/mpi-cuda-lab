#ifndef FILE_QUEUE_HPP
#define FILE_QUEUE_HPP

#include <unordered_set>
#include <vector>

std::vector<std::string> get_task_files(const size_t rank, const size_t cluster_size, const std::string& directory, const std::unordered_set<std::string>& allowed_extensions = { ".png", ".bmp", ".jpg", ".jpeg" });

#endif