#include <cstdio>
#include <cassert>
#include <algorithm>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <chrono>
#include <tuple>
#include <random>
#include <map>
#include <set>
using namespace std;

#define EDGELIST
// read edgelist
vector<pair<int, int>> GetEdgeList(string &input_file_path, int &max_ele)
{
    vector<pair<int, int>> lines;

    ifstream ifs(input_file_path);

    while (ifs.good())
    {
        string tmp_str;
        stringstream ss;
        std::getline(ifs, tmp_str);
        if (!ifs.good())
            break;
        if (tmp_str[0] != '#')
        {
            ss.clear();
            ss << tmp_str;
            int first, second;
            ss >> first >> second;

            if (first > second)
            {
                swap(first, second);
            }
            // 1st case first == second: skip these self loop, (i,i)
            if (first == second)
                continue;
            // 2nd case first > second: unique (i,j), (j,i)

            assert(first < INT32_MAX and second < INT32_MAX);
            if (second > max_ele)
                max_ele = second;
            lines.emplace_back(first, second);
        }
    }
    sort(lines.begin(), lines.end(), [](const pair<int, int> &left, const pair<int, int> &right)
         {
        if (left.first == right.first) {
            return left.second < right.second;
        }
        return left.first < right.first; });
    ifs.close();
    return lines;
}

bool IsAlreadyCSROrder(vector<pair<int, int>> &lines)
{
    int cur_src_vertex = -1;
    int prev_dst_val = -1;
    auto line_count = 0u;
    for (const auto &line : lines)
    {
        int src, dst;
        std::tie(src, dst) = line;
        if (src >= dst)
        {
            cout << "src >= dst"
                 << "\n";
            return false;
        }

        if (src == cur_src_vertex)
        {
            if (dst < prev_dst_val)
            {
                cout << "dst < prev_dst_val"
                     << "\n";
                cout << "cur line:" << line_count << "\n";
                return false;
            }
        }
        else
        {
            cur_src_vertex = src;
        }
        prev_dst_val = dst;
        line_count++;
    }
    return true;
}

void print_arr(vector<uint> degree_arr, vector<vector<uint>> matrix, vector<vector<int>> labels, vector<vector<float>> weights)
{
    cout << "degree_arr: ";
    for (auto i : degree_arr)
        cout << i << " ";
    cout << endl;
    cout << "matrix: " << endl;
    for (auto i : matrix)
    {
        for (auto j : i)
            cout << j << " ";
        cout << endl;
    }
    cout << "labels: " << endl;
    for (auto i : labels)
    {
        for (auto j : i)
            cout << j << " ";
        cout << endl;
    }
    cout << "weights: " << endl;
    for (auto i : weights)
    {
        for (auto j : i)
            cout << j << " ";
        cout << endl;
    }
}

void WriteCSR(string &deg_output_file, string &adj_output_file, string &label_output_file, string &edgelist_output_file, string &weight_output_file, int num_labels,
              vector<pair<int, int>> &lines, int &vertex_num, int &edge_num)
{
    vertex_num = static_cast<unsigned long>(vertex_num + 1);
    // edge_num = lines.size();
    vector<uint> degree_arr(vertex_num, 0);
    vector<vector<uint>> matrix(vertex_num);
    std::vector<vector<int>> labels(vertex_num);
    std::vector<vector<float>> weights(vertex_num);
    // vector<pair<int, int>> nz_list;
    map<uint, uint> vtx_new;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, num_labels - 1);
    // std::uniform_real_distribution<> dis2(0, 5.0);
    std::uniform_real_distribution<> dis2(1.0, 5.0);
    set<pair<uint, uint>> s;

    ofstream deg_ofs(deg_output_file, ios::binary);

    auto vertex_num_nozero = 0;

    for (const auto &line : lines)
    {
        int src, dst;
        std::tie(src, dst) = line;

        if (vtx_new.find(src) == vtx_new.end())
        {
            vtx_new[src] = vertex_num_nozero;
            vertex_num_nozero++;
        }
        if (vtx_new.find(dst) == vtx_new.end())
        {
            vtx_new[dst] = vertex_num_nozero;
            vertex_num_nozero++;
        }
        // degree_arr[vtx_new[src]]++;
        // degree_arr[vtx_new[dst]]++;
        uint new_src = vtx_new[src];
        uint new_dst = vtx_new[dst];
        pair<uint, uint> p1 = make_pair(new_src, new_dst);
        pair<uint, uint> p2 = make_pair(new_dst, new_src);
        if (s.find(p1) != s.end() || s.find(p2) != s.end())
            continue;
        else
        {
            s.insert(p1);
            s.insert(p2);
        }
        matrix[new_src].emplace_back(new_dst);
        matrix[new_dst].emplace_back(new_src);

        int gen_label = dis(gen);
        labels[new_src].emplace_back(gen_label);
        labels[new_dst].emplace_back(gen_label);

        float gen_weight = dis2(gen);
        weights[new_src].emplace_back(gen_weight);
        weights[new_dst].emplace_back(gen_weight);

        // nz_list.emplace_back(vtx_new[src], vtx_new[dst]);
        // matrix[src].emplace_back(dst);
        // matrix[dst].emplace_back(src);
    }
    // printf("vertex_num_nozero=%d\n", vertex_num_nozero);
    degree_arr.resize(vertex_num_nozero + 1);
    matrix.resize(vertex_num_nozero);
    labels.resize(vertex_num_nozero);
    weights.resize(vertex_num_nozero);

    edge_num = 0;
    degree_arr[0] = 0;
    for (int i = 0; i < vertex_num_nozero; i++)
    {
        // erase duplication
        // printf("i=%d,before erase duplication: %d\n", i, matrix[i].size());
        // set<uint> s(matrix[i].begin(), matrix[i].end());
        // matrix[i].assign(s.begin(), s.end());
        edge_num += matrix[i].size();
        degree_arr[i + 1] = matrix[i].size() + degree_arr[i];
    }
    // vertex_num = vertex_num_nozero;

    cout << "begin write" << endl;
    int int_size = sizeof(int);
    // deg_ofs.write(reinterpret_cast<const char *>(&int_size), 4);
    deg_ofs.write(reinterpret_cast<const char *>(&vertex_num_nozero), 4);
    deg_ofs.write(reinterpret_cast<const char *>(&edge_num), 4);
    deg_ofs.write(reinterpret_cast<const char *>(&degree_arr.front()), (vertex_num_nozero + 1) * sizeof(uint));
    deg_ofs.close();

    cout << "finish xadj write..." << endl;

    ofstream adj_ofs(adj_output_file, ios::binary);

    for (auto &adj_arr : matrix)
    {
        adj_ofs.write(reinterpret_cast<const char *>(&adj_arr.front()), adj_arr.size() * 4);
    }
    adj_ofs.close();
    cout << "finish edge write..." << endl;

    ofstream label_ofs(label_output_file, ios::binary);
    // label_ofs.write(reinterpret_cast<const char *>(&labels.front()), labels.size() * 4);
    for (auto &label_arr : labels)
    {
        label_ofs.write(reinterpret_cast<const char *>(&label_arr.front()), label_arr.size() * sizeof(int));
    }
    cout << "finish label write..." << std::endl;
    label_ofs.close();

    ofstream weight_ofs(weight_output_file, ios::binary);
    // weight_ofs.write(reinterpret_cast<const char *>(&weights.front()), weights.size() * sizeof(float));
    for (auto &weight_arr : weights)
    {
        weight_ofs.write(reinterpret_cast<const char *>(&weight_arr.front()), weight_arr.size() * sizeof(float));
    }
    cout << "finish weight write..." << std::endl;
    weight_ofs.close();

    // #ifdef EDGELIST
    ofstream edgelist_ofs(edgelist_output_file);
    for (int i = 0; i < vertex_num_nozero; i++)
    {
        for (auto &j : matrix[i])
        {
            edgelist_ofs << i << " " << j << endl;
        }
    }
    cout << "finish edgelist write..." << std::endl;
    edgelist_ofs.close();
    // cout << "1111111111" << endl;
    // #endif
    // print_arr(degree_arr, matrix, labels, weights);
    // return vertex_num_nozero;
}

int main(int argc, char *argv[])
{
    string input_file_path(argv[1]);
    string output_file_path(argv[2]);
    string num_labels(argv[3]);

    string deg_output_file_path = output_file_path + "_xadj.bin";
    string adj_output_file_path = output_file_path + "_edge.bin";
    string label_output_file_path = output_file_path + "_label.bin";
    string edgelist_output_file_path = output_file_path + ".edgelist";
    string weight_output_file_path = output_file_path + "_weight.bin";
    cout << "input: " << input_file_path << endl;
    int label_num = std::stoi(num_labels);

    int max_ele = -1;
    using namespace std::chrono;
    auto io_start = high_resolution_clock::now();

#ifdef WITHGPERFTOOLS
    cout << "\nwith google perf start\n";
    ProfilerStart("converterProfile.log");
#endif
    auto lines = GetEdgeList(input_file_path, max_ele);

    auto io_end = high_resolution_clock::now();
    cout << "1st, read file and parse string cost:" << duration_cast<milliseconds>(io_end - io_start).count()
         << " ms\n";
#ifdef WITHGPERFTOOLS
    cout << "\nwith google perf end\n";
    ProfilerStop();
#endif
    cout << "max vertex id:" << max_ele << "\n";
    cout << "number of edges:" << lines.size() << "\n";
    // auto check_start = high_resolution_clock::now();
    // if (IsAlreadyCSROrder(lines))
    // {
    // cout << "already csr"
    //      << "\n";
    auto check_end = high_resolution_clock::now();
    // cout << "2nd, check csr representation cost:" << duration_cast<milliseconds>(check_end - check_start).count()
    //      << " ms\n";
    auto vtx_num = max_ele;
    auto edge_num = 0;

    WriteCSR(deg_output_file_path, adj_output_file_path, label_output_file_path, edgelist_output_file_path, weight_output_file_path, label_num, lines, vtx_num, edge_num);
    auto write_end = high_resolution_clock::now();
    cout << "3rd, construct csr and write file cost:" << duration_cast<milliseconds>(write_end - check_end).count()
         << " ms\n";
    // cout << "after convert"
    //      << "\n";
    // cout << "vertex number:" << vtx_num << "\n";
    // cout << "edge number:" << lines.size() << "\n";
    // }
}
