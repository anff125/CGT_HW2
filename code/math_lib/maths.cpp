#include "maths.hpp"

#include <math.h>
// you can change it in other function
// float ucb_param_C = 1.41421;
float ucb_param_C = 3.0;
// very fast
// using code from __builtin_clzl(unsigned long x)
int fast_log2(unsigned long x) {
    int r = 63;
    if (!(x & 0xFFFFFFFF00000000))
        r -= 32, x <<= 32;
    if (!(x & 0xFFFF000000000000))
        r -= 16, x <<= 16;
    if (!(x & 0xFF00000000000000))
        r -= 8, x <<= 8;
    if (!(x & 0xF000000000000000))
        r -= 4, x <<= 4;
    if (!(x & 0xC000000000000000))
        r -= 2, x <<= 2;
    if (!(x & 0x8000000000000000))
        r -= 1, x <<= 1;
    return r;
}

// very fast UCB!
float fast_UCB(unsigned int win_num, unsigned int node_sample_num, unsigned int total_sample_num) {
    return (float)win_num / (float)node_sample_num + ucb_param_C * sqrt((float)fast_log2(total_sample_num) / (float)node_sample_num);
}
// very fast LCB!
float fast_LCB(unsigned int win_num, unsigned int node_sample_num, unsigned int total_sample_num) {
    return (float)win_num / (float)node_sample_num - ucb_param_C * sqrt((float)fast_log2(total_sample_num) / (float)node_sample_num);
}