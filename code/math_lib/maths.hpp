#ifndef __MATHS__
#define __MATHS__ 1
// you can change it in other function
extern float ucb_param_C;
int fast_log2(unsigned long x);
// very fast UCB!
float fast_UCB(unsigned int win_num, unsigned int node_sample_num, unsigned int total_sample_num);
// very fast LCB!
float fast_LCB(unsigned int win_num, unsigned int node_sample_num, unsigned int total_sample_num);
#endif