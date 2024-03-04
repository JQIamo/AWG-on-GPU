#include "parameters.h"
unsigned int dynamic_num[4] = {1,2,0,0};
unsigned int static_num[4] = {5,3,0,1};
extern const int lThreadsPerBlock = 256;
extern const unsigned long long  llSamplerate = 280000000;
extern const double ramp_time = 0.1;
double static_freq[4][4096]={{10000000,20000000,30000000,40000000,50000000},
    {100000000,110000000,120000000},
    {},
    {70000000}};
double destination_freq[4][4096]={{100000000},
    {130000000,140000000},
    {},
    {}};
unsigned int tone_count[5];
unsigned int dynamic_tone_count[5];
int dynamic_list[1024] = {0,5,6};   // Index of tones that is to be moved
int static_list[16384]={1,2,3,4,7,8}; // Index of tones that is not moved