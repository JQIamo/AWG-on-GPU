#include "parameters.h"
unsigned int dynamic_num[4] = {0,0,0,0};
unsigned int static_num[4] = {0,0,2,3};
extern const int lThreadsPerBlock = 256;
extern const unsigned long long  llSamplerate = 280000000;
extern const double ramp_time = 3;
double static_freq[4][4096]={{},
    {},
    {99683000-2500000,99683000+2500000}, // The centering frequency for the 850nm X axis is 99.683 MHz
    {98900000-5000000,98900000,98900000+5000000}}; // The centering frequency for the 850nm Y axis is 98.9 MHz
double destination_freq[4][1024]={{},
    {},
    {},
    {}};
unsigned int tone_count[5];
unsigned int dynamic_tone_count[5];
int dynamic_list[1024] = {};   // Index of tones that are to be moved
int static_list[16384]={}; // Index of tones that are not moved
double amp_list[16384]={};
double power_normalizer[4] = {0,0,0,0};
double frequency_limits[4] = {59e6-5e5,105e6+5e5,80e6-5e5,112e6+5e5};
double new_static_freq[4][4096]={{0},{0},{0},{0}};
int amplitude_limit=450;