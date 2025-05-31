#include "parameters.h"
unsigned int dynamic_num[4] = {0,0,0,0};
unsigned int static_num[4] = {0,0,2,3};
const int lThreadsPerBlock = 256;
const unsigned long long  llSamplerate = 280000000;
const int max_streaming_tweezers = 3;
const double ramp_time = 3;
const int lNumCh = 4;
// bool if_mapped = true;
bool if_mapped = false;
double static_freq[4][16384];
double destination_freq[4][16384];
unsigned int tone_count[5];
unsigned int dynamic_tone_count[5];
int dynamic_list[16384] = {};   // Index of tones that are to be moved
int static_list[16384]={}; // Index of tones that are not moved
float amp_list[16384]={};
float final_amp_list[16384]={};
double power_normalizer[4] = {0,0,0,0};
double frequency_limits[4] = {0,205e6+5e5,0,212e6+5e5};
double new_static_freq[4][16384]={{0},{0},{0},{0}};
int amplitude_limit[4]={1500,1500,1500,1500};