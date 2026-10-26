#include "parameters.h"
unsigned int dynamic_num[4] = {0,0,0,0};
unsigned int static_num[4] = {0,0,2,3};
const int lThreadsPerBlock = 256;
const unsigned long long  llSamplerate = 280000000;
const int max_streaming_tweezers = 10;
const double ramp_time = 0.001; // in seconds
const int lNumCh = 4;
bool if_mapped = false;
double static_freq[4][MAX_TWEEZERS_PER_CHANNEL];
double destination_freq[4][MAX_TWEEZERS_PER_CHANNEL];
unsigned int tone_count[5];
unsigned int dynamic_tone_count[5];
unsigned int dynamic_buffersize;
int dynamic_list[4*MAX_TWEEZERS_PER_CHANNEL] = {};   // Index of tones that are to be moved
int dynamic_to_all_indexes[4*MAX_TWEEZERS_PER_CHANNEL] = {};   // Index of tones that are to be moved
int static_list[4*MAX_TWEEZERS_PER_CHANNEL]={}; // Index of tones that are not moved
float amp_list[4*MAX_TWEEZERS_PER_CHANNEL]={};
float final_amp_list[4*MAX_TWEEZERS_PER_CHANNEL]={};
double power_normalizer[4] = {0,0,0,0};
double frequency_limits[4] = {53e6,111e6,74e6,119e6};
double new_static_freq[4][MAX_TWEEZERS_PER_CHANNEL]={{0},{0},{0},{0}};
int amplitude_limit[4]={1500,1500,600, 600}; // 
int maximal_tweezer_number[4] = {7,7,7,7}; // All numbers must be >= 2
double shake_amp = 0.;
double shake_freq = 0.;
