#ifndef _parameters_included_
#define _parameters_included_
#define MAX_TWEEZERS_PER_CHANNEL 256
extern unsigned int dynamic_num[4];
extern unsigned int static_num[4];
extern const double ramp_time;
extern bool if_mapped;

extern double static_freq[4][MAX_TWEEZERS_PER_CHANNEL];
extern double new_static_freq[4][MAX_TWEEZERS_PER_CHANNEL];
extern double destination_freq[4][MAX_TWEEZERS_PER_CHANNEL];
extern int dynamic_list[4*MAX_TWEEZERS_PER_CHANNEL];
extern int dynamic_to_all_indexes[4*MAX_TWEEZERS_PER_CHANNEL];   // Index of tones that are to be moved
extern int static_list[4*MAX_TWEEZERS_PER_CHANNEL];
extern const unsigned long long  llSamplerate;
extern const int max_streaming_tweezers;
extern const int lThreadsPerBlock;
extern unsigned int tone_count[5];
extern unsigned int dynamic_tone_count[5];
extern float amp_list[4*MAX_TWEEZERS_PER_CHANNEL];
extern float final_amp_list[4*MAX_TWEEZERS_PER_CHANNEL];
extern double power_normalizer[4];
extern double frequency_limits[4];
extern const int lNumCh;
extern int amplitude_limit[4];
extern double shake_freq;
extern double shake_amp;
extern int maximal_tweezer_number[4];
extern unsigned int dynamic_buffersize;
#define NO_OUTPUT_TESTING ;
#endif