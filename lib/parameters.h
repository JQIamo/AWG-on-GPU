#ifndef _parameters_included_
#define _parameters_included_
extern unsigned int dynamic_num[4];
extern unsigned int static_num[4];
extern const double ramp_time;

extern double static_freq[4][4096];
extern double new_static_freq[4][4096];
extern double destination_freq[4][1024];
extern int dynamic_list[1024];
extern int static_list[16384];
extern const unsigned long long  llSamplerate;
extern const int lThreadsPerBlock;
extern unsigned int tone_count[5];
extern unsigned int dynamic_tone_count[5];
extern double amp_list[16384];
extern double power_normalizer[4];
extern double frequency_limits[4];
extern int lMaxOutputLevel;
extern int amplitude_limit;
extern void reset_amp();
#endif