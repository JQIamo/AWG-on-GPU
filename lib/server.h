#include <iostream>
#include <cstring>
#include <cstdlib>
#include <unistd.h>
#include <arpa/inet.h>
#ifndef _server_included_
#define _server_included_
#include <sys/socket.h>
#include <math.h>
#include <errno.h>
#include <vector>
#include <utility>
#include <fstream>
#include <algorithm>
#include <atomic>
#include "parameters.h"
#include <sys/select.h>

// MODES
#define MODE_INIT 0
#define MODE_REARRANGE 1
#define MODE_ACCELERATE 2



extern int server_fd;
extern volatile std::atomic<bool>  server_flag;
extern volatile std::atomic<int>  mode_flag;
extern volatile std::atomic<bool>  stop_flag;
extern volatile std::atomic<bool>  continue_flag;
extern volatile std::atomic<bool> amp_flag;
extern volatile std::atomic<bool> static_endflag;
extern volatile std::atomic<bool>  static_flag;
extern int TCP_server();
extern volatile std::atomic<bool>  static_pulseflag;
extern volatile std::atomic<bool> error_flag;
extern volatile std::atomic<bool>  update_flag;
extern int *update_index_map;
extern void reset_amp(int channel, int value);
extern int index_flat_map[4][MAX_TWEEZERS_PER_CHANNEL];  
extern int dynamic_single_index[MAX_TWEEZERS_PER_CHANNEL];
extern int static_single_index[MAX_TWEEZERS_PER_CHANNEL];
extern int single_dynamic_channel;
extern float accel_frequency_diff;
#endif