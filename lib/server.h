#include <iostream>
#include <cstring>
#include <cstdlib>
#include <unistd.h>
#include <arpa/inet.h>
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
#ifndef _server_included_
#define _server_included_




extern volatile std::atomic<bool>  server_flag;
extern volatile std::atomic<bool>  stop_flag;
extern volatile std::atomic<bool>  continue_flag;
extern volatile std::atomic<bool> amp_flag;
extern volatile std::atomic<bool> static_endflag;
extern volatile std::atomic<bool>  static_flag;
extern int TCP_server();
extern void loadINterpolation();
extern volatile std::atomic<bool>  static_pulseflag;
extern volatile std::atomic<bool>  update_flag;
extern double new_static_freq[4][4096];
extern int *update_index_map;
#endif