#include "server.h"
volatile std::atomic<int>  mode_flag(MODE_INIT);
volatile std::atomic<bool>  server_flag(false);
volatile std::atomic<bool>  stop_flag(false);
volatile std::atomic<bool>  continue_flag(false);
volatile std::atomic<bool> static_endflag(true);
volatile std::atomic<bool> amp_flag(false);
volatile std::atomic<bool> error_flag(false);
volatile std::atomic<bool> update_flag(false);
volatile std::atomic<bool> static_pulseflag(false);
int* update_index_map=nullptr;
int single_dynamic_channel = -1;
int dynamic_single_index[MAX_TWEEZERS_PER_CHANNEL];
int static_single_index[MAX_TWEEZERS_PER_CHANNEL];
float accel_frequency_diff = 0;
int server_fd = -1;
fd_set readfds;

const char* response = "Message received";
int index_flat_map[4][MAX_TWEEZERS_PER_CHANNEL] = {{},{},{},{}};   


namespace{
    void tweezerInit(int new_socket, char* buffer){
        // Create a set of static tweezers for atom loading
        send(new_socket, response, strlen(response), 0);
        std::cout << "tweezerInit() sent response: " << response << std::endl;
        // The sent in string should be an array of numbers separated by "," with a char 'g' at the end.
        // The numbers should be a concatenation of static_num, dynamic_num, static_freq, dynamic_freq, and dynamic_list;
        // the latter three should only be included if dynamic_num > 0
        int s_total=0;
        int d_total=0;
        int counter = 0;

        try{
            amp_flag = (bool)atoi(strtok(buffer,","));

            for (int i = 0; i < 4; i++){
                int tmp = atoi(strtok(NULL,","));
                if (tmp > maximal_tweezer_number[i]) {
                    std::cout<<"Too many tweezers in channel "<<i<<"than allowed!"<<std::endl;
                    return;
                }
                static_num[i] = tmp;
                s_total += tmp;
            }

            for (int i = 0; i < 4; i++){
                int tmp = atoi(strtok(NULL,","));
                dynamic_num[i] = tmp;
                d_total += tmp;
            }

            for (int i = 0; i < 4; i++){
                for (int j = 0; j < static_num[i]; j++){
                    double st_freq = atof(strtok(NULL,","));
                    static_freq[i][j] = st_freq;
                    new_static_freq[i][j] = st_freq;
                    index_flat_map[i][j] = counter;
                    counter += 1;
                }
            }

            if (d_total > 0) {
                for (int i = 0; i < 4; i++){
                    for (int j = 0; j < dynamic_num[i]; j++){
                        destination_freq[i][j] = atof(strtok(NULL,","));
                    }
                }

                int lcounter = 0;
                for (int i = 0; i < d_total; i++){
                    int dy_index = atoi(strtok(NULL,","));
                    dynamic_list[i] = dy_index;
                    int tone_count_tmp = 0;
                    int dynamic_tone_list_tmp = 0;
                    for (int j = 0; j < 4; j++){
                        if (tone_count_tmp+static_num[j]>dy_index){
                            new_static_freq[j][dy_index-tone_count_tmp] = destination_freq[j][i-dynamic_tone_list_tmp];
                            break;
                        }

                        tone_count_tmp += static_num[j];
                        dynamic_tone_list_tmp += dynamic_num[j];
                    }
                    for (int j = lcounter; j < dy_index; j++){
                        static_list[j] = j;
                    }
                    lcounter = dy_index;
                }
            }

            if (amp_flag){
                for (int i = 0; i < s_total; i++){
                    amp_list[i] = atof(strtok(NULL,","));
                }
                int count=0;
                for (int i = 0; i < 4; i++){
                    power_normalizer[i] = 0;
                    for (int j = 0; j < static_num[i]; j++){
                        // power_normalizer[i] += amp_list[count];
                        // count += 1;
                        power_normalizer[i] += 1;
                    }
                }
                // for (int i = 0; i < s_total; i++){
                //     final_amp_list[i] = atof(strtok(NULL,","));
                // }
            }else{
                for (int i = 0; i < 4; i++){
                    power_normalizer[i] = static_num[i];
                }
            }

            if (strtok(NULL,",")[0]!='g')
                throw std::runtime_error("Erroneous command format");
            std::cout<<"Done"<<std::endl;
            update_flag=false;
            server_flag=true;
            mode_flag = MODE_INIT;
        }catch (const std::exception& e) {
            std::cerr << "[tweezerInit] COMMAND ERROR: " << e.what() << std::endl;
        }
        catch (...) {
            std::cerr << "[tweezerInit] COMMAND ERROR: Non-standard error" << std::endl;
        }
    }

    void tweezerInit_sort(int new_socket, char* buffer){
        // Create a set of static tweezers for atom loading
        std::vector<std::vector<std::pair<double,int>>> sorter;
        send(new_socket, response, strlen(response), 0);
        std::cout << "tweezerInit_sort() sent response: " << response << std::endl;
        // The sent in string should be an array of numbers separated by "," with a char 'g' at the end.
        // The numbers should be a concanation of static_num, dynamic_num, static_freq, dynamic_freq, and dynamic_list;
        // the latter three are only needed if dynamic_num > 0
        int s_total=0;
        int d_total=0;
        try{
            amp_flag = (bool)atoi(strtok(buffer,","));

            for (int i = 0; i < 4; i++){
                int tmp = atoi(strtok(NULL,","));
                static_num[i] = tmp;
                s_total += tmp;
            }

            for (int i = 0; i < 4; i++){
                int tmp = atoi(strtok(NULL,","));
                dynamic_num[i] = tmp;
                d_total += tmp;
            }
            sorter.clear();
            sorter.resize(4);

            for (int i = 0; i < 4; i++){
                sorter[i].clear();
                sorter[i].resize(static_num[i]);
                for (int j = 0; j < static_num[i]; j++){
                    static_freq[i][j] = atof(strtok(NULL,","));
                    sorter[i][j]=std::make_pair(static_freq[i][j],j);
                }

                std::sort(sorter[i].begin(),sorter[i].end());
                for (int j = 0; j < static_num[i]; j++){
                    static_freq[i][j] = sorter[i][j].first;
                    new_static_freq[i][j] = static_freq[i][j];
                }
            }
            int lcounter = 0;
                std::vector<std::pair<double,int>> sorter_flat(sorter[0].size()+sorter[1].size()+sorter[2].size()+sorter[3].size());
                for (int i = 0; i < 4; i++){
                    for (int j = 0; j < sorter[i].size(); j++){
                        int index = find_if(sorter[i].begin(),sorter[i].end(),[&](std::pair<double,int> p){return p.second==j;})-sorter[i].begin();
                        sorter_flat[j+lcounter] = std::make_pair(sorter[i][index].first,index+lcounter);
                    }
                    lcounter += sorter[i].size();
                }

            int dynamic_indexes[4*MAX_TWEEZERS_PER_CHANNEL];
            printf("Check1\n");
            if (d_total > 0) {
                for (int i = 0; i < 4; i++){
                    for (int j = 0; j < dynamic_num[i]; j++){
                        destination_freq[i][j] = atof(strtok(NULL,","));
                    }
                }

                lcounter = 0;
                for (int i = 0; i < d_total; i++){
                    dynamic_indexes[i] = atoi(strtok(NULL,","));
                    dynamic_list[i] = sorter_flat[dynamic_indexes[i]].second;
                    int tone_count_tmp = 0;
                    int dynamic_tone_list_tmp = 0;
                    for (int j = 0; j < 4; j++){
                        if (tone_count_tmp+static_num[j]>dynamic_list[i]){
                            new_static_freq[j][dynamic_list[i]-tone_count_tmp] = destination_freq[j][i-dynamic_tone_list_tmp];
                            break;
                        }

                        tone_count_tmp += static_num[j];
                        dynamic_tone_list_tmp += dynamic_num[j];
                    }
                    for (int j = lcounter; j < dynamic_indexes[i]; j++){
                        static_list[j] = sorter_flat[j].second;
                    }
                    lcounter = dynamic_indexes[i];
                }
            }

            printf("Check2\n");
            if (amp_flag){
                for (int i = 0; i < s_total; i++){
                    amp_list[sorter_flat[i].second] = atof(strtok(NULL,","));
                }
                int count=0;
                for (int i = 0; i < 4; i++){
                    power_normalizer[i] = 0;
                    for (int j = 0; j < static_num[i]; j++){
                        // power_normalizer[i] += amp_list[count];
                        // count += 1;
                        power_normalizer[i] += 1;
                    }
                }
                // for (int i = 0; i < s_total; i++){
                //     final_amp_list[sorter_flat[i].second] = atof(strtok(NULL,","));
                // }
            }else{
                for (int i = 0; i < 4; i++){
                    power_normalizer[i] = static_num[i];
                }
            }

            if (strtok(NULL,",")[0]!='g')
                throw std::runtime_error("Erroneous command format");
            std::cout<<"Done"<<std::endl;
            update_flag=false;
            server_flag=true;
            mode_flag = MODE_INIT;
        }catch (const std::exception& e) {
            std::cerr << "[tweezerInit_sort] COMMAND ERROR: " << e.what() << std::endl;
        }
        catch (...) {
            std::cerr << "[tweezerInit_sort] COMMAND ERROR: Non-standard error" << std::endl;
        }
    }

    void updateTweezers(int new_socket, char* buffer){
        // Note this function only handles movement in one channel for 1d rearrangement compatible with the code in GPU-labscript-interface project.
        // Used for ramping from the current waveform to new waveform
        // keeping the phase continuity of each tone 
        // The sent in string should be an array of numbers separated by "," with a char 'g' at the end.
        // The numbers should be a concanation of static_num, dynamic_num, static_index, dynamic_freq, and dynamic_list;
        // the latter three are only needed if dynamic_num > 0


        // Send response back to client
        send(new_socket, response, strlen(response), 0);
        std::cout << "updateTweezers() sent response: " << response << std::endl;

        int s_total=0;
        int d_total=0;

        double new_amp_list[4*MAX_TWEEZERS_PER_CHANNEL];
        unsigned int new_static_num[4];
        try{
            int dynamic_channel = atoi(strtok(buffer,","));
            for (int i = 0; i < 4; i++){
                int tmp;
                if (i==dynamic_channel){
                    tmp = atoi(strtok(NULL,","));
                } else {
                    tmp = static_num[i];
                }
                new_static_num[i] = tmp;
                s_total += tmp;
            }
            delete[] update_index_map;
            update_index_map = new int[s_total];
            for (int i = 0; i < 4; i++){
                int tmp;
                if (i==dynamic_channel){
                    tmp = atoi(strtok(NULL,","));
                } else {
                    tmp = 0;
                }
                dynamic_num[i] = tmp;
                d_total += tmp;
            }

            int tmp_indexer = 0;
            for (int i = 0; i < 4; i++){
                for (int j = 0; j < new_static_num[i]; j++){
                    int static_index;
                    if (i==dynamic_channel){
                        static_index = atoi(strtok(NULL,","));
                    }else {
                        static_index = j;
                    }
                    new_static_freq[i][j] = static_freq[i][static_index];
                    update_index_map[tmp_indexer+j] = index_flat_map[i][static_index];
                    if (amp_flag){
                        new_amp_list[tmp_indexer+j] = amp_list[index_flat_map[i][static_index]];
                    }
                }
                for (int j = 0; j < new_static_num[i]; j++){
                    static_freq[i][j] = new_static_freq[i][j];
                    if (amp_flag)amp_list[tmp_indexer+j] = new_amp_list[tmp_indexer+j];
                }
                tmp_indexer += new_static_num[i];
            }


            for (int j = 0; j < d_total; j++){
                destination_freq[dynamic_channel][j] = atof(strtok(NULL,","));
            }

            int lcounter = 0;
            int pre_count = 0;
            for (int i = 0; i < dynamic_channel; i++){
                pre_count += static_num[i];
            }
            for (int i = 0; i < d_total; i++){
                int tmp = atoi(strtok(NULL,","));
                dynamic_list[i] = tmp+pre_count;
                int tone_count_tmp = 0;
                int dynamic_tone_list_tmp = 0;
                for (int j = 0; j < 4; j++){
                    if (tone_count_tmp+new_static_num[j]>dynamic_list[i]){
                        new_static_freq[j][dynamic_list[i]-tone_count_tmp] = destination_freq[j][i-dynamic_tone_list_tmp];
                        break;
                    }
                    tone_count_tmp += new_static_num[j];
                    dynamic_tone_list_tmp += dynamic_num[j];
                }
                for (int j = lcounter; j < tmp; j++){
                    static_list[j] = i+j;
                }
                lcounter = tmp;
            }

            for (int j = lcounter; j < s_total; j++){
                static_list[j] = d_total+j;
            }
            for (int i = 0; i < 4; i++){
                static_num[i] = new_static_num[i];
            }

            if (strtok(NULL,",")[0]!='u')
                throw std::runtime_error("Erroneous command format");
            update_flag = true;
            server_flag=true;
        }catch (const std::exception& e) {
            std::cerr << "[updateTweezers] COMMAND ERROR: " << e.what() << std::endl;
        }
        catch (...) {
            std::cerr << "[updateTweezers] COMMAND ERROR: Non-standard error" << std::endl;
        }
    }

    void tweezerShake(int new_socket, char* buffer){
        send(new_socket, response, strlen(response), 0);
        std::cout << "tweezerShake() sent response: " << response << std::endl;
        // The sent in string should be an array of numbers separated by "," with a char 'g' at the end.
        // The numbers should be a concanation of static_num, dynamic_num, static_freq, dynamic_freq, and dynamic_list;
        // the latter three are only needed if dynamic_num > 0
        int s_total=0;
        try{
            amp_flag = (bool)atoi(strtok(buffer,","));
            // amp_flag = flags & 1;
            for (int i = 0; i < 4; i++){
                int tmp = atoi(strtok(NULL,","));
                static_num[i] = tmp;
                s_total += tmp;
            }
            for (int i = 0; i < 4; i++){
                for (int j = 0; j < static_num[i]; j++){
                    static_freq[i][j] = atof(strtok(NULL,","));
                    new_static_freq[i][j] = static_freq[i][j];
                }
            }
            if (amp_flag){
                for (int i = 0; i < s_total; i++){
                    amp_list[i] = atof(strtok(NULL,","));
                }
            }
            for (int i = 0; i < 4; i++){
                power_normalizer[i] = static_num[i];
            }
            shake_amp = atoi(strtok(NULL,","));
            shake_freq = atoi(strtok(NULL,","));
            if (strtok(NULL,",")[0]!='s')
                throw std::runtime_error("Erroneous command format");
            std::cout<<"Done"<<std::endl;
            update_flag=false;
            server_flag=true;
        }catch (const std::exception& e) {
            std::cerr << "[tweezerShake] COMMAND ERROR: " << e.what() << std::endl;
        }
        catch (...) {
            std::cerr << "[tweezerShake] COMMAND ERROR: Non-standard error" << std::endl;
        }
    }

    void tweezerRearr(int new_socket, char* buffer){
        // Note this function only handles movement in one channel for 1d rearrangement compatible with the code in GPU-labscript-interface project.
        // Used for ramping from the current waveform to new waveform
        // keeping the phase continuity of each tone 
        // The sent in string should be an array of numbers separated by "," with a char 'g' at the end.
        // The numbers should be a concanation of static_num, dynamic_num, static_index, dynamic_freq, and dynamic_list;
        // the latter three are only needed if dynamic_num > 0


        // Send response back to client
        send(new_socket, response, strlen(response), 0);
        std::cout << "tweezerRearr() sent response: " << response << std::endl;
        unsigned int new_static_num[4];
        try{
            single_dynamic_channel = atoi(strtok(buffer,","));
            if (single_dynamic_channel < 0 || single_dynamic_channel > 3){
                if (strtok(NULL,",")[0]!='r'){
                    std::cerr << "[tweezerRearr] Erroneous command format" << std::endl;
                }else{
                    error_flag = true;
                    server_flag = true;
                    mode_flag = MODE_REARRANGE;
                }
                return;
            }
            int total_tones = atoi(strtok(NULL,","));
            int dynamic_tones = atoi(strtok(NULL,","));
            for (int i = 0; i < total_tones; i++){
                index_flat_map[single_dynamic_channel][i] = index_flat_map[single_dynamic_channel][atoi(strtok(NULL,","))];
            }
            for (int i = 0; i < dynamic_tones; i++){
                destination_freq[single_dynamic_channel][i] = atof(strtok(NULL,","));
            }
            for (int i = 0; i < dynamic_tones; i++){
                dynamic_single_index[i] = atoi(strtok(NULL,","));
            }
            int dy_counter = 0;
            int st_counter = 0;
            for (int i = 0; i < total_tones; i++){
                if (i<dynamic_single_index[dy_counter]||dy_counter==dynamic_tones){
                    // static_single_index[st_counter] = i;
                    static_single_index[st_counter] = index_flat_map[single_dynamic_channel][i];
                    st_counter++;
                }else{
                    dynamic_single_index[dy_counter] = index_flat_map[single_dynamic_channel][i];
                    dy_counter++;
                }
            }
            if (strtok(NULL,",")[0]!='r')
                throw std::runtime_error("Erroneous command format");
            static_num[single_dynamic_channel] = total_tones;
            dynamic_num[single_dynamic_channel] = dynamic_tones;
            mode_flag = MODE_REARRANGE;
            server_flag=true;
        }catch (const std::exception& e) {
            std::cerr << "[tweezerRearr] COMMAND ERROR: " << e.what() << std::endl;
        }
        catch (...) {
            std::cerr << "[tweezerRearr] COMMAND ERROR: Non-standard error" << std::endl;
        }
    }
}
    void tweezerAccel(int new_socket, char* buffer){
        // Create a set of static tweezers for atom loading
        send(new_socket, response, strlen(response), 0);
        std::cout << "tweezerAccel() sent response: " << response << std::endl;
        // The sent in string should be an array of numbers separated by "," with a char 'g' at the end.
        // The numbers should be a concatenation of static_num, dynamic_num, static_freq, dynamic_freq, and dynamic_list;
        // the latter three should only be included if dynamic_num > 0
        int s_total=0;
        int counter = 0;

        try{
            amp_flag = (bool)atoi(strtok(buffer,","));
            single_dynamic_channel = atoi(strtok(NULL,","));

            for (int i = 0; i < 4; i++){
                int tmp = atoi(strtok(NULL,","));
                if (tmp > maximal_tweezer_number[i]) {
                    std::cout<<"Too many tweezers in channel "<<i<<"than allowed!"<<std::endl;
                    return;
                }
                static_num[i] = tmp;
                s_total += tmp;
            }

            for (int i = 0; i < 4; i++){
                if (i==single_dynamic_channel) {
                    dynamic_num[i] = static_num[i];
                } else {
                    dynamic_num[i] = 0;
                }
            }

            int dy_index = 0;

            for (int i = 0; i < 4; i++){
                for (int j = 0; j < static_num[i]; j++){
                    double st_freq = atof(strtok(NULL,","));
                    static_freq[i][j] = st_freq;
                    new_static_freq[i][j] = st_freq;
                    index_flat_map[i][j] = counter;
                    counter += 1;
                }
                if (i<single_dynamic_channel) dy_index += static_num[i];
            }

            accel_frequency_diff = atof(strtok(NULL,","));

            int lcounter = 0;
            for (int i = 0; i < dynamic_num[single_dynamic_channel]; i++){
                dynamic_list[i] = dy_index;
                int tone_count_tmp = 0;
                int dynamic_tone_list_tmp = 0;
                for (int j = 0; j < 4; j++){
                    tone_count_tmp += static_num[j];
                    dynamic_tone_list_tmp += dynamic_num[j];
                }
                for (int j = lcounter; j < dy_index; j++){
                    static_list[j] = j;
                }
                lcounter = dy_index;
                dy_index++;
            }

            if (amp_flag){
                for (int i = 0; i < s_total; i++){
                    amp_list[i] = atof(strtok(NULL,","));
                }
                for (int i = 0; i < 4; i++){
                    power_normalizer[i] = 0;
                    for (int j = 0; j < static_num[i]; j++){
                        power_normalizer[i] += 1;
                    }
                }
            }else{
                for (int i = 0; i < 4; i++){
                    power_normalizer[i] = static_num[i];
                }
            }

            if (strtok(NULL,",")[0]!='a')
                throw std::runtime_error("Erroneous command format");
            std::cout<<"Done"<<std::endl;
            update_flag=false;
            server_flag=true;
            mode_flag = MODE_ACCELERATE;
        }catch (const std::exception& e) {
            std::cerr << "[tweezerAccel] COMMAND ERROR: " << e.what() << std::endl;
        }
        catch (...) {
            std::cerr << "[tweezerAccel] COMMAND ERROR: Non-standard error" << std::endl;
        }
    }


int TCP_server() {
    const int PORT = 56456;
    const int BUFFER_SIZE = 655360;
    int new_socket, valread;
    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);
    char buffer[BUFFER_SIZE];
    int sd, maxsd, activity;
    int clients[5];

    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        std::cerr << "[TCP_server] Socket creation failed" << std::endl;
        return 1;
    }

    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt))) {
        std::cerr << "[TCP_server] Setsockopt failed" << std::endl;
        return 1;
    }

    struct timeval timeout;
    timeout.tv_sec = 1;      // 5 seconds
    timeout.tv_usec = 0;     // 0 microseconds

    // Set receive timeout
    if (setsockopt(server_fd, SOL_SOCKET, SO_RCVTIMEO, (const char*)&timeout, sizeof(timeout)) < 0) {
        std::cerr << "[TCP_server] Failed to set receive timeout\n";
    }
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(PORT);

    // Bind socket to address
    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        std::cerr << "[TCP_server] Bind failed" << std::endl;
        return 1;
    }

    waiting:

    if (listen(server_fd, 3) < 0) {
        std::cerr << "[TCP_server] Listen failed" << std::endl;
        return 1;
    }

    // Receive data from client
    while (!stop_flag) {
        FD_ZERO(&readfds);
        FD_SET(server_fd, &readfds);
        maxsd = server_fd;

        for (int i = 0; i < 5; i++) {
            sd = clients[i];
            if (sd > 0) {
                FD_SET(sd, &readfds);
            }
            if (sd > maxsd) {
                maxsd = sd;
            }
        }

        activity = select(maxsd + 1, &readfds, NULL, NULL, &timeout);
        if ((activity < 0) && (errno != EINTR)) {
            std::cerr << "[TCP_server] Select failed: " << strerror(errno) << std::endl;
            return 1;
        }


        if (FD_ISSET(server_fd, &readfds)) {
            if ((new_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t*)&addrlen)) < 0) {
                std::cerr << "[TCP_server] Accept failed" << std::endl;
                return 1;
            }
        
            for (int i = 0; i < 5; i++) {
                if (clients[i] == 0) {
                    clients[i] = new_socket;
                    break;
                }
            }
        }else{
            for (int i = 0; i < 5; i++) {
                sd = clients[i];
                if (FD_ISSET(sd, &readfds)) {
                    int buf_pos=0;
                    while (!stop_flag){
                        valread = read(sd, buffer+buf_pos, BUFFER_SIZE);
                        if (valread < 0) {
                            std::cerr << "[TCP_server] Connection closed or read error" << std::endl;\
                            std::cerr << strerror(errno) << std::endl;
                            close(sd);
                            clients[i]=0;
                            goto waiting;
                        }
                        buf_pos+=valread;
                        if (valread==0 && buf_pos>0){break;}
                    }

                    std::cout << "Received message: " << buffer << std::endl;

                    // Check if message contains "go"
                    if (strstr(buffer, "go") != nullptr) {
                        tweezerInit(new_socket, buffer);
                        // tweezerInit_sort(new_socket, buffer);
                    }else if (strstr(buffer, "stop") != nullptr){
                        // Terminate the DMA thread
                        stop_flag=true;
                        break;
                    }else if (strstr(buffer, "cont") != nullptr){
                        // Proceed to the next step in test mode
                        continue_flag=true;
                    }else if (strstr(buffer, "up") != nullptr) {
                        updateTweezers(new_socket, buffer);
                    }else if(strstr(buffer, "amp_set") != nullptr){
                        int channel = atoi(strtok(buffer,","));
                        int value = atoi(strtok(buffer,","));
                        if (value > 80 && value <= 2500 && channel < lNumCh){
                            reset_amp(channel, value);      
                            std::cout<<"Set Amplitude: "<<value<<std::endl;
                        }else{
                            std::cerr << "[TCP_server] Amplitude out of range." << std::endl;
                        }
                    }else if (strstr(buffer, "shake") != nullptr){
                        tweezerShake(new_socket, buffer);
                    }else if (strstr(buffer, "rearr") != nullptr){
                        tweezerRearr(new_socket, buffer);
                    }else if (strstr(buffer, "time_set") != nullptr){
                        try {
                            double ramp_time = atof(strtok(buffer,","));
                            if (ramp_time < 0 || ramp_time > 1) throw std::runtime_error("Invalid ramp time.");
                            dynamic_buffersize = 4*static_cast<unsigned int>(round(0.5*ramp_time*static_cast<double>(llSamplerate)));
                        }catch (const std::exception& e) {
                            std::cerr << "[TCP_server] Error: " << e.what() << std::endl;
                        }catch (...) {
                            std::cerr << "[TCP_server] Unknown error." << std::endl;
                        }
                    }else if (strstr(buffer, "accel") != nullptr){
                        tweezerAccel(new_socket, buffer);
                    }else{
                        std::cout<<"Unknown command: "<<buffer<<std::endl;
                    }
                    close(sd);
                    clients[i]=0;
                    // Clear buffer
                    memset(buffer, 0, BUFFER_SIZE);
                }
            }
        }
    }
    std::cout << "Server stopped." << std::endl;
    memset(buffer, 0, BUFFER_SIZE);
    stop_flag=true;
    // Close sockets
    close(server_fd);

    return 0;
}
