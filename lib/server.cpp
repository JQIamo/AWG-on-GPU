#include "server.h"
volatile std::atomic<bool>  server_flag(false);
volatile std::atomic<bool>  stop_flag(false);
volatile std::atomic<bool>  continue_flag(false);
volatile std::atomic<bool> static_endflag(true);
volatile std::atomic<bool> amp_flag(false);
volatile std::atomic<bool> update_flag(false);
volatile std::atomic<bool> static_pulseflag(false);
int* update_index_map=nullptr;
std::vector<std::vector<std::pair<double,int>>> sorter;
fd_set readfds;

int TCP_server() {
    const int PORT = 56456;
    const int BUFFER_SIZE = 65536;
    int server_fd, new_socket, valread;
    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);
    char buffer[BUFFER_SIZE];
    const char* response = "Message received";
    int sd, maxsd, activity;
    int clients[5];

    // server_flag=false;

    // Create socket file descriptor
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        std::cerr << "Socket creation failed" << std::endl;
        return 1;
    }

    // Forcefully attach socket to the port 8080
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt))) {
        std::cerr << "Setsockopt failed" << std::endl;
        return 1;
    }
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(PORT);

    // Bind socket to address
    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        std::cerr << "Bind failed" << std::endl;
        return 1;
    }

    waiting:

    // Listen for incoming connections
    if (listen(server_fd, 3) < 0) {
        std::cerr << "Listen failed" << std::endl;
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

        activity = select(maxsd + 1, &readfds, NULL, NULL, NULL);
        if ((activity < 0) && (errno != EINTR)) {
            std::cerr << "Select failed: " << strerror(errno) << std::endl;
            return 1;
        }


        if (FD_ISSET(server_fd, &readfds)) {
            if ((new_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t*)&addrlen)) < 0) {
                std::cerr << "Accept failed" << std::endl;
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
                    valread = read(sd, buffer, BUFFER_SIZE);
                    if (valread < 0) {
                        std::cerr << "Connection closed or read error" << std::endl;\
                        std::cerr << strerror(errno) << std::endl;
                        close(sd);
                        clients[i]=0;
                        goto waiting;
                    }

                    std::cout << "Received message: " << buffer << std::endl;

                    // Check if message contains "go"
                    if (strstr(buffer, "go") != nullptr) {
                        // Send response back to client

                        send(new_socket, response, strlen(response), 0);
                        std::cout << "Sent response: " << response << std::endl;
                        // The sent in string should be an array of numbers separated by "," with a char 'g' at the end.
                        // The numbers should be a concanation of static_num, dynamic_num, static_freq, dynamic_freq, and dynamic_list;
                        // the latter three are only needed if dynamic_num > 0
                        int s_total=0;
                        int d_total=0;
                        
                        amp_flag = (bool)atoi(strtok(buffer,","));
                        // amp_flag = flags & 1;
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
                        int dynamic_indexes[1024];
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
                                    static_list[j] = sorter_flat[j+i].second;
                                }           
                                lcounter = dynamic_indexes[i];             
                            }
                        }

                        if (amp_flag){
                            for (int i = 0; i < s_total; i++){
                                amp_list[sorter_flat[i].second] = atof(strtok(NULL,","));
                            } 
                            int count=0;
                            for (int i = 0; i < 4; i++){
                                power_normalizer[i] = 0;
                                for (int j = 0; j < static_num[i]; j++){
                                    power_normalizer[i] += amp_list[count];
                                    count += 1;
                                }
                            }
                        }else{
                            for (int i = 0; i < 4; i++){
                                power_normalizer[i] = static_num[i];
                            }
                        }

                        if (strtok(NULL,",")[0]!='g'){
                            std::cerr << "Error in receiving." << std::endl;
                        }else{
                            update_flag=false;
                            server_flag=true;
                        }
                    }else if (strstr(buffer, "stop") != nullptr){
                        stop_flag=true;
                        break;
                    }else if (strstr(buffer, "cont") != nullptr){
                        continue_flag=true;
                    }else if (strstr(buffer, "up") != nullptr) {
                        // Used for ramping from the current waveform to new waveform
                        // keeping the phase continuity of each tone 
                        // The sent in string should be an array of numbers separated by "," with a char 'g' at the end.
                        // The numbers should be a concanation of static_num, dynamic_num, static_index, dynamic_freq, and dynamic_list;
                        // the latter three are only needed if dynamic_num > 0


                        // Send response back to client
                        send(new_socket, response, strlen(response), 0);
                        std::cout << "Sent response: " << response << std::endl;

                        int s_total=0;
                        int d_total=0;

                        double new_amp_list[4096];
                        unsigned int new_static_num[4];
                        
                        
                        new_static_num[0] = atoi(strtok(buffer,","));
                        for (int i = 1; i < 4; i++){
                            int tmp = atoi(strtok(NULL,","));
                            new_static_num[i] = tmp;
                            s_total += tmp;
                        }
                        delete[] update_index_map;
                        update_index_map = new int[s_total];
                        for (int i = 0; i < 4; i++){
                            int tmp = atoi(strtok(NULL,","));
                            dynamic_num[i] = tmp;
                            d_total += tmp;
                        }
                        int tmp_indexer = 0;
                        for (int i = 0; i < 4; i++){
                            for (int j = 0; j < new_static_num[i]; j++){
                                int static_index = atoi(strtok(NULL,","));
                                new_static_freq[i][j] = static_freq[i][static_index-tone_count[i]];
                                update_index_map[tmp_indexer+j] = static_index;
                                if (amp_flag){
                                    new_amp_list[tmp_indexer+j] = amp_list[static_index];
                                }
                            }
                            for (int j = 0; j < new_static_num[i]; j++){
                                static_freq[i][j] = new_static_freq[i][j];
                                // std::cout << "Saved frequency: " << i << "," << j << "," << static_freq[i][j] << std::endl;
                                if (amp_flag)amp_list[tmp_indexer+j] = new_amp_list[tmp_indexer+j];
                            }
                            tmp_indexer += new_static_num[i];
                        }
                        

                        for (int i = 0; i < 4; i++){
                            for (int j = 0; j < dynamic_num[i]; j++){
                                destination_freq[i][j] = atof(strtok(NULL,","));
                            }
                        }

                        int lcounter = 0;
                        for (int i = 0; i < d_total; i++){
                            int tmp = atoi(strtok(NULL,","));
                            dynamic_list[i] = tmp;
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

                        if (strtok(NULL,",")[0]!='u'){
                            std::cerr << "Error in receiving." << std::endl;
                        }else{
                            update_flag = true;
                            server_flag=true;
                        }
                    }else if(strstr(buffer, "amp_set") != nullptr){
                        lMaxOutputLevel = atoi(strtok(buffer,","));
                        if (lMaxOutputLevel > 80 && lMaxOutputLevel <= 2500){
                            reset_amp();      
                        }else{
                            std::cerr << "Amplitude out of range." << std::endl;
                        }
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
    close(server_fd);
    return 0;
}
