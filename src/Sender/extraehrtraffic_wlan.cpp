#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include <vector>
#include <thread>
#include <chrono>
#include <cstring>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netinet/udp.h>
#include <netinet/ip.h>
#include <netpacket/packet.h>
#include <net/ethernet.h>
#include <net/if.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>
#include <nlohmann/json.hpp>

#include <fcntl.h>
#include <linux/ptp_clock.h>


using json = nlohmann::json;

#define ETHER_TYPE_VLAN 0x8100
#define ETHER_TYPE_IPV4 0x0800

#ifndef FD_TO_CLOCKID
#define FD_TO_CLOCKID(fd) ((~(clockid_t) (fd) << 3) | 3) 
#endif
/*struct StreamConfig {
    std::string name;
    int interval_ms;
    int vlan_id;
    int dscp;
    int port;
    std::string filename;
    size_t max_paylod_size;
};*/
//new in v2
struct StreamConfig {
    std::string name;
    int interval_ms;
    int vlan_id;
    int pcp;        // <-- new field for VLAN PCP
    int dscp;       // <-- still used for IP header TOS
    int port;
    std::string filename;
    size_t max_paylod_size;
};


/* from system clock
std::string get_current_timestamp() {
    using namespace std::chrono;
    auto now = system_clock::now();
    auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;
    std::time_t t = system_clock::to_time_t(now);
    std::ostringstream oss;
    oss << std::put_time(std::localtime(&t), "%F %T") << "." << std::setw(3) << std::setfill('0') << ms.count();
    return oss.str();
} */

//from PTP Clock : 16 AUG 2025 
std::string get_current_timestamp() {
    int fd = open("/dev/ptp0", O_RDWR);
    clockid_t clkid = FD_TO_CLOCKID(fd);
    
    //ioctl(fd, PTP_CLOCK_GETCLOCKID,&clkid);
    struct timespec ts;
    if (clock_gettime(clkid,&ts)!=0 ){
     close(fd);
     return 0;
    }
    close(fd);
    std::time_t t = ts.tv_sec;
    std::tm tm = *std::localtime(&t);
    
    std::ostringstream oss;
    //long long total_ms = static_cast<long long>(ts.tv_sec)* 1000 + (ts.tv_nsec/1000000);
    oss << std::put_time(&tm, "%F %T") << "." << std::setw(3) << std::setfill('0') << (ts.tv_sec/1000000);
    
    return oss.str();
}




uint16_t checksum(uint16_t *buf, int nwords) {
    uint32_t sum = 0;
    for (; nwords > 0; nwords--)
        sum += *buf++;
    while (sum >> 16)
        sum = (sum & 0xffff) + (sum >> 16);
    return ~sum;
}

void send_stream(const std::string& iface, const std::string& src_mac,
                 const std::string& dst_mac, const std::string& src_ip,
                 const std::string& dst_ip, const StreamConfig& stream_cfg) {

    //constexpr size_t MAX_PAYLOAD_SIZE = 1024;  // You can adjust this as needed
    //uint8_t buffer[MAX_PAYLOAD_SIZE];
    std::vector<uint8_t> buffer(stream_cfg.max_paylod_size);

    int sockfd = socket(AF_PACKET, SOCK_RAW, htons(ETH_P_ALL));
    if (sockfd < 0) {
        perror("socket");
        return;
    }

     struct ifreq if_idx, if_mac;
    memset(&if_idx, 0, sizeof(struct ifreq));
    memset(&if_mac, 0, sizeof(struct ifreq));

    strncpy(if_idx.ifr_name, iface.c_str(), IFNAMSIZ-1);
    if (ioctl(sockfd, SIOCGIFINDEX, &if_idx) < 0) {
        perror("SIOCGIFINDEX");
        return;
    }

    strncpy(if_mac.ifr_name, iface.c_str(), IFNAMSIZ-1);
    if (ioctl(sockfd, SIOCGIFHWADDR, &if_mac) < 0) {
        perror("SIOCGIFHWADDR");
        return;
    }

    int ifindex = if_idx.ifr_ifindex;

    sockaddr_ll socket_address{};
    socket_address.sll_ifindex = ifindex;
    socket_address.sll_halen = ETH_ALEN;
    sscanf(dst_mac.c_str(), "%hhx:%hhx:%hhx:%hhx:%hhx:%hhx",
           &socket_address.sll_addr[0], &socket_address.sll_addr[1], &socket_address.sll_addr[2],
           &socket_address.sll_addr[3], &socket_address.sll_addr[4], &socket_address.sll_addr[5]);

    uint8_t frame[1500];
    memset(frame, 0, sizeof(frame));

    // Construct MAC header
    uint8_t* ether_ptr = frame;
    sscanf(dst_mac.c_str(), "%hhx:%hhx:%hhx:%hhx:%hhx:%hhx",
           &ether_ptr[0], &ether_ptr[1], &ether_ptr[2],
           &ether_ptr[3], &ether_ptr[4], &ether_ptr[5]);
    memcpy(&ether_ptr[6], &if_mac.ifr_hwaddr.sa_data, 6);
    *(uint16_t*)&ether_ptr[12] = htons(ETHER_TYPE_VLAN);

    // VLAN Header
   //old uint16_t vlan_tag = (0 << 13) | (stream_cfg.dscp << 5) | (stream_cfg.vlan_id & 0xFFF);
  //new in v2
  uint16_t vlan_tag = ((stream_cfg.pcp & 0x7) << 13) |   // PCP = 3 bits
                    (0 << 12) |                        // DEI = 0
                    (stream_cfg.vlan_id & 0x0FFF);     // VLAN ID = 12 bits

    *(uint16_t*)&ether_ptr[14] = htons(vlan_tag);
    *(uint16_t*)&ether_ptr[16] = htons(ETHER_TYPE_IPV4);

    int payload_offset = 18 + sizeof(iphdr) + sizeof(udphdr);
    int packet_len;

  
    std::string filename="input_data/"+ stream_cfg.filename;
    std::cout << "filename " << filename << std::endl;
    std::ifstream infile(filename, std::ios::binary);
    if (!infile) {
        std::cerr << "Unable to open file: " << filename << std::endl;
        close(sockfd);
        return;
    }

    int counter = 0;
    while (true) {
        infile.read(reinterpret_cast<char*>(buffer.data()), stream_cfg.max_paylod_size);
        std::streamsize bytesRead = infile.gcount();

        if (bytesRead <= 0) {
            // EOF or error reached — stop sending
            std::cout << "[" << stream_cfg.name << "] End of file reached or no data to read. Stopping stream." << std::endl;
            infile.clear();
            //infile.seekg(0);
            break;
        }

        // Build the Ethernet/IP/UDP headers and insert the payload:

        uint8_t frame[1500];
        memset(frame, 0, sizeof(frame));

        // MAC header setup (copy dst MAC, src MAC, VLAN tag, EtherType same as before)...
        uint8_t* ether_ptr = frame;
        sscanf(dst_mac.c_str(), "%hhx:%hhx:%hhx:%hhx:%hhx:%hhx",
               &ether_ptr[0], &ether_ptr[1], &ether_ptr[2],
               &ether_ptr[3], &ether_ptr[4], &ether_ptr[5]);

        memcpy(&ether_ptr[6], &if_mac.ifr_hwaddr.sa_data, 6);  // You should have copied src_mac into a byte array src_mac_data somewhere
        *(uint16_t*)&ether_ptr[12] = htons(ETHER_TYPE_VLAN);
        uint16_t vlan_tag = (0 << 13) | (stream_cfg.dscp << 5) | (stream_cfg.vlan_id & 0xFFF);
        *(uint16_t*)&ether_ptr[14] = htons(vlan_tag);
        *(uint16_t*)&ether_ptr[16] = htons(ETHER_TYPE_IPV4);

        int payload_offset = 18 + sizeof(iphdr) + sizeof(udphdr);

        // IP header pointers
        iphdr *ip = reinterpret_cast<iphdr*>(frame + 18);
        udphdr *udp = reinterpret_cast<udphdr*>(frame + 18 + sizeof(iphdr));
        uint8_t *payload_ptr = frame + payload_offset;

        /* add time to payload pointer begin - Abhilash 16 AUG 2025*/
        
         int fd = open("/dev/ptp0", O_RDWR);
         clockid_t clkid = FD_TO_CLOCKID(fd);
         struct timespec ts;
          if (clock_gettime(clkid,&ts)!=0 ){
           close(fd);
           return;
          }
        close(fd);
        double sent_secs = ts.tv_sec + ts.tv_nsec / 1e9;
        
        memcpy(payload_ptr, &sent_secs, sizeof(sent_secs));
        
        
        /*add time to payload pointer end*/ 
        
        // Copy payload read from file into frame payload area
        memcpy(payload_ptr+sizeof(sent_secs), buffer.data(), bytesRead);

        // Fill IP header
        ip->ihl = 5;
        ip->version = 4;
        ip->tos = stream_cfg.dscp << 2;
        ip->tot_len = htons(sizeof(iphdr) + sizeof(udphdr) + bytesRead);
        ip->id = htons(0);
        ip->frag_off = 0;
        ip->ttl = 64;
        ip->protocol = IPPROTO_UDP;
        ip->check = 0;
        inet_pton(AF_INET, src_ip.c_str(), &ip->saddr);
        inet_pton(AF_INET, dst_ip.c_str(), &ip->daddr);
        ip->check = checksum(reinterpret_cast<uint16_t*>(ip), sizeof(iphdr)/2);

        // UDP header
        udp->source = htons(12345);
        udp->dest = htons(stream_cfg.port);
        udp->len = htons(sizeof(udphdr) + bytesRead);
        udp->check = 0;

        int frame_len = payload_offset + bytesRead;

        if (sendto(sockfd, frame, frame_len, 0,
                   (struct sockaddr*)&socket_address, sizeof(sockaddr_ll)) < 0) {
            perror("sendto");
        } else {
            //std::cout << "[" << stream_cfg.name << "] sent packet #" << counter << ", bytes: " << bytesRead << " at time: " << sent_secs << std::endl;
        }

        counter++;

        std::this_thread::sleep_for(std::chrono::milliseconds(stream_cfg.interval_ms));
    }

    infile.close();
    close(sockfd);
}
int main() {
    std::ifstream f("qos_udp_config_updated_wlan.json");
    if (!f) {
        std::cerr << "Unable to open qos_udp_config_updated_wlan.json" << std::endl;
        return 1;
    }
  
    json config = json::parse(f);
    std::string iface = config["sender"]["interface"];
    std::string src_mac = config["sender"]["mac"];
    std::string src_ip = config["sender"]["ip"];
    std::string dst_mac = config["receiver"]["mac"];
    std::string dst_ip = config["receiver"]["ip"];

    // Manual stream definitions (overrides from file)
    std::vector<StreamConfig> streams = {
         {"ehr",1, config["traffic_streams"]["ehr"]["vlan_id"], 3, config["traffic_streams"]["ehr"]["dscp"], config["traffic_streams"]["ehr"]["port"],"ehr50k.dat" ,1400} //,
      //{"video",10,config["traffic_streams"]["video"]["vlan_id"],7, config["traffic_streams"]["video"]["dscp"], config["traffic_streams"]["video"]["port"],"video.dat",1400},
      //  {"vitals",10,config["traffic_streams"]["vitals"]["vlan_id"],6, config["traffic_streams"]["vitals"]["dscp"], config["traffic_streams"]["vitals"]["port"],"vitals.dat",1200}
    
        
    };
    std::cout << "[Running extratraffic]" << std::endl;
    std::vector<std::thread> threads;
    for (const auto& s : streams) {
        threads.emplace_back(send_stream, iface, src_mac, dst_mac, src_ip, dst_ip, s);
    }
    for (auto& t : threads) t.join();
    return 0;
}
