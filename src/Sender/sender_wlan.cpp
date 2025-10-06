#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <cstring>
#include <iomanip>

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
#include <fcntl.h>

//#include <linux/if_packet.h>
#include <linux/ptp_clock.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

#define ETHER_TYPE_VLAN 0x8100
#define ETHER_TYPE_IPV4 0x0800

#ifndef FD_TO_CLOCKID
#define FD_TO_CLOCKID(fd) ((~(clockid_t) (fd) << 3) | 3)
#endif

struct StreamConfig {
    std::string name;
    int interval_ms;        // inter-packet interval
    int vlan_id;            // 0..4095
    int pcp;                // VLAN PCP 0..7 (used for on-wire PCP AND for SO_PRIORITY)
    int dscp;               // 0..63 -> goes into IP TOS (dscp<<2)
    int port;               // UDP destination port
    std::string filename;   // payload source file (read in chunks)
    size_t max_paylod_size; // max bytes per read (note: keeps user's original field name)
};

static inline double ptp_time_seconds(const char* dev = "/dev/ptp0") {
    // Return PTP clock time in seconds if available; fallback to CLOCK_REALTIME
    int fd = open(dev, O_RDWR);
    timespec ts{};
    if (fd >= 0) {
        clockid_t clkid = FD_TO_CLOCKID(fd);
        if (clock_gettime(clkid, &ts) != 0) {
            clock_gettime(CLOCK_REALTIME, &ts);
        }
        close(fd);
    } else {
        clock_gettime(CLOCK_REALTIME, &ts);
    }
    return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) / 1e9;
}

static uint16_t ip_checksum(uint16_t* buf, int nwords) {
    uint32_t sum = 0;
    for (; nwords > 0; --nwords) sum += *buf++;
    while (sum >> 16) sum = (sum & 0xFFFF) + (sum >> 16);
    return static_cast<uint16_t>(~sum);
}

static bool parse_mac(const std::string& mac_str, uint8_t out[6]) {
    unsigned int b[6];
    if (sscanf(mac_str.c_str(), "%02x:%02x:%02x:%02x:%02x:%02x",
               &b[0], &b[1], &b[2], &b[3], &b[4], &b[5]) != 6) return false;
    for (int i = 0; i < 6; ++i) out[i] = static_cast<uint8_t>(b[i]);
    return true;
}

// Just to send EOF
void send_stream_EOF(const std::string& iface, const std::string& /*src_mac_str*/, const std::string& dst_mac_str,
                 const std::string& src_ip_str, const std::string& dst_ip_str, const StreamConfig& s) {

    // Open raw socket
    int sockfd = socket(AF_PACKET, SOCK_RAW, htons(ETH_P_ALL));
    if (sockfd < 0) { perror("socket"); return; }

    // Set SO_PRIORITY so taprio can classify into TCs by skb->priority
    {
        int prio = s.pcp; // map PCP (0..7) to skb priority; your taprio map handles 0..15
        if (setsockopt(sockfd, SOL_SOCKET, SO_PRIORITY, &prio, sizeof(prio)) < 0) {
            perror("setsockopt(SO_PRIORITY)");
        }
    }

    // Set QDISC BYPASS disable 
    //{
        int qdisc_bypass = 0; // disable
        setsockopt(sockfd, SOL_PACKET, PACKET_QDISC_BYPASS,&qdisc_bypass, sizeof(qdisc_bypass));
        //if (setsockopt(sockfd, SOL_PACKET, PACKET_QDISC_BYPASS,&qdisc_bypass, sizeof(qdisc_bypass)) <0) {
           // perror("setsockopt(QDISC BYPASS not able to disable)");
        //}
   // }

    // Resolve interface index and MAC
    struct ifreq if_idx{}; strncpy(if_idx.ifr_name, iface.c_str(), IFNAMSIZ - 1);
    if (ioctl(sockfd, SIOCGIFINDEX, &if_idx) < 0) { perror("SIOCGIFINDEX"); close(sockfd); return; }
    int ifindex = if_idx.ifr_ifindex;

    struct ifreq if_mac{}; strncpy(if_mac.ifr_name, iface.c_str(), IFNAMSIZ - 1);
    if (ioctl(sockfd, SIOCGIFHWADDR, &if_mac) < 0) { perror("SIOCGIFHWADDR"); close(sockfd); return; }

    // Destination MAC
    uint8_t dst_mac[6];
    if (!parse_mac(dst_mac_str, dst_mac)) { std::cerr << "Invalid dst MAC: " << dst_mac_str << "\n"; close(sockfd); return; }

    // sockaddr_ll target
    sockaddr_ll saddr{};
    saddr.sll_ifindex = ifindex;
    saddr.sll_halen = ETH_ALEN;
    memcpy(saddr.sll_addr, dst_mac, 6);

    // No Payload 

    // Constants for header sizes
    constexpr size_t ETH_VLAN_HDR = 18; // 14 (ETH) + 4 (802.1Q)
    constexpr size_t IP_HDR_LEN   = sizeof(iphdr);   // typically 20
    constexpr size_t UDP_HDR_LEN  = sizeof(udphdr);  // 8
    constexpr size_t TS_LEN       = sizeof(double);  // we prepend a double timestamp
    constexpr size_t EOF_LEN       = sizeof(uint8_t);  // we EOF 
    // Payload buffer
    
    // Frame buffer (do not exceed 1500 bytes for standard MTU)
    uint8_t frame[1500];

    // Precompute static Ethernet + VLAN header fields that don't change per packet
    // (We still rebuild below for clarity.)

    int counter = 0;
    uint8_t eof_reached =0;
   
    //Frame size
    memset(frame, 0, sizeof(frame));
    uint8_t* eth = frame; // start of Ethernet header

    // ETH DST/SRC
    memcpy(&eth[0], dst_mac, 6);
    memcpy(&eth[6], &if_mac.ifr_hwaddr.sa_data, 6);

    // 802.1Q tag
    *reinterpret_cast<uint16_t*>(&eth[12]) = htons(ETHER_TYPE_VLAN);
    uint16_t vlan_tci = static_cast<uint16_t>(((s.pcp & 0x7) << 13) | (0 << 12) | (s.vlan_id & 0x0FFF));
    *reinterpret_cast<uint16_t*>(&eth[14]) = htons(vlan_tci);
    *reinterpret_cast<uint16_t*>(&eth[16]) = htons(ETHER_TYPE_IPV4);

    // Pointers to IP/UDP
    const int payload_offset = ETH_VLAN_HDR + IP_HDR_LEN + UDP_HDR_LEN; // 18 + 20 + 8 = 46
    iphdr* ip  = reinterpret_cast<iphdr*>(frame + ETH_VLAN_HDR);
    udphdr* udp = reinterpret_cast<udphdr*>(frame + ETH_VLAN_HDR + IP_HDR_LEN);
    uint8_t* payload = frame + payload_offset;

    // Add PTP timestamp at the start of payload
    double sent_secs = ptp_time_seconds();
    memcpy(payload, &sent_secs, TS_LEN);
        
    //Add eof_reached after this
    memcpy(payload +TS_LEN, &eof_reached, EOF_LEN);
        
    size_t payload_len = TS_LEN + EOF_LEN  ;

    // Build IP header
    ip->ihl = 5;                    // 20 bytes
    ip->version = 4;
    ip->tos = static_cast<uint8_t>(s.dscp << 2);  // DSCP in upper 6 bits
    ip->tot_len = htons(static_cast<uint16_t>(IP_HDR_LEN + UDP_HDR_LEN + TS_LEN + EOF_LEN));
    ip->id = htons(0);
    ip->frag_off = 0;
    ip->ttl = 64;
    ip->protocol = IPPROTO_UDP;
    ip->check = 0;
    inet_pton(AF_INET, src_ip_str.c_str(), &ip->saddr);
    inet_pton(AF_INET, dst_ip_str.c_str(), &ip->daddr);
    ip->check = ip_checksum(reinterpret_cast<uint16_t*>(ip), IP_HDR_LEN / 2);

    // UDP header
    udp->source = htons(12345);
    udp->dest   = htons(static_cast<uint16_t>(s.port));
    udp->len    = htons(static_cast<uint16_t>(UDP_HDR_LEN + payload_len));
    udp->check  = 0; // optional for IPv4; set 0 to skip

    int frame_len = payload_offset + static_cast<int>(payload_len);

    if (sendto(sockfd, frame, frame_len, 0, reinterpret_cast<sockaddr*>(&saddr), sizeof(saddr)) < 0) {
        perror("sendto");
    } else {
        std::cout << "[" << s.name << "] EOF sent #" << counter
                     << ", frame_len=" << frame_len
                      << ", t=" << std::fixed << std::setprecision(9) << sent_secs << "\n";
    }

       

}


void send_stream(const std::string& iface, const std::string& /*src_mac_str*/, const std::string& dst_mac_str,
                 const std::string& src_ip_str, const std::string& dst_ip_str, const StreamConfig& s) {

    // Open raw socket
    int sockfd = socket(AF_PACKET, SOCK_RAW, htons(ETH_P_ALL));
    if (sockfd < 0) { perror("socket"); return; }

    // Set SO_PRIORITY so taprio can classify into TCs by skb->priority
    {
        int prio = s.pcp; // map PCP (0..7) to skb priority; your taprio map handles 0..15
        if (setsockopt(sockfd, SOL_SOCKET, SO_PRIORITY, &prio, sizeof(prio)) < 0) {
            perror("setsockopt(SO_PRIORITY)");
        }
    }

    // Set QDISC BYPASS disable 
    //{
        int qdisc_bypass = 0; // disable
        setsockopt(sockfd, SOL_PACKET, PACKET_QDISC_BYPASS,&qdisc_bypass, sizeof(qdisc_bypass));
        //if (setsockopt(sockfd, SOL_PACKET, PACKET_QDISC_BYPASS,&qdisc_bypass, sizeof(qdisc_bypass)) <0) {
           // perror("setsockopt(QDISC BYPASS not able to disable)");
        //}
   // }

    // Resolve interface index and MAC
    struct ifreq if_idx{}; strncpy(if_idx.ifr_name, iface.c_str(), IFNAMSIZ - 1);
    if (ioctl(sockfd, SIOCGIFINDEX, &if_idx) < 0) { perror("SIOCGIFINDEX"); close(sockfd); return; }
    int ifindex = if_idx.ifr_ifindex;

    struct ifreq if_mac{}; strncpy(if_mac.ifr_name, iface.c_str(), IFNAMSIZ - 1);
    if (ioctl(sockfd, SIOCGIFHWADDR, &if_mac) < 0) { perror("SIOCGIFHWADDR"); close(sockfd); return; }

    // Destination MAC
    uint8_t dst_mac[6];
    if (!parse_mac(dst_mac_str, dst_mac)) { std::cerr << "Invalid dst MAC: " << dst_mac_str << "\n"; close(sockfd); return; }

    // sockaddr_ll target
    sockaddr_ll saddr{};
    saddr.sll_ifindex = ifindex;
    saddr.sll_halen = ETH_ALEN;
    memcpy(saddr.sll_addr, dst_mac, 6);

    // Open payload file
    const std::string filename = std::string("input_data/") + s.filename;
    std::ifstream infile(filename, std::ios::binary);
    if (!infile) {
        std::cerr << "Unable to open file: " << filename << "\n";
        close(sockfd);
        return;
    }

    // Constants for header sizes
    constexpr size_t ETH_VLAN_HDR = 18; // 14 (ETH) + 4 (802.1Q)
    constexpr size_t IP_HDR_LEN   = sizeof(iphdr);   // typically 20
    constexpr size_t UDP_HDR_LEN  = sizeof(udphdr);  // 8
    constexpr size_t TS_LEN       = sizeof(double);  // we prepend a double timestamp
    constexpr size_t EOF_LEN       = sizeof(bool);  // we prepend a double timestamp
    // Payload buffer
    std::vector<uint8_t> filebuf(s.max_paylod_size);

    // Frame buffer (do not exceed 1500 bytes for standard MTU)
    uint8_t frame[1500];

    // Precompute static Ethernet + VLAN header fields that don't change per packet
    // (We still rebuild below for clarity.)

    int counter = 0;
    uint8_t eof_reached =0;
    while (true) {
        infile.read(reinterpret_cast<char*>(filebuf.data()), s.max_paylod_size);
        std::streamsize bytes_read = infile.gcount();
        if (bytes_read <= 0) {
            std::cout << "[" << s.name << "] EOF or no data. Stopping.\n";
            eof_reached=true; //need to be copied to payload beginning or so
            break;
        }

        memset(frame, 0, sizeof(frame));
        uint8_t* eth = frame; // start of Ethernet header

        // ETH DST/SRC
        memcpy(&eth[0], dst_mac, 6);
        memcpy(&eth[6], &if_mac.ifr_hwaddr.sa_data, 6);

        // 802.1Q tag
        *reinterpret_cast<uint16_t*>(&eth[12]) = htons(ETHER_TYPE_VLAN);
        uint16_t vlan_tci = static_cast<uint16_t>(((s.pcp & 0x7) << 13) | (0 << 12) | (s.vlan_id & 0x0FFF));
        *reinterpret_cast<uint16_t*>(&eth[14]) = htons(vlan_tci);
        *reinterpret_cast<uint16_t*>(&eth[16]) = htons(ETHER_TYPE_IPV4);

        // Pointers to IP/UDP
        const int payload_offset = ETH_VLAN_HDR + IP_HDR_LEN + UDP_HDR_LEN; // 18 + 20 + 8 = 46
        iphdr* ip  = reinterpret_cast<iphdr*>(frame + ETH_VLAN_HDR);
        udphdr* udp = reinterpret_cast<udphdr*>(frame + ETH_VLAN_HDR + IP_HDR_LEN);
        uint8_t* payload = frame + payload_offset;

        // Add PTP timestamp at the start of payload
        double sent_secs = ptp_time_seconds();
        memcpy(payload, &sent_secs, TS_LEN);
        
        //Add eof_reached after this
        memcpy(payload+TS_LEN, &eof_reached, EOF_LEN);
        
        // Copy file bytes after timestamp, but ensure MTU safety
        size_t max_file_bytes = sizeof(frame) - payload_offset - TS_LEN; // 1500 - headers - timestamp
        size_t copy_len = static_cast<size_t>(bytes_read) > max_file_bytes ? max_file_bytes : static_cast<size_t>(bytes_read);
        memcpy(payload + TS_LEN + EOF_LEN , filebuf.data(), copy_len);

        size_t payload_len = TS_LEN + EOF_LEN  + copy_len;

        // Build IP header
        ip->ihl = 5;                    // 20 bytes
        ip->version = 4;
        ip->tos = static_cast<uint8_t>(s.dscp << 2);  // DSCP in upper 6 bits
        ip->tot_len = htons(static_cast<uint16_t>(IP_HDR_LEN + UDP_HDR_LEN + payload_len));
        ip->id = htons(0);
        ip->frag_off = 0;
        ip->ttl = 64;
        ip->protocol = IPPROTO_UDP;
        ip->check = 0;
        inet_pton(AF_INET, src_ip_str.c_str(), &ip->saddr);
        inet_pton(AF_INET, dst_ip_str.c_str(), &ip->daddr);
        ip->check = ip_checksum(reinterpret_cast<uint16_t*>(ip), IP_HDR_LEN / 2);

        // UDP header
        udp->source = htons(12345);
        udp->dest   = htons(static_cast<uint16_t>(s.port));
        udp->len    = htons(static_cast<uint16_t>(UDP_HDR_LEN + payload_len));
        udp->check  = 0; // optional for IPv4; set 0 to skip

        int frame_len = payload_offset + static_cast<int>(payload_len);

        if (sendto(sockfd, frame, frame_len, 0, reinterpret_cast<sockaddr*>(&saddr), sizeof(saddr)) < 0) {
            perror("sendto");
        } else {
            std::cout << "[" << s.name << "] sent #" << counter
                      << ", bytes(file)=" << copy_len
                      << ", frame_len=" << frame_len
                      << ", t=" << std::fixed << std::setprecision(9) << sent_secs << "\n";
        }

        ++counter;
        std::this_thread::sleep_for(std::chrono::milliseconds(s.interval_ms));
    }
    //signal EOF reached
    send_stream_EOF(iface,"", dst_mac_str,
                 src_ip_str,dst_ip_str, s);
    infile.close();
    close(sockfd);
}




int main() {
    std::ifstream f("qos_udp_config_updated_wlan.json");
    if (!f) {
        std::cerr << "Unable to open qos_udp_config_updated_eth.json\n";
        return 1;
    }

    json config = json::parse(f);
    std::string iface   = config["sender"]["interface"];
    std::string src_mac = config["sender"]["mac"]; // not strictly needed; NIC MAC used
    std::string src_ip  = config["sender"]["ip"];
    std::string dst_mac = config["receiver"]["mac"];
    std::string dst_ip  = config["receiver"]["ip"];

    // Define streams (PCP chosen to map into taprio TCs by your map: 7->TC0, 6->TC1, 3->TC2, 0->TC3)
    std::vector<StreamConfig> streams = {
        {"ehr",    2,  config["traffic_streams"]["ehr"]["vlan_id"],    3, config["traffic_streams"]["ehr"]["dscp"],    config["traffic_streams"]["ehr"]["port"],    "ehr.dat",    1400},
        {"video", 10,  config["traffic_streams"]["video"]["vlan_id"],  7, config["traffic_streams"]["video"]["dscp"],  config["traffic_streams"]["video"]["port"],  "video.dat",  1400},
        {"vitals",10,  config["traffic_streams"]["vitals"]["vlan_id"], 6, config["traffic_streams"]["vitals"]["dscp"], config["traffic_streams"]["vitals"]["port"], "vitals.dat", 1200}
    };

    std::vector<std::thread> threads;
    for (const auto& s : streams) {
        threads.emplace_back(send_stream, iface, src_mac, dst_mac, src_ip, dst_ip, s);
    }
    for (auto& t : threads) t.join();
    return 0;
}
