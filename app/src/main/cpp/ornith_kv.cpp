#include <jni.h>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

namespace {

enum class KvMode : int { F16 = 0, Q8Q8 = 1, Q8Q5 = 2, Q8Q4 = 3 };

constexpr uint64_t kFullLayers = 8;
constexpr uint64_t kKvHeads = 4;
constexpr uint64_t kHeadDim = 256;
constexpr uint64_t kStreams = 2;
constexpr uint64_t kPageTokens = 256;
constexpr uint64_t kBlock = 32;

static inline uint16_t f32_to_f16(float x) {
    uint32_t u; std::memcpy(&u, &x, 4);
    uint32_t s=(u>>16)&0x8000u, e=(u>>23)&0xffu, m=u&0x7fffffu;
    if(e==255) return (uint16_t)(s|(m?0x7e00u:0x7c00u));
    int32_t he=(int32_t)e-127+15;
    if(he>=31) return (uint16_t)(s|0x7c00u);
    if(he<=0) {
        if(he<-10) return (uint16_t)s;
        m|=0x800000u;
        uint32_t q=m>>(14-he);
        return (uint16_t)(s|q);
    }
    uint16_t h=(uint16_t)(s|((uint32_t)he<<10)|(m>>13));
    uint32_t r=m&0x1fffu;
    if(r>0x1000u || (r==0x1000u && (h&1))) ++h;
    return h;
}

static inline float f16_to_f32(uint16_t h) {
    uint32_t s=((uint32_t)h&0x8000u)<<16, e=((uint32_t)h>>10)&31u, m=h&1023u, u;
    if(!e) {
        if(!m) u=s;
        else {
            int ee=-14;
            while(!(m&1024u)){m<<=1;--ee;}
            m&=1023u; u=s|((uint32_t)(ee+127)<<23)|(m<<13);
        }
    } else if(e==31) u=s|0x7f800000u|(m<<13);
    else u=s|((e-15u+127u)<<23)|(m<<13);
    float x; std::memcpy(&x,&u,4); return x;
}

struct Encoded {
    std::vector<uint8_t> payload;
    std::vector<float> scales;
    uint64_t logicalBytes = 0;
};

static inline uint64_t packedBytes(uint64_t elements, int bits) {
    return (elements * (uint64_t)bits + 7u) / 8u;
}

static Encoded encodeQ(const float* src, size_t n, int bits) {
    Encoded out;
    const size_t blocks = (n + kBlock - 1) / kBlock;
    out.scales.resize(blocks, 1.f);
    out.payload.resize((size_t)packedBytes(n, bits), 0);

    const int maxQ = (1 << (bits - 1)) - 1;
    const int minQ = -(1 << (bits - 1));
    for(size_t b=0; b<blocks; ++b) {
        const size_t begin=b*kBlock, end=std::min(n,begin+kBlock);
        float mx=0.f;
        for(size_t i=begin;i<end;i++) mx=std::max(mx,std::fabs(src[i]));
        float scale = mx > 0.f ? mx/(float)maxQ : 1.f;
        out.scales[b]=scale;
        for(size_t i=begin;i<end;i++) {
            int q=(int)std::lrint(src[i]/scale);
            q=std::max(minQ,std::min(maxQ,q));
            uint32_t uq=(uint32_t)(q-minQ);
            size_t bit=i*(size_t)bits;
            size_t byte=bit>>3;
            unsigned shift=(unsigned)(bit&7u);
            uint32_t mask=(1u<<bits)-1u;
            uint32_t word=(uint32_t)out.payload[byte] | ((uint32_t)q /* overwritten below */ << 0);
            (void)word;
            uint32_t value=(uint32_t)out.payload[byte] | ((uq & mask) << shift);
            out.payload[byte]=(uint8_t)(value&0xffu);
            if(shift+bits>8) {
                uint32_t spill=uq >> (8-shift);
                out.payload[byte+1]=(uint8_t)(out.payload[byte+1] | (uint8_t)(spill&0xffu));
                if(shift+bits>16)
                    out.payload[byte+2]=(uint8_t)(out.payload[byte+2] | (uint8_t)(spill>>8));
            }
        }
    }
    out.logicalBytes=out.payload.size()+out.scales.size()*sizeof(float);
    return out;
}

static void decodeQ(const Encoded& in, float* dst, size_t n, int bits) {
    const int minQ=-(1<<(bits-1));
    const uint32_t mask=(1u<<bits)-1u;
    for(size_t i=0;i<n;i++) {
        size_t bit=i*(size_t)bits, byte=bit>>3;
        unsigned shift=(unsigned)(bit&7u);
        uint32_t raw=(uint32_t)in.payload[byte] >> shift;
        if(shift+bits>8) raw|=(uint32_t)in.payload[byte+1] << (8-shift);
        if(shift+bits>16) raw|=(uint32_t)in.payload[byte+2] << (16-shift);
        int q=(int)(raw&mask)+minQ;
        dst[i]=(float)q*in.scales[i/kBlock];
    }
}

static Encoded encodeF16(const float* src, size_t n) {
    Encoded out;
    out.payload.resize(n*2);
    for(size_t i=0;i<n;i++) {
        uint16_t h=f32_to_f16(src[i]);
        out.payload[i*2]=(uint8_t)(h&255u);
        out.payload[i*2+1]=(uint8_t)(h>>8);
    }
    out.logicalBytes=out.payload.size();
    return out;
}

static void decodeF16(const Encoded& in, float* dst, size_t n) {
    for(size_t i=0;i<n;i++) {
        uint16_t h=(uint16_t)in.payload[i*2] | ((uint16_t)in.payload[i*2+1]<<8);
        dst[i]=f16_to_f32(h);
    }
}

static float rmse(const std::vector<float>& a,const std::vector<float>& b) {
    long double s=0;
    for(size_t i=0;i<a.size();i++){ long double d=a[i]-b[i]; s+=d*d; }
    return (float)std::sqrt((double)(s/a.size()));
}

static float maxAbs(const std::vector<float>& a,const std::vector<float>& b) {
    float m=0.f;
    for(size_t i=0;i<a.size();i++) m=std::max(m,std::fabs(a[i]-b[i]));
    return m;
}

static uint64_t ceilDiv(uint64_t a,uint64_t b){return b?((a+b-1)/b):0;}

static std::string probe(int requestedTokens, int mode) {
    if(requestedTokens<=0 || requestedTokens>262144) return "ERR ORNITH15_KV bad_tokens";
    if(mode<0 || mode>3) return "ERR ORNITH15_KV bad_mode";

    const uint64_t elementsPerPage=kPageTokens*kFullLayers*kKvHeads*kHeadDim;
    const size_t testElements=std::min<uint64_t>(elementsPerPage, 1u<<20);

    std::vector<float> k(testElements), v(testElements), rk(testElements), rv(testElements);
    for(size_t i=0;i<testElements;i++) {
        k[i]=std::sin((float)i*0.017f)*1.7f+std::cos((float)i*0.0031f)*0.23f;
        v[i]=std::cos((float)i*0.013f)*0.91f-std::sin((float)i*0.0017f)*0.31f;
    }

    Encoded ek, ev;
    const char* name="";
    if(mode==0) {
        name="F16"; ek=encodeF16(k.data(),k.size()); ev=encodeF16(v.data(),v.size());
        decodeF16(ek,rk.data(),rk.size()); decodeF16(ev,rv.data(),rv.size());
    } else if(mode==1) {
        name="Q8_Q8"; ek=encodeQ(k.data(),k.size(),8); ev=encodeQ(v.data(),v.size(),8);
        decodeQ(ek,rk.data(),rk.size(),8); decodeQ(ev,rv.data(),rv.size(),8);
    } else if(mode==2) {
        name="Q8_Q5"; ek=encodeQ(k.data(),k.size(),8); ev=encodeQ(v.data(),v.size(),5);
        decodeQ(ek,rk.data(),rk.size(),8); decodeQ(ev,rv.data(),rv.size(),5);
    } else {
        name="Q8_Q4"; ek=encodeQ(k.data(),k.size(),8); ev=encodeQ(v.data(),v.size(),4);
        decodeQ(ek,rk.data(),rk.size(),8); decodeQ(ev,rv.data(),rv.size(),4);
    }

    const uint64_t pages=ceilDiv((uint64_t)requestedTokens,kPageTokens);
    const uint64_t storedTokens=pages*kPageTokens;
    const uint64_t elementsPerToken=kFullLayers*kKvHeads*kHeadDim;
    const uint64_t f16PerToken=elementsPerToken*kStreams*2;
    uint64_t bytesPerToken=0;
    if(mode==0) bytesPerToken=f16PerToken;
    else {
        const int kb=8, vb=(mode==1?8:(mode==2?5:4));
        const uint64_t blocksPerToken=ceilDiv(elementsPerToken,kBlock);
        bytesPerToken=elementsPerToken*(kb+vb)/8 + blocksPerToken*2*sizeof(float);
    }

    std::string out="OK ORNITH15_KV/2";
    out+=" mode="+std::string(name);
    out+=" page_tokens="+std::to_string(kPageTokens);
    out+=" pages="+std::to_string(pages);
    out+=" stored_tokens="+std::to_string(storedTokens);
    out+=" bytes_per_token_est="+std::to_string(bytesPerToken);
    out+=" bytes_est="+std::to_string(storedTokens*bytesPerToken);
    out+=" codec_payload_bytes="+std::to_string(ek.payload.size()+ev.payload.size());
    out+=" codec_scale_bytes="+std::to_string((ek.scales.size()+ev.scales.size())*sizeof(float));
    out+=" codec_test_elements_per_stream="+std::to_string(testElements);
    out+=" k_rmse="+std::to_string(rmse(k,rk));
    out+=" v_rmse="+std::to_string(rmse(v,rv));
    out+=" k_max_abs="+std::to_string(maxAbs(k,rk));
    out+=" v_max_abs="+std::to_string(maxAbs(v,rv));
    out+=" implementation=REAL_PAGE_CODEC";
    return out;
}

} // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_bslsjdk_mcnpu_Ornith15KvProbe_nativeProbe(JNIEnv* env,jclass,jint tokens,jint mode) {
    std::string s=probe(tokens,mode);
    return env->NewStringUTF(s.c_str());
}
