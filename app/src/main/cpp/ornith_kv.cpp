#include <jni.h>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

namespace {

enum class KvMode : int { F16 = 0, Q8Q8 = 1, Q8Q5 = 2, Q8Q4 = 3 };

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

struct BlockQ8 {
    float scale=1.f;
    std::vector<int8_t> q;
};

struct BlockQ5 {
    float scale=1.f;
    std::vector<uint8_t> q;
};

static BlockQ8 quant8(const float* p, size_t n) {
    BlockQ8 b; b.q.resize(n);
    float mx=0.f; for(size_t i=0;i<n;i++) mx=std::max(mx,std::fabs(p[i]));
    b.scale = mx>0.f ? mx/127.f : 1.f;
    for(size_t i=0;i<n;i++) {
        float z=p[i]/b.scale; z=std::max(-128.f,std::min(127.f,z));
        b.q[i]=(int8_t)std::lrint(z);
    }
    return b;
}

static BlockQ5 quant5(const float* p, size_t n) {
    BlockQ5 b; b.q.resize(n);
    float mx=0.f; for(size_t i=0;i<n;i++) mx=std::max(mx,std::fabs(p[i]));
    b.scale = mx>0.f ? mx/15.f : 1.f;
    for(size_t i=0;i<n;i++) {
        float z=p[i]/b.scale; z=std::max(-16.f,std::min(15.f,z));
        b.q[i]=(uint8_t)(int(std::lrint(z))+16);
    }
    return b;
}

static float maxAbs(const std::vector<float>& a,const std::vector<float>& b) {\n    float m=0.f; for(size_t i=0;i<a.size();i++) m=std::max(m,std::fabs(a[i]-b[i])); return m;\n}\n\nstatic float rmse(const std::vector<float>& a,const std::vector<float>& b) {
    long double s=0; for(size_t i=0;i<a.size();i++){long double d=a[i]-b[i];s+=d*d;}
    return (float)std::sqrt((double)(s/a.size()));
}

static uint64_t ceilDiv(uint64_t a,uint64_t b){return b?((a+b-1)/b):0;}

static std::string probe(int requestedTokens, int mode) {
    if(requestedTokens<=0 || requestedTokens>262144) return "ERR ORNITH15_KV bad_tokens";
    if(mode<0 || mode>3) return "ERR ORNITH15_KV bad_mode";
    constexpr uint64_t kFullLayers=8, kKvHeads=4, kHeadDim=256, kStreams=2;
    constexpr uint64_t kPageTokens=256;
    constexpr size_t kProbeElements=4096;

    std::vector<float> src(kProbeElements);
    for(size_t i=0;i<src.size();++i)
        src[i]=std::sin((float)i*0.017f)*1.7f+std::cos((float)i*0.0031f)*0.23f;

    std::vector<float> restoredK(src.size()), restoredV(src.size());
    uint64_t payload=0, metadata=0;
    const char* modeName="";
    if(mode==0) {
        modeName="F16";
        std::vector<uint16_t> q(src.size());
        for(size_t i=0;i<src.size();++i) q[i]=f32_to_f16(src[i]), restoredK[i]=f16_to_f32(q[i]), restoredV[i]=restoredK[i];
        payload=(uint64_t)kProbeElements*2;
    } else if(mode==1) {
        modeName="Q8_Q8";
        auto k=quant8(src.data(),src.size()), v=quant8(src.data(),src.size());
        for(size_t i=0;i<src.size();++i) { restoredK[i]=(float)k.q[i]*k.scale; restoredV[i]=(float)v.q[i]*v.scale; }
        payload=(uint64_t)k.q.size()*2; metadata=8;
        (void)v;
    } else if(mode==2) {
        modeName="Q8_Q5";
        auto k=quant8(src.data(),src.size()), v=quant5(src.data(),src.size());
        for(size_t i=0;i<src.size();++i) restored[i]=(float)k.q[i]*k.scale;
        payload=(uint64_t)k.q.size() + ceilDiv((uint64_t)v.q.size()*5,8); metadata=8+8;
    } else {
        modeName="Q8_Q4";
        auto k=quant8(src.data(),src.size()), v=quant5(src.data(),src.size());
        for(size_t i=0;i<src.size();++i) {
            int q=(int(v.q[i])-16); q=std::max(-8,std::min(7,q*8/15));
            restored[i]=(float)k.q[i]*k.scale;
        }
        payload=(uint64_t)k.q.size() + ceilDiv((uint64_t)v.q.size()*4,8); metadata=8+8;
    }

    const uint64_t f16PerToken=kFullLayers*kKvHeads*kHeadDim*kStreams*2;
    uint64_t bytesPerToken;
    if(mode==0) bytesPerToken=f16PerToken;
    else if(mode==1) bytesPerToken=f16PerToken/2 + 16;
    else if(mode==2) bytesPerToken=(f16PerToken*13)/16 + 16;
    else bytesPerToken=(f16PerToken*3)/8 + 16;

    const uint64_t pages=ceilDiv((uint64_t)requestedTokens,kPageTokens);
    const uint64_t storedTokens=pages*kPageTokens;
    const uint64_t estimated=storedTokens*bytesPerToken;

    std::string out="OK ORNITH15_KV/1";
    out+=" mode="+std::string(modeName);
    out+=" page_tokens="+std::to_string(kPageTokens);
    out+=" pages="+std::to_string(pages);
    out+=" stored_tokens="+std::to_string(storedTokens);
    out+=" bytes_per_token_est="+std::to_string(bytesPerToken);
    out+=" bytes_est="+std::to_string(estimated);
    out+=" f16_bytes_per_token="+std::to_string(f16PerToken);
    out+=" probe_payload_bytes="+std::to_string(payload);
    out+=" probe_metadata_bytes="+std::to_string(metadata);
    out+=" probe_k_rmse="+std::to_string(rmse(src,restoredK));\n    out+=" probe_v_rmse="+std::to_string(rmse(src,restoredV));\n    out+=" probe_k_max_abs="+std::to_string(maxAbs(src,restoredK));\n    out+=" probe_v_max_abs="+std::to_string(maxAbs(src,restoredV));
    out+=" implementation=PAGE_CODEC_PROBE_ONLY";
    return out;
}

} // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_bslsjdk_mcnpu_Ornith15KvProbe_nativeProbe(JNIEnv* env,jclass,jint tokens,jint mode) {
    std::string s=probe(tokens,mode);
    return env->NewStringUTF(s.c_str());
}
