#include <jni.h>
#include <cstdint>
#include <fstream>
#include <string>
#include <cstring>

namespace {

struct RuntimeState {
    std::string path;
    uint64_t fileBytes = 0;
    uint64_t context = 0;
    uint64_t blocks = 0;
    uint64_t hidden = 0;
    uint64_t vocab = 0;
    std::string arch;
    bool loaded = false;
} g;

static bool readU32(std::ifstream &f, uint32_t &v) {
    unsigned char b[4];
    if (!f.read(reinterpret_cast<char *>(b), 4)) return false;
    v=(uint32_t)b[0]|((uint32_t)b[1]<<8)|((uint32_t)b[2]<<16)|((uint32_t)b[3]<<24);
    return true;
}
static bool readU64(std::ifstream &f, uint64_t &v) {
    unsigned char b[8];
    if (!f.read(reinterpret_cast<char *>(b), 8)) return false;
    v=0;
    for(int i=0;i<8;i++) v|=((uint64_t)b[i])<<(8*i);
    return true;
}
static bool skipString(std::ifstream &f, std::string *out=nullptr) {
    uint64_t n;
    if(!readU64(f,n) || n>(1u<<20)) return false;
    std::string s((size_t)n,'\\0');
    if(n && !f.read(s.data(),(std::streamsize)n)) return false;
    if(out) *out=std::move(s);
    return true;
}
static bool skipValue(std::ifstream &f, uint32_t type, int depth=0) {
    if(depth>4) return false;
    uint64_t n;
    switch(type) {
        case 0: case 1: case 7: return f.ignore(1).good();
        case 2: case 3: return f.ignore(2).good();
        case 4: case 5: case 6: return f.ignore(4).good();
        case 8: return skipString(f);
        case 9: {
            uint32_t elem; uint64_t count;
            return readU32(f,elem) && readU64(f,count) && count<=1000000 &&
                   [&]() { for(uint64_t i=0;i<count;i++) if(!skipValue(f,elem,depth+1)) return false; return true; }();
        }
        case 10: case 11: case 12: return f.ignore(8).good();
        default: return false;
    }
}

static std::string loadModel(const std::string &path, uint64_t requested) {
    std::ifstream f(path, std::ios::binary|std::ios::ate);
    if(!f) return "ERR ORNITH15_RUNTIME open_failed";
    const std::streamoff end=f.tellg();
    if(end<24) return "ERR ORNITH15_RUNTIME file_too_small";
    g.fileBytes=(uint64_t)end;
    f.seekg(0);

    char magic[4];
    if(!f.read(magic,4) || std::memcmp(magic,"GGUF",4)!=0)
        return "ERR ORNITH15_RUNTIME bad_magic";
    uint32_t version; uint64_t tensors,kvs;
    if(!readU32(f,version) || (version!=2 && version!=3) ||
       !readU64(f,tensors) || !readU64(f,kvs) || kvs>100000)
        return "ERR ORNITH15_RUNTIME bad_header";

    g.arch=""; g.blocks=g.hidden=g.vocab=0;
    for(uint64_t i=0;i<kvs;i++) {
        std::string key; uint32_t type;
        if(!skipString(f,&key) || !readU32(f,type))
            return "ERR ORNITH15_RUNTIME bad_kv";
        if(type==8) {
            std::string value;
            if(!skipString(f,&value)) return "ERR ORNITH15_RUNTIME bad_string";
            if(key=="general.architecture") g.arch=value;
        } else if(type==4 || type==5 || type==10 || type==11) {
            uint64_t value=0;
            if(type==4 || type==5) { uint32_t x; if(!readU32(f,x)) return "ERR ORNITH15_RUNTIME bad_u32"; value=x; }
            else if(!readU64(f,value)) return "ERR ORNITH15_RUNTIME bad_u64";
            if(key=="qwen35.block_count") g.blocks=value;
            if(key=="qwen35.embedding_length") g.hidden=value;
            if(key=="qwen35.vocab_size") g.vocab=value;
            if(key=="qwen35.context_length") g.context=value;
        } else {
            if(!skipValue(f,type)) return "ERR ORNITH15_RUNTIME bad_value";
        }
    }

    if(g.arch!="qwen35") return "ERR ORNITH15_RUNTIME arch="+(g.arch.empty()?"unknown":g.arch);
    if(g.blocks==0) return "ERR ORNITH15_RUNTIME missing_block_count";
    if(g.context==0) return "ERR ORNITH15_RUNTIME missing_context";
    if(requested>g.context) return "ERR ORNITH15_RUNTIME requested_context="+std::to_string(requested)+" native="+std::to_string(g.context);

    g.path=path; g.context=requested; g.loaded=true;
    return "OK ORNITH15_RUNTIME/1 arch="+g.arch+
           " layers="+std::to_string(g.blocks)+
           " hidden="+std::to_string(g.hidden)+
           " vocab="+std::to_string(g.vocab)+
           " context="+std::to_string(requested)+
           " file_bytes="+std::to_string(g.fileBytes)+
           " inference=NOT_ATTACHED";
}

} // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_bslsjdk_mcnpu_Ornith15Runtime_nativeLoad(JNIEnv* env,jclass,jstring jpath,jlong requested) {
    const char* p=env->GetStringUTFChars(jpath,nullptr);
    std::string s=loadModel(p?p:"",(uint64_t)requested);
    if(p) env->ReleaseStringUTFChars(jpath,p);
    return env->NewStringUTF(s.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_bslsjdk_mcnpu_Ornith15Runtime_nativeInfo(JNIEnv* env,jclass) {
    std::string s=g.loaded
        ? "OK ORNITH15_RUNTIME/1 loaded=true path="+g.path+" context="+std::to_string(g.context)
        : "OK ORNITH15_RUNTIME/1 loaded=false";
    return env->NewStringUTF(s.c_str());
}

extern "C" JNIEXPORT void JNICALL
Java_bslsjdk_mcnpu_Ornith15Runtime_nativeUnload(JNIEnv*,jclass) {
    g=RuntimeState{};
}
