#include <jni.h>
#include <cstdint>
#include <fstream>
#include <string>
#include <cstring>
#include <vector>
#include <algorithm>
#if MCNPU_HAS_LLAMA
#include "llama.h"
#endif

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
#if MCNPU_HAS_LLAMA
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    llama_sampler * sampler = nullptr;
#endif
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
#if MCNPU_HAS_LLAMA
    if (g.loaded) {
        if (g.sampler) { llama_sampler_free(g.sampler); g.sampler=nullptr; }
        if (g.ctx) { llama_free(g.ctx); g.ctx=nullptr; }
        if (g.model) { llama_model_free(g.model); g.model=nullptr; }
        g.loaded=false;
    }
#endif
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

#if MCNPU_HAS_LLAMA
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.use_mmap = true;
    g.model = llama_model_load_from_file(path.c_str(), mp);
    if (!g.model) return "ERR ORNITH15_RUNTIME llama_model_load_failed";
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t) requested;
    cp.n_batch = (uint32_t) std::min<uint64_t>(requested, 512);
    cp.n_ubatch = std::min<uint32_t>(cp.n_batch, 512);
    cp.n_seq_max = 1;
    g.ctx = llama_init_from_model(g.model, cp);
    if (!g.ctx) {
        llama_model_free(g.model); g.model=nullptr;
        return "ERR ORNITH15_RUNTIME llama_context_failed";
    }
    auto sp = llama_sampler_chain_default_params();
    sp.no_perf = true;
    g.sampler = llama_sampler_chain_init(sp);
    if (!g.sampler) {
        llama_free(g.ctx); g.ctx=nullptr;
        llama_model_free(g.model); g.model=nullptr;
        return "ERR ORNITH15_RUNTIME llama_sampler_failed";
    }
    llama_sampler_chain_add(g.sampler, llama_sampler_init_greedy());
#endif

    g.path=path; g.context=requested; g.loaded=true;
    return "OK ORNITH15_RUNTIME/1 arch="+g.arch+
           " layers="+std::to_string(g.blocks)+
           " hidden="+std::to_string(g.hidden)+
           " vocab="+std::to_string(g.vocab)+
           " context="+std::to_string(requested)+
           " file_bytes="+std::to_string(g.fileBytes)+
           " inference=LLAMA_CPU_BASELINE";
}

static std::string generateModel(const std::string &prompt, int maxTokens) {
#if !MCNPU_HAS_LLAMA
    return "ERR ORNITH15_RUNTIME llama_backend_not_compiled";
#else
    if (!g.loaded || !g.model || !g.ctx || !g.sampler)
        return "ERR ORNITH15_RUNTIME not_loaded";
    if (maxTokens <= 0 || maxTokens > 4096)
        return "ERR ORNITH15_RUNTIME bad_max_tokens";
    const llama_vocab * vocab = llama_model_get_vocab(g.model);
    int n = -llama_tokenize(vocab, prompt.c_str(), (int32_t)prompt.size(), nullptr, 0, true, true);
    if (n <= 0) return "ERR ORNITH15_RUNTIME tokenize_failed";
    std::vector<llama_token> tokens((size_t)n);
    if (llama_tokenize(vocab, prompt.c_str(), (int32_t)prompt.size(), tokens.data(), n, true, true) < 0)
        return "ERR ORNITH15_RUNTIME tokenize_failed";
    llama_sampler_reset(g.sampler);
    llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t)tokens.size());
    if (llama_decode(g.ctx, batch) != 0)
        return "ERR ORNITH15_RUNTIME prompt_decode_failed";
    std::string out;
    out.reserve((size_t)maxTokens * 4);
    for (int i=0; i<maxTokens; ++i) {
        llama_token tok = llama_sampler_sample(g.sampler, g.ctx, -1);
        if (llama_vocab_is_eog(vocab, tok)) break;
        char buf[4096];
        int m = llama_token_to_piece(vocab, tok, buf, (int32_t)sizeof(buf), 0, false);
        if (m < 0) return "ERR ORNITH15_RUNTIME token_to_piece_failed";
        out.append(buf, (size_t)m);
        batch = llama_batch_get_one(&tok, 1);
        if (llama_decode(g.ctx, batch) != 0)
            return "ERR ORNITH15_RUNTIME decode_failed";
    }
    return "OK ORNITH15_GENERATE/1 text=" + out;
#endif
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
Java_bslsjdk_mcnpu_Ornith15Runtime_nativeGenerate(JNIEnv* env,jclass,jstring jprompt,jint maxTokens) {
    const char* p=env->GetStringUTFChars(jprompt,nullptr);
    std::string s=generateModel(p?p:"",(int)maxTokens);
    if(p) env->ReleaseStringUTFChars(jprompt,p);
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
#if MCNPU_HAS_LLAMA
    if (g.sampler) { llama_sampler_free(g.sampler); g.sampler=nullptr; }
    if (g.ctx) { llama_free(g.ctx); g.ctx=nullptr; }
    if (g.model) { llama_model_free(g.model); g.model=nullptr; }
#endif
    g=RuntimeState{};
}
