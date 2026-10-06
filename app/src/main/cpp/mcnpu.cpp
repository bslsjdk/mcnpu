#include <jni.h>
#include <dlfcn.h>
#include <android/log.h>
#include <chrono>
#include <cstdlib>
#include <string>
#include <vector>
#include <unistd.h>
#include <cerrno>
#include <cstdio>
#include <fstream>
#include <sys/stat.h>
#include <cstring>
#include <cmath>
#include <mutex>
#include <atomic>
#include <algorithm>
#include <unordered_map>
#include <cstdint>
#include <sched.h>
#include <deque>
#include <map>
#include "QnnInterface.h"
#include "QnnLog.h"
#include "QnnBackend.h"
#include "QnnDevice.h"
#include "QnnContext.h"
#include "QnnGraph.h"
#include "QnnTensor.h"
#include "QnnTypes.h"
#include "QnnOpDef.h"

#define TAG "MCNPU"
#define HTP_ID 6
#define I(...) __android_log_print(ANDROID_LOG_INFO,TAG,__VA_ARGS__)
#define E(...) __android_log_print(ANDROID_LOG_ERROR,TAG,__VA_ARGS__)

namespace {
struct Runtime {
    uint64_t diagCount=0;
    uint64_t graphSeq=0;
    int graphCount=0;
    struct AddGraph {
        Qnn_GraphHandle_t graph=nullptr;
        uint32_t dims[1]={0};
        Qnn_Tensor_t a=QNN_TENSOR_INIT;
        Qnn_Tensor_t b=QNN_TENSOR_INIT;
        Qnn_Tensor_t c=QNN_TENSOR_INIT;
        bool fp16=false;
    };
    struct MatMulGraph {
        Qnn_GraphHandle_t graph=nullptr;
        uint32_t m=0,k=0,n=0;
        uint32_t dimsA[2]={0,0}, dimsB[2]={0,0}, dimsC[2]={0,0};
        Qnn_Tensor_t a=QNN_TENSOR_INIT, b=QNN_TENSOR_INIT, c=QNN_TENSOR_INIT;
        bool fp16=false;
        // Binary int8 path: the requantisation factor is a property of the graph,
        // not of the data, so the host-side calibration only has to run once per
        // shape. Caching it removes an O(refRows*n*k) CPU triple loop from every
        // later call (that loop alone cost ~90ms at 512^3).
        float scaleEff=0.02f;
        bool calibrated=false;
    };
    // Keyed on (n, fp16). Keying on n alone would hand an fp32 graph to an
    // fp16 call and the buffer would be read at half the element size.
    std::unordered_map<uint64_t, AddGraph> addGraphs;
    std::unordered_map<uint64_t, MatMulGraph> matMulGraphs;
    std::unordered_map<uint64_t, MatMulGraph> matMulGraphs8;
    void* qnn=nullptr;
    const QnnInterface_t* api=nullptr;
    Qnn_BackendHandle_t backend=nullptr;
    Qnn_DeviceHandle_t device=nullptr;
    Qnn_ContextHandle_t context=nullptr;
    bool ready=false;
    std::string info;
    std::string err;
    Qnn_LogHandle_t logger=nullptr;
    std::string libDir;
    std::string loadError;
    uint32_t providerCount=0;
    uint32_t selectedBackend=0;
    Qnn_ErrorHandle_t backendRc=QNN_SUCCESS;
    Qnn_ErrorHandle_t deviceRc=QNN_SUCCESS;
    std::string backendVerbose;
    std::string deviceVerbose;
} g;
static thread_local std::string g_lastNativeError;
static std::mutex gRuntimeMutex;

using GetProviders = Qnn_ErrorHandle_t (*)(const QnnInterface_t ***,uint32_t *);

static std::string errnoText(int e) { char buf[128] = {}; strerror_r(e, buf, sizeof(buf)); return std::string(buf); }

static std::string envv(const char* n) {
    const char* v=getenv(n);
    return v?v:"<unset>";
}

static std::string statFile(const std::string& p) {
    struct stat st{};
    if(stat(p.c_str(),&st)!=0)
        return "MISSING errno="+std::to_string(errno)+"("+errnoText(errno)+")";
    return "OK mode="+std::to_string((unsigned)(st.st_mode&07777))+
           " size="+std::to_string((long long)st.st_size);
}


static std::string probeSystemRpc() {
    const char* paths[] = {
        "/vendor/lib64/libcdsprpc.so",
        "/vendor/lib64/libadsprpc.so",
        "/vendor/lib64/libsdsprpc.so",
        "/vendor/lib/rfsa/adsp/libcdsprpc.so",
        "/vendor/lib/rfsa/adsp/libadsprpc.so",
        "/system/lib64/libcdsprpc.so",
        "/system/lib64/libadsprpc.so",
        "/odm/lib64/libcdsprpc.so",
        "/odm/lib64/libadsprpc.so"
    };
    std::string out;
    for (const char* p : paths) {
        struct stat st{};
        int rc=stat(p,&st);
        out += std::string("path=")+p+" stat="+(rc==0?"FOUND":"MISSING")+
               " errno="+std::to_string(rc==0?0:errno)+" size="+(rc==0?std::to_string((long long)st.st_size):"0")+"\n";
        if(rc==0) {
            // Diagnostic probe only: stat() confirms presence without creating
            // a transient loader handle that could affect QNN/RPC lifetimes.
            out += "  dlopen=SKIPPED (stat-only probe)\n";
        }
    }
    return out;
}

static std::string mapsSummary() {
    std::ifstream in("/proc/self/maps");
    if(!in) return "maps=UNREADABLE";
    int qnn=0, rpc=0, stub=0, skel=0, prepare=0;
    std::string line;
    while(std::getline(in,line)) {
        if(line.find("libQnnHtp.so")!=std::string::npos) qnn++;
        if(line.find("libQnnHtpV73Stub.so")!=std::string::npos) stub++;
        if(line.find("libQnnHtpV73Skel.so")!=std::string::npos) skel++;
        if(line.find("libQnnHtpPrepare.so")!=std::string::npos) prepare++;
        if(line.find("rpc")!=std::string::npos || line.find("Rpc")!=std::string::npos) rpc++;
    }
    return "maps_qnnhtp="+std::to_string(qnn)+" maps_stub="+std::to_string(stub)+
           " maps_skel="+std::to_string(skel)+" maps_prepare="+std::to_string(prepare)+
           " maps_rpc="+std::to_string(rpc);
}

static std::string mapsForQnn() {
    std::ifstream in("/proc/self/maps");
    if(!in) return "MAPS_UNREADABLE";
    std::string line,out;
    while(std::getline(in,line)) {
        if(line.find("Qnn")!=std::string::npos ||
           line.find("qnn")!=std::string::npos ||
           line.find("rpc")!=std::string::npos ||
           line.find("cdsp")!=std::string::npos) {
            out += line+"\n";
            if(out.size()>10000) break;
        }
    }
    return out.empty()?"<no QNN/RPC mappings>":out;
}

static std::string verbose(Qnn_ErrorHandle_t rc) {
    if(!g.api) return "";
    const auto& f=g.api->QNN_INTERFACE_VER_NAME;
    const char* msg=nullptr;
    if(f.errorGetVerboseMessage) {
        Qnn_ErrorHandle_t vr=f.errorGetVerboseMessage(rc,&msg);
        if(msg && *msg)
            return std::string("verbose_rc=")+std::to_string((int)vr)+" msg="+msg;
        return "verbose_rc="+std::to_string((int)vr)+" msg=<empty>";
    }
    return "errorGetVerboseMessage=<unavailable>";
}

static std::string deepReport() {
    std::string r;
    const auto diagStart = std::chrono::steady_clock::now();
    const uint64_t diagNo = ++g.diagCount;
    r += "MCNPU_DEEP_DIAGNOSTIC\n";
    r += "diag_no="+std::to_string(diagNo)+"\n";
    r += "uid="+std::to_string((int)getuid())+" euid="+std::to_string((int)geteuid())+
         " pid="+std::to_string((int)getpid())+"\n";
    r += "cwd=";
    char cwd[1024];
    r += getcwd(cwd,sizeof(cwd))?cwd:"<getcwd failed>";
    r += "\n";
    r += "libDir="+g.libDir+"\n";
    r += "LD_LIBRARY_PATH="+envv("LD_LIBRARY_PATH")+"\n";
    r += "ADSP_LIBRARY_PATH="+envv("ADSP_LIBRARY_PATH")+"\n";
    r += "MCNPU_TUNING="+envv("MCNPU_TUNING")+"\n";
    r += "qnnHandle="+std::to_string((uintptr_t)g.qnn)+" dlerror="+(g.loadError.empty()?"<none>":g.loadError)+"\n";
    if(!g.libDir.empty()) {
        const char* libs[]={"libQnnHtp.so","libQnnHtpV73.so","libQnnHtpV73Stub.so",
                            "libQnnHtpV73Skel.so","libQnnHtpPrepare.so","libQnnSystem.so",
                            "libcdsprpc.so","libadsprpc.so"};
        for(const char* n:libs) r += std::string(n)+" "+statFile(g.libDir+"/"+n)+"\n";
    }
    r += "providers="+std::to_string(g.providerCount)+" selectedBackendId="+std::to_string(g.selectedBackend)+"\n";
    r += "backendHandle="+std::to_string((uintptr_t)g.backend)+" rc="+std::to_string((int)g.backendRc)+
         " "+g.backendVerbose+"\n";
    r += "deviceHandle="+std::to_string((uintptr_t)g.device)+" rc="+std::to_string((int)g.deviceRc)+
         " "+g.deviceVerbose+"\n";
    r += "contextHandle="+std::to_string((uintptr_t)g.context)+" ready="+(g.ready?"true":"false")+"\n";
    r += "maps_summary="+mapsSummary()+"\n";
    r += "SYSTEM_RPC_PROBE:\n"+probeSystemRpc();
    r += "QNN/RPC memory mappings:\n"+mapsForQnn();
    const auto diagMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-diagStart).count();
    r += "diag_time_ms="+std::to_string((long long)diagMs)+"\n";
    return r;
}

bool loadRuntime(const std::string& qnnDir, const std::string& workDir) {
    g.libDir=qnnDir;
    struct stat qnnStat{};
    if(g.libDir.empty() || stat(g.libDir.c_str(), &qnnStat)!=0 || !S_ISDIR(qnnStat.st_mode)){
        g.err="qnnDir invalid: "+g.libDir+" errno="+std::to_string(errno)+"("+errnoText(errno)+")";
        return false;
    }

    std::string adsp=g.libDir+";/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp";
    setenv("ADSP_LIBRARY_PATH",adsp.c_str(),1);
    setenv("LD_LIBRARY_PATH",(g.libDir+":/vendor/dsp/cdsp:/vendor/lib64/").c_str(),1);
    if(workDir.empty() || chdir(workDir.c_str())!=0){
        g.err="chdir workDir failed errno="+std::to_string(errno)+"("+errnoText(errno)+")";
        return false;
    }

    g.qnn=dlopen((g.libDir+"/libQnnHtp.so").c_str(),RTLD_NOW|RTLD_GLOBAL);
    if(!g.qnn){
        const char* x=dlerror();
        g.loadError=x?x:"<unknown>";
        g.err="QNN load failed: "+g.loadError;
        return false;
    }
    return true;
}

bool initRuntime(const std::string& qnnDir, const std::string& workDir){
    std::lock_guard<std::mutex> lock(gRuntimeMutex);
    if(g.ready)return true;
    if(!loadRuntime(qnnDir, workDir))return false;
    auto gp=(GetProviders)dlsym(g.qnn,"QnnInterface_getProviders");
    if(!gp){
        g.err="QnnInterface_getProviders missing";
        return false;
    }
    const QnnInterface_t** providers=nullptr;
    uint32_t count=0;
    Qnn_ErrorHandle_t rc=gp(&providers,&count);
    g.providerCount=count;
    if(rc!=QNN_SUCCESS||!providers||count==0){
        g.err="getProviders rc="+std::to_string((int)rc)+" "+verbose(rc);
        return false;
    }
    for(uint32_t i=0;i<count;i++)
        if(providers[i]&&providers[i]->backendId==HTP_ID){
            g.api=providers[i];
            g.selectedBackend=providers[i]->backendId;
            break;
        }
    if(!g.api){
        g.err="HTP provider backendId=6 not found";
        return false;
    }

    const auto& f=g.api->QNN_INTERFACE_VER_NAME;
    if(f.logCreate){
        rc=f.logCreate(nullptr,QNN_LOG_LEVEL_VERBOSE,&g.logger);
        if(rc!=QNN_SUCCESS) g.logger=nullptr;
    }

    rc=f.backendCreate(g.logger,nullptr,&g.backend);
    g.backendRc=rc;
    if(rc!=QNN_SUCCESS||!g.backend){
        g.backendVerbose=verbose(rc);
        g.err="backendCreate rc="+std::to_string((int)rc)+" "+g.backendVerbose;
        return false;
    }

    rc=f.deviceCreate(g.logger,nullptr,&g.device);
    g.deviceRc=rc;
    if(rc!=QNN_SUCCESS||!g.device){
        g.deviceVerbose=verbose(rc);
        g.err="deviceCreate rc="+std::to_string((int)rc)+" "+g.deviceVerbose;
        E("DEVICE_CREATE_FAIL %s",g.err.c_str());
        const std::string report = deepReport();
        E("%s",report.c_str());
        return false;
    }

    rc=f.contextCreate(g.backend,g.device,nullptr,&g.context);
    if(rc!=QNN_SUCCESS||!g.context){
        g.err="contextCreate rc="+std::to_string((int)rc)+" "+verbose(rc);
        return false;
    }
    // Stay off the prime core. This SoC has exactly one (cpu7) and Minecraft's
    // main thread lives there; on-device measurements showed the service getting
    // 40% slower while C2ME saturated the other cores and fought for it. The NPU
    // maths runs on the DSP, so yielding the fastest core costs almost nothing
    // and makes the service far more predictable under load.
    {
        cpu_set_t aff;
        CPU_ZERO(&aff);
        const long nc=sysconf(_SC_NPROCESSORS_ONLN);
        for(long i=0;i<nc-1;i++) CPU_SET((int)i,&aff);
        if(nc>1 && sched_setaffinity(0,sizeof(aff),&aff)==0)
            I("AFFINITY cpu0-%ld (prime core excluded)",nc-2);
        else
            I("AFFINITY failed errno=%d",errno);
    }

    g.info="QNN HTP ready backendId=6 providers="+std::to_string(count);
    g.ready=true;
    I("MCNPU HTP READY");
    return true;
}

// QNN 2.27 cannot free a single graph; only contextFree() releases them.
// An unbounded set of graph sizes would therefore leak device memory, so ADD
// is restricted to a fixed bucket list with one cached graph per bucket.
// Explicit IEEE-754 binary16 conversion. The compiler's __fp16 may map to the
// ARM alternative format (different bit layout from IEEE), which makes QNN read
// garbage. These helpers are pure integer/bit manipulation.
static inline uint16_t f2h(float f){
    uint32_t x; std::memcpy(&x,&f,4);
    const uint32_t sign=(x>>16)&0x8000u;
    const uint32_t exp=(x>>23)&0xffu;
    uint32_t m=x&0x7fffffu;
    if(exp==0xffu) return (uint16_t)(sign|(m?0x7e00u:0x7c00u));
    int32_t e=(int32_t)exp-127+15;
    if(e>=31) return (uint16_t)(sign|0x7c00u);
    if(e<=0){
        if(e<-10) return (uint16_t)sign;
        m|=0x800000u;
        const uint32_t shift=(uint32_t)(14-e);
        uint32_t sub=m>>(shift+13);
        if((m>>(shift+12))&1u) sub+=1u;
        return (uint16_t)(sign|sub);
    }
    uint16_t h=(uint16_t)(sign|((uint32_t)e<<10)|(m>>13));
    const uint32_t rem=m&0x1fffu;
    if(rem>0x1000u || (rem==0x1000u && (h&1u))) h++;
    return h;
}
static inline float h2f(uint16_t u){
    const uint32_t sign=((uint32_t)u&0x8000u)<<16;
    uint32_t e=((uint32_t)u>>10)&0x1fu;
    uint32_t m=(uint32_t)u&0x3ffu;
    uint32_t x;
    if(e==0){
        if(m==0){ x=sign; }
        else {
            int32_t ee=-14;
            while(!(m&0x400u)){ m<<=1; ee--; }
            m&=0x3ffu;
            x=sign|((uint32_t)(ee+127)<<23)|(m<<13);
        }
    } else if(e==31){
        x=sign|0x7f800000u|(m<<13);
    } else {
        x=sign|(((uint32_t)e-15u+127u)<<23)|(m<<13);
    }
    float f; std::memcpy(&f,&x,4); return f;
}

// Fixed scales so the deterministic benchmark stays in range. INT8 with explicit
// quantize params is the only matmul path HTP computes natively and correctly.
// Forward declaration: makeTensorN is defined further down this file.
Qnn_Tensor_t makeTensorN(const char* name,Qnn_TensorType_t type,Qnn_DataType_t dt,uint32_t* dims,uint32_t rank);

static const float Q_SCALE_A = 0.001f;
static const float Q_SCALE_B = 0.005f;
static const float Q_SCALE_C = 0.02f;

Qnn_Tensor_t makeTensorQ(const char* name,Qnn_TensorType_t type,Qnn_DataType_t dt,uint32_t* dims,uint32_t rank,float scale){
    Qnn_Tensor_t t=makeTensorN(name,type,dt,dims,rank);
    Qnn_QuantizeParams_t q=QNN_QUANTIZE_PARAMS_INIT;
    q.encodingDefinition=QNN_DEFINITION_DEFINED;
    q.quantizationEncoding=QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
    q.scaleOffsetEncoding.scale=scale;
    q.scaleOffsetEncoding.offset=0;
    t.v1.quantizeParams=q;
    return t;
}

static inline int8_t quantize8(float v,float scale){
    float q=v/scale;
    if(q>127.f) q=127.f;
    if(q<-128.f) q=-128.f;
    return (int8_t)(q>=0.f ? (int)(q+0.5f) : (int)(q-0.5f));
}

// Largest bucket we will build a graph for. Declared here, above its first use,
// because a namespace-scope const must be declared before it is referenced; the
// bucket ladder itself lives further down next to bucketize().
// Keep this in sync with the last entry of MM_BUCKETS.
static const uint32_t MM_BUCKET_MAX = 65536;

// Forward declared: runBatchXform() uses it around 250 lines above where it is
// defined. Same rule as MM_BUCKET_MAX - a name must be declared before use.
static uint64_t checkedTensorBytes(uint32_t rows, uint32_t cols, uint32_t elemSize);

// Any reasonable shape is allowed (real workloads are not nice powers of two,
// e.g. 128 entities x 8 features x 16 outputs). Graphs are cached per shape.
static bool mmSizeAllowed(uint32_t v){ return v>=1 && v<=MM_BUCKET_MAX; }

// Defined further down; runBatchXform (declared above it) needs it.
static uint32_t bucketize(uint32_t v);

Qnn_Tensor_t makeTensorN(const char* name,Qnn_TensorType_t type,Qnn_DataType_t dt,uint32_t* dims,uint32_t rank){
    Qnn_Tensor_t t=QNN_TENSOR_INIT;
    t.version=QNN_TENSOR_VERSION_1;
    t.v1.name=name;t.v1.type=type;t.v1.dataFormat=QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    t.v1.dataType=dt;t.v1.rank=rank;t.v1.dimensions=dims;t.v1.memType=QNN_TENSORMEMTYPE_RAW;
    return t;
}

// QNN 2.27 has no graphFree: a graph lives until its context dies, and a
// failed graph cannot be reclaimed either. So the cache must be bounded, and
// when the budget is exhausted the whole context is rebuilt (which frees every
// graph at once). Without this, ~10 distinct shapes poison the context and
// every later call fails with rc=1007.
static const int MAX_CACHED_GRAPHS = 8;

static bool resetContextLocked(){
    if(!g.api) return false;
    const auto& f=g.api->QNN_INTERFACE_VER_NAME;
    g.addGraphs.clear();
    g.matMulGraphs.clear();
    g.matMulGraphs8.clear();
    if(f.contextFree && g.context) f.contextFree(g.context,nullptr);
    g.context=nullptr;
    // Log why this happened. A silent teardown looks exactly like a slow call:
    // the caller sees one 800 ms submit and no indication that every cached graph
    // (and its calibration) was just thrown away.
    I("CONTEXT RESET: graph budget %d/%d exhausted, dropping every cached graph",
      g.graphCount, MAX_CACHED_GRAPHS);
    if(!f.contextCreate) { g.ready=false; g.err="contextCreate missing"; return false; }
    Qnn_ErrorHandle_t rc=f.contextCreate(g.backend,g.device,nullptr,&g.context);
    if(rc!=QNN_SUCCESS || !g.context){
        g.ready=false;
        g.err="contextCreate rc="+std::to_string((int)rc)+" "+verbose(rc);
        E("CONTEXT RESET FAILED %s",g.err.c_str());
        return false;
    }
    g.graphCount=0;
    I("CONTEXT RECREATED: graph cache flushed");
    return true;
}

// Must be called while holding gRuntimeMutex.
static bool ensureGraphBudget(){
    if(g.graphCount < MAX_CACHED_GRAPHS) return true;
    return resetContextLocked();
}

// Last-resort recovery. resetContextLocked() drops the context and rebuilds it;
// if that rebuild returns an error it also clears g.ready, and every later call
// then fails with NPU_NOT_READY for the rest of the process lifetime. The backend
// and device handles survive (only the context is freed), so the same call can be
// retried. Without this one bad rebuild turns a recoverable hiccup into a
// permanently dead service that still reports itself as up.
static bool tryRecoverContextLocked(){
    if(!g.api || !g.backend || !g.device) return false;
    const auto& f=g.api->QNN_INTERFACE_VER_NAME;
    if(!f.contextCreate) return false;
    Qnn_ErrorHandle_t rc=f.contextCreate(g.backend,g.device,nullptr,&g.context);
    if(rc!=QNN_SUCCESS || !g.context){
        g.err="contextRecover rc="+std::to_string((int)rc)+" "+verbose(rc);
        E("CONTEXT RECOVER FAILED %s",g.err.c_str());
        return false;
    }
    g.addGraphs.clear();
    g.matMulGraphs.clear();
    g.matMulGraphs8.clear();
    g.graphCount=0;
    g.ready=true;
    I("CONTEXT RECOVERED: graph cache flushed");
    return true;
}

// Consecutive transport-class failures, counted across calls.
//
// rc=1007 is QNN_COMMON_ERROR_SYSTEM_COMMUNICATION (QNN_MIN_ERROR_COMMON(1000)+7):
// communication with the platform service failed, service recoverable. It is a
// COMMON error, not a GRAPH error, so QNN never rejected the graph - the FastRPC
// link to the DSP dropped underneath a perfectly valid one.
//
// That is precisely why it is invisible to the existing recovery: g.ready stays
// true and the context handle stays non-null, so the `!g.ready` branch above
// never fires. Every later call reuses the dead context and fails identically,
// which is the shape of the outage - tens of thousands of submits all returning
// the same rc while the service still advertised itself as UP. Rebuilding on a
// small run of these is what turns a dropped link back into working NPU.
static int g_dspFaults = 0;
static const int DSP_FAULT_REBUILD_THRESHOLD = 3;

static bool isTransportError(Qnn_ErrorHandle_t rc){
    const int v=(int)rc;
    return v>=1000 && v<2000;   // QNN_COMMON_ERROR_*: system/transport, not graph
}

// Must be called while holding gRuntimeMutex.
static void noteTransportFaultLocked(int rc){
    if(!isTransportError((Qnn_ErrorHandle_t)rc)){ g_dspFaults=0; return; }
    g_dspFaults++;
    if(g_dspFaults < DSP_FAULT_REBUILD_THRESHOLD) return;
    g_dspFaults=0;
    I("DSP LINK FAULT x%d rc=%d: rebuilding context (g.ready was still true)",
      DSP_FAULT_REBUILD_THRESHOLD, rc);
    resetContextLocked();
}

static void noteTransportSuccessLocked(){ g_dspFaults=0; }

// Smallest supported size that can hold n, or 0 when n is too large.
//
// runAdd builds one QNN graph per size and the graph cache is bounded, which is
// why a fixed set of sizes existed at all. Rejecting everything outside that set
// was the wrong way to enforce it: the caller learned only that it was wrong,
// never what would work, and a perfectly ordinary request simply died. Padding
// up to the next bucket keeps the graph count bounded while accepting any
// length. This is exactly what turned the 5x5 chunk-load simulation into
// pass=0/100 - every one of its cases carried 8 values.
// Base rungs. These are the sizes that have actually been seen to work on a
// Snapdragon 8s Gen 3, so they are trusted without a probe.
static const uint32_t ADD_LADDER_BASE[]={16,64,256,1024,4096,16384};

// Highest rung the ladder may pad to. 16384 is NOT a device limit - it is only
// the largest size we had verified, and it lives entirely in our own .so. The
// real ceiling is whatever the HTP accepts, which is what nativeAddProbe()
// measures on the device itself. Raising this by guesswork would be silently
// wrong on any phone whose ceiling is lower.
//
// Guarded by gRuntimeMutex.
static uint32_t g_addLadderMax = 16384;
// Same measurement for fp16 tensors. 0 until probed. The data path stays fp32
// for now, so this is reported but not adopted: switching the wire format is a
// separate decision and should be made with the numbers in hand, not before.

static uint32_t addPadSize(uint32_t n){
    for(uint32_t v:ADD_LADDER_BASE) if(n<=v) return v;
    if(n<=g_addLadderMax) return g_addLadderMax;
    return 0;
}

// Ladder as a comma-separated list, for diagnostics.
static std::string addLadderText(){
    std::string s;
    char buf[32];
    for(uint32_t v:ADD_LADDER_BASE){ if(!s.empty()) s+=","; snprintf(buf,sizeof buf,"%u",(unsigned)v); s+=buf; }
    if(g_addLadderMax>ADD_LADDER_BASE[5]){
        snprintf(buf,sizeof buf,"%u",(unsigned)g_addLadderMax);
        s+=","; s+=buf;
    }
    return s;
}


// IEEE-754 binary16 <-> binary32, no libm and no compiler-specific _Float16,
// because this .so is built for a different ABI than the QNN sample apps.
static uint16_t f32ToF16(float v){
    uint32_t x; memcpy(&x,&v,4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const int32_t  exp  = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;
    if (exp >= 31) return (uint16_t)(sign | 0x7C00u);          // inf / overflow
    if (exp <= 0) {                                            // zero / subnormal
        if (exp < -10) return (uint16_t)sign;                  // rounds to zero
        mant |= 0x800000u;                                     // implicit leading 1
        return (uint16_t)(sign | (mant >> (14 - exp)));
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}
static float f16ToF32(uint16_t h){
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (uint32_t)((h >> 10) & 0x1F);
    uint32_t mant = (uint32_t)(h & 0x3FF);
    uint32_t x;
    if (exp == 0) {
        if (mant == 0) x = sign;                               // +-0
        else {                                                 // subnormal -> normalise
            exp = 127 - 15 + 1;
            while (!(mant & 0x400)) { mant <<= 1; exp--; }
            mant &= 0x3FF;
            x = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        x = sign | 0x7F800000u | (mant << 13);                 // inf / NaN
    } else {
        x = sign | ((exp + 112) << 23) | (mant << 13);
    }
    float f; memcpy(&f,&x,4); return f;
}

// Defined below; the probe calls it once per candidate size. Declared without
// static to match the definition's linkage.
std::string runAddEx(const float* av,const float* bv,uint32_t n,float* out,bool verify,bool fp16=false);

// Walk candidate sizes on the real device and report which ones the HTP accepts.
//
// The point is to replace a guess with a measurement. Everything above 16384 is
// unknown territory: the sizes were never tried because the ladder rejected them
// before they could reach the device, so "the device caps ADD at 16384" was a
// conclusion our own code produced, not something the HTP ever said.
//
// Ascending, and it stops at the first failure: if 65536 cannot be built then
// 131072 will not either, and continuing would only burn graphs.
//
// Deliberately does NOT hold gRuntimeMutex. runAddEx takes it itself, and
// std::mutex is not recursive, so a probe that held the lock across those calls
// would deadlock on its own first candidate. The lock is taken only for the two
// short updates at the end.
// One ascending pass over the candidate sizes for one tensor datatype.
// Returns the largest size that built, and appends one line per candidate.
// Bounded on purpose.
//
// Every candidate costs one graph build and runAddEx holds gRuntimeMutex for the
// whole of it, so an unbounded walk is not merely slow: it parks every IPC
// handler on that mutex. A size the HTP stalls on then turns the service into a
// queue of handlers that never reply, which is the exact shape of the 9x9
// failure - each segment burned three 15 s client read timeouts, no reply ever
// arrived, and ADD_PROBE was never logged because the probe had not finished, so
// it had nothing to log yet.
//
// Three guards, any of which ends the walk: the candidate list stops at 65536,
// one step past the largest size actually verified; a wall-clock budget; and
// g_probeAbort, which the service sets the moment real work arrives.
static std::atomic<bool> g_probeAbort{false};
static const long long PROBE_TOTAL_BUDGET_US = 20LL * 1000 * 1000;

static uint32_t probeAddPass(bool fp16, std::string& lines, long long& budgetUs){
    static const uint32_t cand[]={16,64,256,1024,4096,16384,32768,65536};
    uint32_t best=0;
    char buf[256];
    for(uint32_t c:cand){
        if(g_probeAbort.load()){
            snprintf(buf,sizeof buf," %s ABORTED before %u\n", fp16?"fp16":"fp32",(unsigned)c);
            lines += buf;
            break;
        }
        if(budgetUs<=0){
            snprintf(buf,sizeof buf," %s BUDGET_EXHAUSTED before %u\n", fp16?"fp16":"fp32",(unsigned)c);
            lines += buf;
            break;
        }
        std::vector<float> a(c,1.f), b(c,2.f), o(c,-999.f);
        long long t0=std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        std::string r=runAddEx(a.data(),b.data(),c,o.data(),false,fp16);
        long long us=std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count()-t0;
        budgetUs -= us;
        bool ok = r.rfind("OK",0)==0;
        snprintf(buf,sizeof buf," %s %u %s elapsed_us=%lld",
                 fp16?"fp16":"fp32",(unsigned)c, ok?"ok":"FAIL", us);
        lines += buf;
        if(ok){
            // Surface the timings that matter: create is paid once per size,
            // execute is paid on every call afterwards.
            size_t p=r.find("create_us=");
            if(p!=std::string::npos){ size_t e=r.find_first_of(' ',p); if(e==std::string::npos) e=r.size(); lines += " "+r.substr(p,e-p); }
            p=r.find("execute_us=");
            if(p!=std::string::npos){ size_t e=r.find_first_of(' ',p); if(e==std::string::npos) e=r.size(); lines += " "+r.substr(p,e-p); }
            best=c;
        }else{
            lines += " "; lines += r;
            break;
        }
        lines += "\n";
    }
    return best;
}

static std::string probeAddLadder(){
    std::string lines;
    char buf[256];

    // fp32 first: it is the datatype the data path actually uses, so its result
    // is the one that may be adopted.
    g_probeAbort.store(false);
    long long budgetUs = PROBE_TOTAL_BUDGET_US;
    uint32_t best32 = probeAddPass(false, lines, budgetUs);
    // fp16 used to be probed here as a comparison. It is gone deliberately: the
    // datatype is the HTP native one and it did accept larger sizes, but its
    // elapsed time exploded - 473 ms at n=16384 and 25 s at n=32768 against an
    // execute of about 2 ms, so the cost is entirely in the conversion path and
    // grows faster than linearly. Probing it cost 26 s of the startup window on
    // every boot to produce a number that is already known and already rejected.

    {
        std::lock_guard<std::mutex> lock(gRuntimeMutex);
        if(best32>g_addLadderMax) g_addLadderMax=best32;
        // The probe just built one graph per candidate, which is exactly the
        // cache pressure the ladder exists to avoid. Flush before real work.
        resetContextLocked();
        snprintf(buf,sizeof buf,
                 "OK ADD_PROBE max_fp32=%u ladder=%s\n",
                 (unsigned)g_addLadderMax, addLadderText().c_str());
    }
    return std::string(buf)+lines;
}

Qnn_Tensor_t makeTensor(const char* name,Qnn_TensorType_t type,Qnn_DataType_t dt,uint32_t* dims){
    Qnn_Tensor_t t=QNN_TENSOR_INIT;
    t.version=QNN_TENSOR_VERSION_1;
    t.v1.name=name;t.v1.type=type;t.v1.dataFormat=QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    t.v1.dataType=dt;t.v1.rank=1;t.v1.dimensions=dims;t.v1.memType=QNN_TENSORMEMTYPE_RAW;
    return t;
}

// runAddEx writes the result into outBuf. verify runs the per-element compare
// against a CPU reference, which costs a full CPU pass over n floats - the same
// work the graph was supposed to replace. It belongs in tests, never in the data
// path, so it is opt-in here.
std::string runAddEx(const float* av,const float* bv,uint32_t n,float* out,bool verify,bool fp16){
    // fp16 halves the bytes the HTP has to move, and HTP is the one backend
    // where fp16 is native rather than emulated. The conversion is a host-side
    // pass over n floats, so it is opt-in: correct to measure, wrong to impose
    // on a caller whose data is already fp32.
    std::vector<uint16_t> a16,b16;
    std::vector<float> o32;
    if(fp16){
        a16.resize(n); b16.resize(n); o32.resize(n);
        for(uint32_t i=0;i<n;i++){ a16[i]=f32ToF16(av[i]); b16[i]=f32ToF16(bv[i]); }
    }
    std::lock_guard<std::mutex> lock(gRuntimeMutex);
    const auto total0=std::chrono::steady_clock::now();
    if(!g.ready || !g.api || !g.context) { g_lastNativeError="ERR NPU_NOT_READY"; return g_lastNativeError; }
    const auto& f=g.api->QNN_INTERFACE_VER_NAME;

    const uint64_t key=(uint64_t)n|(fp16?(1ULL<<40):0ULL);
    Runtime::AddGraph* ag=nullptr;
    bool cached=false;
    auto found=g.addGraphs.find(key);
    if(found!=g.addGraphs.end()){
        ag=&found->second;
        cached=true;
    }

    Qnn_ErrorHandle_t rc=QNN_SUCCESS;
    long long createUs=0, finalizeUs=0;

    if(!cached){
        // Insert the cache entry BEFORE creating graph tensors so every tensor
        // descriptor points at dimensions owned by the final cached object.
        if(!ensureGraphBudget()) return "ERR GRAPH_BUDGET_EXHAUSTED graphs="+std::to_string(g.graphCount)+"/"+std::to_string(MAX_CACHED_GRAPHS);
        g.graphCount++;
        auto inserted=g.addGraphs.emplace(key, Runtime::AddGraph{});
        ag=&inserted.first->second;
        ag->dims[0]=n;
        ag->fp16=fp16;

        const std::string graphName = "mcnpu_add_" + std::to_string(++g.graphSeq);

        auto tCreate0=std::chrono::steady_clock::now();
        rc=f.graphCreate(g.context,graphName.c_str(),nullptr,&ag->graph);
        createUs=std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now()-tCreate0).count();
        if(rc!=QNN_SUCCESS || !ag->graph){
            g.addGraphs.erase(inserted.first);
            return "ERR GRAPH_CREATE rc="+std::to_string((int)rc)+
                   " create_us="+std::to_string(createUs)+" "+verbose(rc);
        }

        const Qnn_DataType_t adt = fp16 ? QNN_DATATYPE_FLOAT_16 : QNN_DATATYPE_FLOAT_32;
        ag->a=makeTensor("a",QNN_TENSOR_TYPE_APP_WRITE,adt,ag->dims);
        ag->b=makeTensor("b",QNN_TENSOR_TYPE_APP_WRITE,adt,ag->dims);
        ag->c=makeTensor("c",QNN_TENSOR_TYPE_APP_READ, adt,ag->dims);

        rc=f.tensorCreateGraphTensor(ag->graph,&ag->a);
        if(rc==QNN_SUCCESS) rc=f.tensorCreateGraphTensor(ag->graph,&ag->b);
        if(rc==QNN_SUCCESS) rc=f.tensorCreateGraphTensor(ag->graph,&ag->c);
        if(rc!=QNN_SUCCESS){
            // QNN 2.27 exposes no graphFree in QnnInterface. The graph is
            // owned by the context and is released by contextFree.
            g.addGraphs.erase(key);
            E("ADD TENSOR_CREATE_FAIL n=%u rc=%d %s",(unsigned)n,(int)rc,verbose(rc).c_str());
            return "ERR TENSOR_CREATE rc="+std::to_string((int)rc)+" "+verbose(rc);
        }

        Qnn_Scalar_t scalar=QNN_SCALAR_INIT;
        scalar.dataType=QNN_DATATYPE_UINT_32;
        scalar.uint32Value=QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD;
        Qnn_Param_t param=QNN_PARAM_INIT;
        param.paramType=QNN_PARAMTYPE_SCALAR;
        param.name=QNN_OP_ELEMENT_WISE_BINARY_PARAM_OPERATION;
        param.scalarParam=scalar;
        Qnn_Tensor_t ins[2]={ag->a,ag->b};
        Qnn_OpConfig_t op=QNN_OPCONFIG_INIT;
        op.v1.name="add";
        op.v1.packageName="qti.aisw";
        op.v1.typeName=QNN_OP_ELEMENT_WISE_BINARY;
        op.v1.numOfParams=1;
        op.v1.params=&param;
        op.v1.numOfInputs=2;
        op.v1.inputTensors=ins;
        op.v1.numOfOutputs=1;
        op.v1.outputTensors=&ag->c;

        rc=f.graphAddNode(ag->graph,op);
        if(rc!=QNN_SUCCESS){
            g.addGraphs.erase(key);
            E("ADD GRAPH_NODE_FAIL n=%u rc=%d %s",(unsigned)n,(int)rc,verbose(rc).c_str());
            return "ERR GRAPH_NODE rc="+std::to_string((int)rc)+" "+verbose(rc);
        }

        auto tFinalize0=std::chrono::steady_clock::now();
        rc=f.graphFinalize(ag->graph,nullptr,nullptr);
        finalizeUs=std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now()-tFinalize0).count();
        if(rc!=QNN_SUCCESS){
            g.addGraphs.erase(key);
            E("ADD GRAPH_FINALIZE_FAIL n=%u rc=%d create_us=%lld finalize_us=%lld %s",
              (unsigned)n,(int)rc,createUs,finalizeUs,verbose(rc).c_str());
            return "ERR GRAPH_FINALIZE rc="+std::to_string((int)rc)+
                   " create_us="+std::to_string(createUs)+
                   " finalize_us="+std::to_string(finalizeUs)+" "+verbose(rc);
        }

        I("ADD GRAPH READY n=%u a_id=%u b_id=%u c_id=%u",
          (unsigned)n,(unsigned)ag->a.v1.id,(unsigned)ag->b.v1.id,(unsigned)ag->c.v1.id);
    }


    // QNN graphExecute requires the same tensor IDs assigned during
    // tensorCreateGraphTensor(). Reuse the registered descriptors and only
    // replace their client buffers for each execution.
    Qnn_Tensor_t ea=ag->a, eb=ag->b, ec=ag->c;
    const size_t elemBytes = fp16 ? sizeof(uint16_t) : sizeof(float);
    if(fp16){
        ea.v1.clientBuf.data=a16.data();
        eb.v1.clientBuf.data=b16.data();
        ec.v1.clientBuf.data=o32.data();
    }else{
        ea.v1.clientBuf.data=(void*)av;
        eb.v1.clientBuf.data=(void*)bv;
        ec.v1.clientBuf.data=out;
    }
    ea.v1.clientBuf.dataSize=n*elemBytes;
    eb.v1.clientBuf.dataSize=n*elemBytes;
    ec.v1.clientBuf.dataSize=n*elemBytes;
    Qnn_Tensor_t execIn[2]={ea,eb};
    Qnn_Tensor_t execOut[1]={ec};

    auto t0=std::chrono::steady_clock::now();
    rc=f.graphExecute(ag->graph,execIn,2,execOut,1,nullptr,nullptr);
    auto us=std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now()-t0).count();
    if(rc!=QNN_SUCCESS){
        noteTransportFaultLocked((int)rc);
        E("ADD GRAPH_EXECUTE_FAIL n=%u rc=%d %s",(unsigned)n,(int)rc,verbose(rc).c_str());
        return "ERR GRAPH_EXECUTE rc="+std::to_string((int)rc)+" "+verbose(rc);
    }

    // Tolerance rather than exact equality: the HTP may round through an
    // intermediate precision, and an exact compare would then report a perfectly
    // good execution as a failure. The first divergence is reported with both
    // values, which also distinguishes "slightly off" from "never written" - an
    // untouched output buffer reads back as the -999 fill, and that is visible
    // immediately instead of hiding behind a bare verify failure.
    noteTransportSuccessLocked();
    if(fp16) for(uint32_t i=0;i<n;i++) out[i]=f16ToF32(o32[i]);

    if(verify) for(uint32_t i=0;i<n;i++){
        float want=av[i]+bv[i];
        float got=out[i];
        float diff=got-want;
        if(diff<0.f) diff=-diff;
        float scale=want<0.f?-want:want;
        if(scale<1e-6f) scale=1e-6f;
        const float tol = fp16 ? 5e-3f : 1e-3f;
        if(diff > tol*scale){
            char vb[256];
            std::snprintf(vb,sizeof(vb),
                "ERR OUTPUT_VERIFY i=%u want=%.9g got=%.9g diff=%.9g",
                (unsigned)i,(double)want,(double)got,(double)diff);
            E("ADD OUTPUT_VERIFY n=%u %s",(unsigned)n,vb);
            return std::string(vb);
        }
    }

    auto totalUs=std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now()-total0).count();
    char buf[512];
    std::snprintf(buf,sizeof(buf),
        "OK HTP_GRAPH_EXECUTE graph_cached=%s n=%u dtype=%s create_us=%lld finalize_us=%lld execute_us=%lld total_us=%lld out0=%g out_last=%g",
        cached?"true":"false",(unsigned)n,fp16?"fp16":"fp32",createUs,finalizeUs,(long long)us,
        (long long)totalUs,(double)out[0],(double)out[n-1]);
    return buf;
}

std::string runAdd(const float* av,const float* bv,uint32_t n){
    std::vector<float> tmp(n,-999.f);
    return runAddEx(av,bv,n,tmp.data(),true);
}

// Deterministic matmul on HTP with an in-service CPU reference so the caller can
// see a real speedup number instead of a synthetic latency.
std::string runMatMul(uint32_t m,uint32_t k,uint32_t n,bool fp16){
    std::lock_guard<std::mutex> lock(gRuntimeMutex);
    const auto total0=std::chrono::steady_clock::now();
    if(!g.ready || !g.api || !g.context) return "ERR NPU_NOT_READY";
    const auto& f=g.api->QNN_INTERFACE_VER_NAME;
    const uint64_t key=((uint64_t)m<<42)|((uint64_t)k<<21)|(uint64_t)n|(fp16?(1ULL<<62):0ULL);
    Runtime::MatMulGraph* mg=nullptr;
    bool cached=false;
    auto found=g.matMulGraphs.find(key);
    if(found!=g.matMulGraphs.end()){ mg=&found->second; cached=true; }
    long long createUs=0, finalizeUs=0;
    Qnn_ErrorHandle_t rc=QNN_SUCCESS;
    if(!cached){
        if(!ensureGraphBudget()) return "ERR GRAPH_BUDGET_EXHAUSTED";
        g.graphCount++;
        auto inserted=g.matMulGraphs.emplace(key, Runtime::MatMulGraph{});
        mg=&inserted.first->second;
        mg->m=m; mg->k=k; mg->n=n; mg->fp16=fp16;
        // QNN dimensions are innermost-first (buffers are row-major).
        mg->dimsA[0]=k; mg->dimsA[1]=m;
        mg->dimsB[0]=n; mg->dimsB[1]=k;
        mg->dimsC[0]=n; mg->dimsC[1]=m;
        const std::string graphName="mcnpu_mm_"+std::to_string(++g.graphSeq);
        auto tCreate0=std::chrono::steady_clock::now();
        rc=f.graphCreate(g.context,graphName.c_str(),nullptr,&mg->graph);
        createUs=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-tCreate0).count();
        if(rc!=QNN_SUCCESS||!mg->graph){
            g.matMulGraphs.erase(inserted.first);
            return "ERR MM_GRAPH_CREATE rc="+std::to_string((int)rc)+" create_us="+std::to_string(createUs)+" "+verbose(rc);
        }
        const Qnn_DataType_t dt = fp16 ? QNN_DATATYPE_FLOAT_16 : QNN_DATATYPE_FLOAT_32;
        mg->a=makeTensorN("a",QNN_TENSOR_TYPE_APP_WRITE,dt,mg->dimsA,2);
        mg->b=makeTensorN("b",QNN_TENSOR_TYPE_APP_WRITE,dt,mg->dimsB,2);
        mg->c=makeTensorN("c",QNN_TENSOR_TYPE_APP_READ,dt,mg->dimsC,2);
        rc=f.tensorCreateGraphTensor(mg->graph,&mg->a);
        if(rc==QNN_SUCCESS) rc=f.tensorCreateGraphTensor(mg->graph,&mg->b);
        if(rc==QNN_SUCCESS) rc=f.tensorCreateGraphTensor(mg->graph,&mg->c);
        if(rc!=QNN_SUCCESS){
            g.matMulGraphs.erase(key);
            return "ERR MM_TENSOR_CREATE rc="+std::to_string((int)rc)+" "+verbose(rc);
        }
        Qnn_Tensor_t mmIns[2]={mg->a,mg->b};
        Qnn_OpConfig_t op=QNN_OPCONFIG_INIT;
        op.v1.name="matmul";
        op.v1.packageName="qti.aisw";
        op.v1.typeName=QNN_OP_MAT_MUL;
        op.v1.numOfParams=0;
        op.v1.params=nullptr;
        op.v1.numOfInputs=2;
        op.v1.inputTensors=mmIns;
        op.v1.numOfOutputs=1;
        op.v1.outputTensors=&mg->c;
        rc=f.graphAddNode(mg->graph,op);
        if(rc!=QNN_SUCCESS){
            g.matMulGraphs.erase(key);
            return "ERR MM_GRAPH_NODE rc="+std::to_string((int)rc)+" "+verbose(rc);
        }
        auto tF=std::chrono::steady_clock::now();
        rc=f.graphFinalize(mg->graph,nullptr,nullptr);
        finalizeUs=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-tF).count();
        if(rc!=QNN_SUCCESS){
            g.matMulGraphs.erase(key);
            return "ERR MM_GRAPH_FINALIZE rc="+std::to_string((int)rc)+" finalize_us="+std::to_string(finalizeUs)+" "+verbose(rc);
        }
        I("MATMUL GRAPH READY m=%u k=%u n=%u",(unsigned)m,(unsigned)k,(unsigned)n);
    }

    std::vector<float> A((size_t)m*k), B((size_t)k*n);
    for(size_t i=0;i<A.size();i++) A[i]=0.01f*(float)((i*7)%23)-0.1f;
    for(size_t i=0;i<B.size();i++) B[i]=0.05f*(float)((i*5)%17)-0.2f;

    // CPU reference (also the baseline for the speedup report).
    std::vector<float> Ccpu((size_t)m*n,0.f);
    auto tc0=std::chrono::steady_clock::now();
    for(uint32_t i=0;i<m;i++){
        for(uint32_t p=0;p<k;p++){
            const float av=A[(size_t)i*k+p];
            for(uint32_t j=0;j<n;j++) Ccpu[(size_t)i*n+j]+=av*B[(size_t)p*n+j];
        }
    }
    const long long cpuUs=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-tc0).count();

    std::vector<float> CnpuF((size_t)m*n,0.f);
    std::vector<uint16_t> Ah, Bh, Ch;
    if(fp16){
        Ah.resize(A.size()); Bh.resize(B.size()); Ch.assign(CnpuF.size(),0);
        for(size_t i=0;i<A.size();i++) Ah[i]=f2h(A[i]);
        for(size_t i=0;i<B.size();i++) Bh[i]=f2h(B[i]);
    }
    Qnn_Tensor_t ea=mg->a, eb=mg->b, ec=mg->c;
    if(fp16){
        ea.v1.clientBuf.data=Ah.data(); ea.v1.clientBuf.dataSize=(uint32_t)(Ah.size()*sizeof(uint16_t));
        eb.v1.clientBuf.data=Bh.data(); eb.v1.clientBuf.dataSize=(uint32_t)(Bh.size()*sizeof(uint16_t));
        ec.v1.clientBuf.data=Ch.data(); ec.v1.clientBuf.dataSize=(uint32_t)(Ch.size()*sizeof(uint16_t));
    } else {
        ea.v1.clientBuf.data=A.data();  ea.v1.clientBuf.dataSize=(uint32_t)(A.size()*sizeof(float));
        eb.v1.clientBuf.data=B.data();  eb.v1.clientBuf.dataSize=(uint32_t)(B.size()*sizeof(float));
        ec.v1.clientBuf.data=CnpuF.data(); ec.v1.clientBuf.dataSize=(uint32_t)(CnpuF.size()*sizeof(float));
    }
    Qnn_Tensor_t execIn[2]={ea,eb};
    Qnn_Tensor_t execOut[1]={ec};
    auto t0=std::chrono::steady_clock::now();
    rc=f.graphExecute(mg->graph,execIn,2,execOut,1,nullptr,nullptr);
    const long long execUs=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-t0).count();
    if(rc!=QNN_SUCCESS){
        noteTransportFaultLocked((int)rc);
        return "ERR MM_EXECUTE rc="+std::to_string((int)rc)+" "+verbose(rc);
    }
    noteTransportSuccessLocked();

    if(fp16) for(size_t i=0;i<CnpuF.size();i++) CnpuF[i]=h2f(Ch[i]);

    double maxAbs=0.0;
    int bad=0;
    const double relTol = fp16 ? 1e-2 : 1e-3;
    for(size_t i=0;i<CnpuF.size();i++){
        const double diff=std::fabs((double)CnpuF[i]-(double)Ccpu[i]);
        const double tol=relTol*std::max(1.0,std::fabs((double)Ccpu[i]));
        if(diff>maxAbs) maxAbs=diff;
        if(diff>tol) bad++;
    }
    const double speedup = execUs>0 ? (double)cpuUs/(double)execUs : 0.0;
    const long long totalUs=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-total0).count();
    char buf[640];
    std::snprintf(buf,sizeof(buf),
        "OK MATMUL m=%u k=%u n=%u dtype=%s cached=%s create_us=%lld finalize_us=%lld npu_exec_us=%lld cpu_us=%lld speedup=%.2fx bad=%d max_abs=%.5g total_us=%lld",
        (unsigned)m,(unsigned)k,(unsigned)n,fp16?"fp16":"fp32",cached?"true":"false",createUs,finalizeUs,execUs,cpuUs,speedup,bad,maxAbs,totalUs);
    return buf;
}

// Element-wise batch transform: NO accumulation, so none of the requantisation
// weirdness of matmul applies. C_int8 = op(A_int8, B_int8) with a single shared
// scale of 1/127. op 0 = add, 1 = multiply (scaled by 1/127 to stay in range).
std::string runBatchXform(uint32_t n,int op){
    std::lock_guard<std::mutex> lock(gRuntimeMutex);
    const auto tAll0=std::chrono::steady_clock::now();
    if(!g.ready || !g.api || !g.context) return "ERR NPU_NOT_READY";
    const auto& f=g.api->QNN_INTERFACE_VER_NAME;
    const uint32_t Nb=bucketize(n);
    if(Nb==0) return "ERR XFORM_TOO_LARGE (max "+std::to_string((unsigned)MM_BUCKET_MAX)+")";
    if(checkedTensorBytes(Nb,1,sizeof(int8_t))==0)
        return "ERR XFORM_BYTES_EXCEEDED n="+std::to_string((unsigned)Nb);
    const float sc=1.0f/127.0f;
    const uint64_t key=(2ULL<<60)|((uint64_t)(op&0xff)<<52)|(uint64_t)Nb;
    Runtime::MatMulGraph* mg=nullptr;
    bool graphCached=false;
    auto found=g.matMulGraphs8.find(key);
    if(found!=g.matMulGraphs8.end()){ mg=&found->second; graphCached=true; }
    Qnn_ErrorHandle_t rc=QNN_SUCCESS;
    if(!mg){
        if(!ensureGraphBudget()) return "ERR GRAPH_BUDGET_EXHAUSTED";
        auto inserted=g.matMulGraphs8.emplace(key, Runtime::MatMulGraph{});
        mg=&inserted.first->second;
        mg->m=Nb; mg->k=1; mg->n=1;
        mg->dimsA[0]=Nb; mg->dimsA[1]=1;
        mg->dimsB[0]=Nb; mg->dimsB[1]=1;
        mg->dimsC[0]=Nb; mg->dimsC[1]=1;
        const std::string gn="mcnpu_xf_"+std::to_string(op)+"_"+std::to_string(Nb);
        rc=f.graphCreate(g.context,gn.c_str(),nullptr,&mg->graph);
        if(rc!=QNN_SUCCESS||!mg->graph){ g.matMulGraphs8.erase(inserted.first); return "ERR XF_GRAPH_CREATE rc="+std::to_string((int)rc); }
        mg->a=makeTensorQ("a",QNN_TENSOR_TYPE_APP_WRITE,QNN_DATATYPE_SFIXED_POINT_8,mg->dimsA,2,sc);
        mg->b=makeTensorQ("b",QNN_TENSOR_TYPE_APP_WRITE,QNN_DATATYPE_SFIXED_POINT_8,mg->dimsB,2,sc);
        mg->c=makeTensorQ("c",QNN_TENSOR_TYPE_APP_READ,QNN_DATATYPE_SFIXED_POINT_8,mg->dimsC,2,sc);
        rc=f.tensorCreateGraphTensor(mg->graph,&mg->a);
        if(rc==QNN_SUCCESS) rc=f.tensorCreateGraphTensor(mg->graph,&mg->b);
        if(rc==QNN_SUCCESS) rc=f.tensorCreateGraphTensor(mg->graph,&mg->c);
        if(rc!=QNN_SUCCESS){ g.matMulGraphs8.erase(key); return "ERR XF_TENSOR rc="+std::to_string((int)rc); }
        Qnn_Param_t xp;
        std::memset(&xp,0,sizeof(xp));
        // Qnn_Param_t has no version field in this QNN release.
        xp.name=QNN_OP_ELEMENT_WISE_BINARY_PARAM_OPERATION;
        xp.scalarParam.dataType=QNN_DATATYPE_UINT_32;
        xp.scalarParam.uint32Value=(op==0)?QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD:QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MULTIPLY;
        Qnn_Tensor_t xin[2]={mg->a,mg->b};
        Qnn_OpConfig_t o=QNN_OPCONFIG_INIT;
        o.v1.name="xf"; o.v1.packageName="qti.aisw"; o.v1.typeName=QNN_OP_ELEMENT_WISE_BINARY;
        o.v1.numOfParams=1; o.v1.params=&xp;
        o.v1.numOfInputs=2; o.v1.inputTensors=xin;
        o.v1.numOfOutputs=1; o.v1.outputTensors=&mg->c;
        rc=f.graphAddNode(mg->graph,o);
        if(rc!=QNN_SUCCESS){ g.matMulGraphs8.erase(key); return "ERR XF_NODE rc="+std::to_string((int)rc); }
        rc=f.graphFinalize(mg->graph,nullptr,nullptr);
        if(rc!=QNN_SUCCESS){ g.matMulGraphs8.erase(key); return "ERR XF_FINALIZE rc="+std::to_string((int)rc); }
        I("XFORM GRAPH READY op=%d n=%u",op,(unsigned)Nb);
    }
    std::vector<int8_t> A(Nb,0),B(Nb,0),C(Nb,0);
    for(uint32_t i=0;i<Nb;i++) A[i]=(int8_t)(((int)((i*7)%41))-20);
    for(uint32_t i=0;i<Nb;i++) B[i]=(int8_t)(((int)((i*5)%31))-10);
    auto tc0=std::chrono::steady_clock::now();
    std::vector<int8_t> ref(n,0);
    for(uint32_t i=0;i<n;i++){
        int v=(op==0)?((int)A[i]+(int)B[i]):(int)(((int)A[i]*(int)B[i])/127);
        if(v>127)v=127; if(v<-128)v=-128;
        ref[i]=(int8_t)v;
    }
    const long long cpuUs=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-tc0).count();
    Qnn_Tensor_t ea=mg->a,eb=mg->b,ec=mg->c;
    ea.v1.clientBuf.data=A.data(); ea.v1.clientBuf.dataSize=(uint32_t)Nb;
    eb.v1.clientBuf.data=B.data(); eb.v1.clientBuf.dataSize=(uint32_t)Nb;
    ec.v1.clientBuf.data=C.data(); ec.v1.clientBuf.dataSize=(uint32_t)Nb;
    Qnn_Tensor_t ein[2]={ea,eb};
    Qnn_Tensor_t eout[1]={ec};
    auto t0=std::chrono::steady_clock::now();
    rc=f.graphExecute(mg->graph,ein,2,eout,1,nullptr,nullptr);
    const long long execUs=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-t0).count();
    if(rc!=QNN_SUCCESS){ noteTransportFaultLocked((int)rc); return "ERR XF_EXECUTE rc="+std::to_string((int)rc); }
    int bad=0; double maxAbs=0.0;
    for(uint32_t i=0;i<n;i++){
        const double d=std::fabs((double)C[i]-(double)ref[i]);
        if(d>maxAbs)maxAbs=d;
        if(d>1.5) bad++;
    }
    const double su=execUs>0?(double)cpuUs/(double)execUs:0.0;
    char buf[360];
    std::snprintf(buf,sizeof(buf),"OK XFORM op=%d n=%u npu_exec_us=%lld cpu_us=%lld speedup=%.2fx bad=%d max_abs=%.1f total_us=%lld",
        op,(unsigned)n,execUs,cpuUs,su,bad,maxAbs,
        (long long)std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-tAll0).count());
    return buf;
}

std::string runMatMulInt8(uint32_t m,uint32_t k,uint32_t n){
    std::lock_guard<std::mutex> lock(gRuntimeMutex);
    const auto total0=std::chrono::steady_clock::now();
    if(!g.ready || !g.api || !g.context) return "ERR NPU_NOT_READY";
    const auto& f=g.api->QNN_INTERFACE_VER_NAME;
    const uint64_t key=((uint64_t)m<<42)|((uint64_t)k<<21)|(uint64_t)n;
    Runtime::MatMulGraph* mg=nullptr;
    bool cached=false;
    auto found=g.matMulGraphs8.find(key);
    if(found!=g.matMulGraphs8.end()){ mg=&found->second; cached=true; }
    long long createUs=0, finalizeUs=0;
    Qnn_ErrorHandle_t rc=QNN_SUCCESS;
    if(!cached){
        if(!ensureGraphBudget()) return "ERR GRAPH_BUDGET_EXHAUSTED";
        auto inserted=g.matMulGraphs8.emplace(key, Runtime::MatMulGraph{});
        mg=&inserted.first->second;
        mg->m=m; mg->k=k; mg->n=n;
        mg->dimsA[0]=k; mg->dimsA[1]=m;
        mg->dimsB[0]=n; mg->dimsB[1]=k;
        mg->dimsC[0]=n; mg->dimsC[1]=m;
        const std::string graphName="mcnpu_mm8_"+std::to_string(++g.graphSeq);
        auto tCreate0=std::chrono::steady_clock::now();
        rc=f.graphCreate(g.context,graphName.c_str(),nullptr,&mg->graph);
        createUs=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-tCreate0).count();
        if(rc!=QNN_SUCCESS||!mg->graph){
            g.matMulGraphs8.erase(inserted.first);
            return "ERR MM8_GRAPH_CREATE rc="+std::to_string((int)rc)+" "+verbose(rc);
        }
        mg->a=makeTensorQ("a",QNN_TENSOR_TYPE_APP_WRITE,QNN_DATATYPE_SFIXED_POINT_8,mg->dimsA,2,Q_SCALE_A);
        mg->b=makeTensorQ("b",QNN_TENSOR_TYPE_APP_WRITE,QNN_DATATYPE_SFIXED_POINT_8,mg->dimsB,2,Q_SCALE_B);
        mg->c=makeTensorQ("c",QNN_TENSOR_TYPE_APP_READ,QNN_DATATYPE_SFIXED_POINT_8,mg->dimsC,2,Q_SCALE_C);
        rc=f.tensorCreateGraphTensor(mg->graph,&mg->a);
        if(rc==QNN_SUCCESS) rc=f.tensorCreateGraphTensor(mg->graph,&mg->b);
        if(rc==QNN_SUCCESS) rc=f.tensorCreateGraphTensor(mg->graph,&mg->c);
        if(rc!=QNN_SUCCESS){
            g.matMulGraphs8.erase(key);
            return "ERR MM8_TENSOR_CREATE rc="+std::to_string((int)rc)+" "+verbose(rc);
        }
        Qnn_Tensor_t mmIns[2]={mg->a,mg->b};
        Qnn_OpConfig_t op=QNN_OPCONFIG_INIT;
        op.v1.name="matmul";
        op.v1.packageName="qti.aisw";
        op.v1.typeName=QNN_OP_MAT_MUL;
        op.v1.numOfParams=0;
        op.v1.params=nullptr;
        op.v1.numOfInputs=2;
        op.v1.inputTensors=mmIns;
        op.v1.numOfOutputs=1;
        op.v1.outputTensors=&mg->c;
        rc=f.graphAddNode(mg->graph,op);
        if(rc!=QNN_SUCCESS){
            g.matMulGraphs8.erase(key);
            return "ERR MM8_GRAPH_NODE rc="+std::to_string((int)rc)+" "+verbose(rc);
        }
        auto tF=std::chrono::steady_clock::now();
        rc=f.graphFinalize(mg->graph,nullptr,nullptr);
        finalizeUs=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-tF).count();
        if(rc!=QNN_SUCCESS){
            g.matMulGraphs8.erase(key);
            return "ERR MM8_GRAPH_FINALIZE rc="+std::to_string((int)rc)+" "+verbose(rc);
        }
        I("MATMUL8 GRAPH READY m=%u k=%u n=%u",(unsigned)m,(unsigned)k,(unsigned)n);
    }

    std::vector<float> A((size_t)m*k), B((size_t)k*n);
    for(size_t i=0;i<A.size();i++) A[i]=0.01f*(float)((i*7)%23)-0.1f;
    for(size_t i=0;i<B.size();i++) B[i]=0.05f*(float)((i*5)%17)-0.2f;

    std::vector<float> Ccpu((size_t)m*n,0.f);
    auto tc0=std::chrono::steady_clock::now();
    for(uint32_t i=0;i<m;i++){
        for(uint32_t p=0;p<k;p++){
            const float av=A[(size_t)i*k+p];
            for(uint32_t j=0;j<n;j++) Ccpu[(size_t)i*n+j]+=av*B[(size_t)p*n+j];
        }
    }
    const long long cpuUs=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-tc0).count();

    std::vector<int8_t> Aq(A.size()), Bq(B.size()), Cq((size_t)m*n,0);
    for(size_t i=0;i<A.size();i++) Aq[i]=quantize8(A[i],Q_SCALE_A);
    for(size_t i=0;i<B.size();i++) Bq[i]=quantize8(B[i],Q_SCALE_B);

    Qnn_Tensor_t ea=mg->a, eb=mg->b, ec=mg->c;
    ea.v1.clientBuf.data=Aq.data(); ea.v1.clientBuf.dataSize=(uint32_t)Aq.size();
    eb.v1.clientBuf.data=Bq.data(); eb.v1.clientBuf.dataSize=(uint32_t)Bq.size();
    ec.v1.clientBuf.data=Cq.data(); ec.v1.clientBuf.dataSize=(uint32_t)Cq.size();
    Qnn_Tensor_t execIn[2]={ea,eb};
    Qnn_Tensor_t execOut[1]={ec};
    auto t0=std::chrono::steady_clock::now();
    rc=f.graphExecute(mg->graph,execIn,2,execOut,1,nullptr,nullptr);
    const long long execUs=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-t0).count();
    if(rc!=QNN_SUCCESS){ noteTransportFaultLocked((int)rc); return "ERR MM8_EXECUTE rc="+std::to_string((int)rc)+" "+verbose(rc); }

    double maxAbs=0.0;
    int bad=0;
    for(size_t i=0;i<Cq.size();i++){
        const double got=(double)Cq[i]*(double)Q_SCALE_C;
        const double diff=std::fabs(got-(double)Ccpu[i]);
        const double tol=0.15+0.10*std::fabs((double)Ccpu[i]);
        if(diff>maxAbs) maxAbs=diff;
        if(diff>tol) bad++;
    }
    const double speedup = execUs>0 ? (double)cpuUs/(double)execUs : 0.0;
    const long long totalUs=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-total0).count();
    char buf[768];
    std::snprintf(buf,sizeof(buf),
        "OK MATMUL8 m=%u k=%u n=%u dtype=int8 quantA=%.4f quantB=%.4f quantC=%.4f cached=%s create_us=%lld finalize_us=%lld npu_exec_us=%lld cpu_us=%lld speedup=%.2fx bad=%d max_abs=%.5g total_us=%lld",
        (unsigned)m,(unsigned)k,(unsigned)n,Q_SCALE_A,Q_SCALE_B,Q_SCALE_C,cached?"true":"false",
        createUs,finalizeUs,execUs,cpuUs,speedup,bad,maxAbs,totalUs);
    return buf;
}

// Real data path: A and B arrive as int8 buffers (normalized to [-1,1], scale
// 1/127 on both sides). The output scale must grow with k because an int8
// accumulator saturates: sum(k terms of |a*b| <= 1) can reach k/127^2.
// Shape bucketing: any (m,k,n) is normalized to the nearest bucket andzero-padded,
// so a bounded set of graphs is ever created. Without this, Minecraft's ever-
// changing batch sizes would silently poison the context.
// Keep this list in exact sync with NpuDispatcher.MM_BUCKETS on the Java side.
// A mismatch is silent and expensive: Java rounds to a bucket the native side
// does not have, native rounds again, and the caller pays for two paddings.
static const uint32_t MM_BUCKETS[] = {32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536};
// MM_BUCKET_MAX is declared near the top of this file (see the forward declaration
// above) so the helpers defined earlier can use it. Keep it in sync with the last
// entry of MM_BUCKETS.

static uint32_t bucketize(uint32_t v){
    for(size_t i=0;i<sizeof(MM_BUCKETS)/sizeof(MM_BUCKETS[0]);i++) if(v<=MM_BUCKETS[i]) return MM_BUCKETS[i];
    return 0;
}

// Hard ceiling on a single tensor's byte size. The per-dimension limit alone is
// NOT enough: 65536 x 65536 int8 is 4 GiB, which overflows the uint32 field in
// Qnn_ClientBuffer_t.dataSize long before it fails on real memory. Every planned
// shape must be checked against BOTH the dimension cap and this byte cap.
static const uint64_t MM_MAX_TENSOR_BYTES = 64ULL * 1024 * 1024;  // 64 MiB

// Returns the byte size, or 0 when the shape is unacceptable. Callers must treat
// 0 as "reject". Checks, in order:
//   - dimension within the bucket ladder
//   - size_t multiplication overflow
//   - fits in the uint32 dataSize field QNN expects
//   - stays inside the per-tensor budget
static uint64_t checkedTensorBytes(uint32_t rows, uint32_t cols, uint32_t elemSize){
    if(rows == 0 || cols == 0 || elemSize == 0) return 0;
    if(rows > MM_BUCKET_MAX || cols > MM_BUCKET_MAX) return 0;

    const uint64_t cells = (uint64_t)rows * (uint64_t)cols;
    // Overflow check: if the division does not give back the inputs, it wrapped.
    if(cells / (uint64_t)rows != (uint64_t)cols) return 0;

    const uint64_t bytes = cells * (uint64_t)elemSize;
    if(bytes / (uint64_t)elemSize != cells) return 0;

    // Qnn_ClientBuffer_t.dataSize is a uint32_t.
    if(bytes > 0xFFFFFFFFULL) return 0;
    if(bytes > MM_MAX_TENSOR_BYTES) return 0;
    return bytes;
}

// Rejects a planned (m,k,n) matmul when ANY of the three tensors exceeds the
// byte budget. Called before graphCreate so a bad shape costs nothing.
static bool mmShapeSafe(uint32_t m, uint32_t k, uint32_t n, uint32_t elemSize){
    return checkedTensorBytes(m, k, elemSize) != 0
        && checkedTensorBytes(k, n, elemSize) != 0
        && checkedTensorBytes(m, n, elemSize) != 0;
}

std::string runMatMulInt8Buf(const int8_t* Ain,const int8_t* Bin,int8_t* Cout,uint32_t m,uint32_t k,uint32_t n,float& scaleCOut){
    const auto lockWait0 = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> lock(gRuntimeMutex);
    const long long lockWaitUs = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - lockWait0).count();
    I("MM8BUF QUEUE lock_wait_us=%lld m=%u k=%u n=%u",
      lockWaitUs,(unsigned)m,(unsigned)k,(unsigned)n);
    if(!g.ready || !g.api || !g.context){
        if(!tryRecoverContextLocked()){
            g_lastNativeError="ERR NPU_NOT_READY (context lost and rebuild failed: "+g.err+")";
            return g_lastNativeError;
        }
    }
    const auto& f=g.api->QNN_INTERFACE_VER_NAME;
    const uint32_t Mb=bucketize(m), Kb=bucketize(k), Nb=bucketize(n);
    if(Mb==0||Kb==0||Nb==0) { g_lastNativeError="ERR BUF_TOO_LARGE (max "+std::to_string((unsigned)MM_BUCKET_MAX)+")"; return g_lastNativeError; }
    // Dimension caps passed, but the resulting tensors can still be gigabytes.
    if(!mmShapeSafe(Mb, Kb, Nb, sizeof(int8_t))) {
        g_lastNativeError="ERR BUF_BYTES_EXCEEDED m="+std::to_string((unsigned)Mb)
              +" k="+std::to_string((unsigned)Kb)
              +" n="+std::to_string((unsigned)Nb)
              +" (per-tensor cap "+std::to_string((unsigned long long)(MM_MAX_TENSOR_BYTES>>20))+" MiB)";
        return g_lastNativeError;
    }
    // Caller contract (matches the probes that already work): A and B are
    // quantised against +-0.127, i.e. one int8 step is 0.001. HTP only executes
    // a FULLY quantised matmul -- int8 in, int8 out with all three scales loaded
    // -- and returns rc=1100 (unsupported) for any float/int32 output.
    const float scaleA = 0.001f;
    const float scaleB = 0.001f;
    // Calibrated value. The raw accumulator dump proved HTP DOES requantise
    // (values like 3/-27/74/-1, not just +-127), but with a factor roughly k
    // larger than scaleA*scaleB/scaleC predicts. Rather than keep guessing the
    // exact formula, use the scaleC that the passing int8 probes use (0.02) and
    // see how the numbers land.
    const float scaleC = 0.02f;
    scaleCOut = scaleC;   // overwritten from the cache below once mg is known

    std::vector<int8_t> Ap, Bp;
    const int8_t* Ause=Ain;
    const int8_t* Buse=Bin;
    if(Mb!=m || Kb!=k || Nb!=n){
        Ap.assign((size_t)Mb*Kb, 0);
        for(uint32_t r=0;r<m;r++) std::memcpy(&Ap[(size_t)r*Kb], &Ain[(size_t)r*k], k);
        Bp.assign((size_t)Kb*Nb, 0);
        for(uint32_t r=0;r<k;r++) std::memcpy(&Bp[(size_t)r*Nb], &Bin[(size_t)r*n], n);
        Ause=Ap.data();
        Buse=Bp.data();
    }

    const uint64_t key=((uint64_t)Mb<<42)|((uint64_t)Kb<<21)|(uint64_t)Nb;
    Runtime::MatMulGraph* mg=nullptr;
    // Tracks whether this call reused a graph, so the EXEC line can say so.
    // It has to be declared here: the graphCached in runBatchXform is a different
    // function's local, and referring to it from runMatMulInt8Buf would not compile.
    bool graphCached=false;
    auto found=g.matMulGraphs8.find(key);
    if(found!=g.matMulGraphs8.end()){ mg=&found->second; graphCached=true; }
    Qnn_ErrorHandle_t rc=QNN_SUCCESS;
    if(!mg){
        if(!ensureGraphBudget()) return "ERR GRAPH_BUDGET_EXHAUSTED";
        // Do NOT increment here. The old code counted a graph before it was built
        // AND again after finalize, so every matmul8 graph cost 2 against
        // MAX_CACHED_GRAPHS and only four shapes fit before the whole context was
        // torn down. A failed create is erased below and must not be counted at all.
        auto inserted=g.matMulGraphs8.emplace(key, Runtime::MatMulGraph{});
        mg=&inserted.first->second;
        mg->m=Mb; mg->k=Kb; mg->n=Nb;
        mg->dimsA[0]=Kb; mg->dimsA[1]=Mb;
        mg->dimsB[0]=Nb; mg->dimsB[1]=Kb;
        mg->dimsC[0]=Nb; mg->dimsC[1]=Mb;
        const std::string graphName="mcnpu_mmb_"+std::to_string(Mb)+"x"+std::to_string(Kb)+"x"+std::to_string(Nb);
        rc=f.graphCreate(g.context,graphName.c_str(),nullptr,&mg->graph);
        if(rc!=QNN_SUCCESS||!mg->graph){
            g.matMulGraphs8.erase(inserted.first);
            return "ERR BUF_GRAPH_CREATE rc="+std::to_string((int)rc);
        }
        mg->a=makeTensorQ("a",QNN_TENSOR_TYPE_APP_WRITE,QNN_DATATYPE_SFIXED_POINT_8,mg->dimsA,2,scaleA);
        mg->b=makeTensorQ("b",QNN_TENSOR_TYPE_APP_WRITE,QNN_DATATYPE_SFIXED_POINT_8,mg->dimsB,2,scaleB);
        mg->c=makeTensorQ("c",QNN_TENSOR_TYPE_APP_READ,QNN_DATATYPE_SFIXED_POINT_8,mg->dimsC,2,scaleC);
        rc=f.tensorCreateGraphTensor(mg->graph,&mg->a);
        if(rc==QNN_SUCCESS) rc=f.tensorCreateGraphTensor(mg->graph,&mg->b);
        if(rc==QNN_SUCCESS) rc=f.tensorCreateGraphTensor(mg->graph,&mg->c);
        if(rc!=QNN_SUCCESS){
            g.matMulGraphs8.erase(key);
            return "ERR BUF_TENSOR_CREATE rc="+std::to_string((int)rc);
        }
        Qnn_Tensor_t ins[2]={mg->a,mg->b};
        Qnn_OpConfig_t op=QNN_OPCONFIG_INIT;
        op.v1.name="matmul";
        op.v1.packageName="qti.aisw";
        op.v1.typeName=QNN_OP_MAT_MUL;
        op.v1.numOfParams=0;
        op.v1.params=nullptr;
        op.v1.numOfInputs=2;
        op.v1.inputTensors=ins;
        op.v1.numOfOutputs=1;
        op.v1.outputTensors=&mg->c;
        rc=f.graphAddNode(mg->graph,op);
        if(rc!=QNN_SUCCESS){
            g.matMulGraphs8.erase(key);
            return "ERR BUF_GRAPH_NODE rc="+std::to_string((int)rc);
        }
        rc=f.graphFinalize(mg->graph,nullptr,nullptr);
        if(rc!=QNN_SUCCESS){
            g.matMulGraphs8.erase(key);
            return "ERR BUF_GRAPH_FINALIZE rc="+std::to_string((int)rc);
        }
        g.graphCount++;
        // Count only a successfully finalized graph. The pre-create increment
        // above reserves no durable resource and caused every cached graph to count twice,
        // forcing premature context resets after only four real graphs.
        I("MATMUL8BUF GRAPH READY bucket=%ux%ux%u scaleC=%.5f",(unsigned)Mb,(unsigned)Kb,(unsigned)Nb,scaleC);
    }

    std::vector<int8_t> Cpad((size_t)Mb*Nb, 0);
    Qnn_Tensor_t ea=mg->a, eb=mg->b, ec=mg->c;
    ea.v1.clientBuf.data=(void*)Ause; ea.v1.clientBuf.dataSize=(uint32_t)((size_t)Mb*Kb);
    eb.v1.clientBuf.data=(void*)Buse; eb.v1.clientBuf.dataSize=(uint32_t)((size_t)Kb*Nb);
    ec.v1.clientBuf.data=Cpad.data(); ec.v1.clientBuf.dataSize=(uint32_t)((size_t)Mb*Nb);
    Qnn_Tensor_t execIn[2]={ea,eb};
    Qnn_Tensor_t execOut[1]={ec};
    const auto exec0 = std::chrono::steady_clock::now();
    rc=f.graphExecute(mg->graph,execIn,2,execOut,1,nullptr,nullptr);
    const long long execUs = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - exec0).count();
    I("MM8BUF EXEC queue_lock_wait_us=%lld qnn_execute_us=%lld graph_cached=%s bucket=%ux%ux%u",
      lockWaitUs,execUs,graphCached?"true":"false",(unsigned)Mb,(unsigned)Kb,(unsigned)Nb);
    if(rc!=QNN_SUCCESS){
        // Enough context to tell "our tensors are wrong" from "the DSP link dropped".
        //
        // rc=1007 is QNN_COMMON_ERROR_SYSTEM_COMMUNICATION (QNN_MIN_ERROR_COMMON(1000)+7):
        // communication with the platform/OS service failed, service recoverable. It is a
        // COMMON error, not a GRAPH error - QNN never rejected the graph. So a 1007 here
        // cannot be a shape, dtype or buffer-size mistake; it means the FastRPC link to the
        // DSP is down, which matches rc=14001 (device not reachable) from the in-process
        // probe. Reporting it as a bare code sent us looking at tensor shapes for three
        // rounds instead of at the transport.
        std::string ctx="ERR BUF_EXECUTE rc="+std::to_string((int)rc)
            +" bucket="+std::to_string((unsigned)Mb)+"x"+std::to_string((unsigned)Kb)
            +"x"+std::to_string((unsigned)Nb)
            +" aBytes="+std::to_string((uint32_t)((size_t)Mb*Kb))
            +" bBytes="+std::to_string((uint32_t)((size_t)Kb*Nb))
            +" cBytes="+std::to_string((uint32_t)((size_t)Mb*Nb))
            +" alignA="+std::to_string((unsigned)((uintptr_t)Ause%64))
            +" alignB="+std::to_string((unsigned)((uintptr_t)Buse%64))
            +" alignC="+std::to_string((unsigned)((uintptr_t)Cpad.data()%64))
            +" cached="+(graphCached?std::string("true"):std::string("false"))
            +" qnn_execute_us="+std::to_string((long long)execUs)
            +" graphs="+std::to_string(g.graphCount)+"/"+std::to_string(MAX_CACHED_GRAPHS);
        if((int)rc==1007) ctx += " (SYSTEM_COMMUNICATION: DSP link lost, not a graph/tensor problem)";
        E("MM8BUF EXEC FAIL %s",ctx.c_str());
        noteTransportFaultLocked((int)rc);
        return ctx;
    }
    // Dynamic quantisation: normalise the true magnitudes first, then map to int8.
    // Because C_int32 is in units of 1/127^2 of the real product, the scale we
    // return must include that factor: C_real = Cq8 * (max|C|/127) / 127^2.
    // Self-calibration. HTP requantises the accumulator with a factor that does
    // not match scaleA*scaleB/scaleC (measured ~11x off on k=32 buckets -- the
    // raw dump showed values far too large for the requested scale). Instead of
    // reverse-engineering that formula, compute the integer dot product of the
    // first few outputs on the host (O(8k), negligible) and solve for the scale
    // that makes c_int*scale == sum(qA*qB)*scaleA*scaleB exactly.
    if(mg->calibrated){
        // Hot path: shape already measured, reuse the cached dequant scale.
        scaleCOut = mg->scaleEff;
    } else {
        // Per-element ratio c_int / sum(qA*qB), then take the median. Summing
        // first (the previous attempt) cancels out on structured data and left
        // the scale untouched; a median over many elements is immune to both
        // cancellation and to the rounding noise of individual elements.
        //
        // Only a handful of rows is needed: the ratio is a property of the
        // requantisation, not of the data. The previous 64-row sweep cost
        // 64*n*k host MAC per call (17M at 512^3 -> ~90ms) and dominated the
        // whole pipeline. 4 rows keeps ~2k samples, which the median handles
        // fine, and is only ever paid once per shape anyway.
        const uint32_t refRows = m<4u ? m : 4u;
        std::vector<float> ratios;
        ratios.reserve((size_t)refRows*n);
        for(uint32_t r=0;r<refRows;r++){
            for(uint32_t j=0;j<n;j++){
                long long s=0;
                for(uint32_t p=0;p<k;p++) s += (long long)Ain[(size_t)r*k+p]*(long long)Bin[(size_t)p*n+j];
                const int ci=(int)Cpad[(size_t)r*Nb+j];
                if(s!=0 && ci!=0) ratios.push_back((float)((double)ci/(double)s));
            }
        }
        if(ratios.size()>=8){
            std::sort(ratios.begin(),ratios.end());
            const double med=(double)ratios[ratios.size()/2];
            if(med!=0.0){
                const double scaleEff=(double)scaleA*(double)scaleB/med;
                if(scaleEff>0.0){
                    scaleCOut=(float)scaleEff;
                    mg->scaleEff=scaleCOut;
                    mg->calibrated=true;      // cache it for every later call
                }
            }
            I("MATMUL8BUF CAL bucket=%ux%ux%u raw=%.8g eff=%.8g med=%.8g n=%u cached=1",
              (unsigned)Mb,(unsigned)Kb,(unsigned)Nb,scaleC,scaleCOut,med,(unsigned)ratios.size());
        } else {
            I("MATMUL8BUF CAL-SKIP bucket=%ux%ux%u samples=%u",(unsigned)Mb,(unsigned)Kb,(unsigned)Nb,(unsigned)ratios.size());
        }
    }
    for(uint32_t r=0;r<m;r++) std::memcpy(&Cout[(size_t)r*n], &Cpad[(size_t)r*Nb], n);
    noteTransportSuccessLocked();
    I("MATMUL8BUF OK bucket=%ux%ux%u scaleC=%.8g",(unsigned)Mb,(unsigned)Kb,(unsigned)Nb,scaleCOut);
    return "OK";
}

void shutdownRuntime(){
    std::lock_guard<std::mutex> lock(gRuntimeMutex);
    if(!g.api){
        g.addGraphs.clear();
        return;
    }
    const auto& f=g.api->QNN_INTERFACE_VER_NAME;
    // QNN 2.27 has no graphFree entry point. Graphs are owned by the context
    // and are released when contextFree() is called. Drop our cached handles
    // before releasing that context so a restart can never reuse stale data.
    g.addGraphs.clear();
    g.matMulGraphs.clear();
    g.matMulGraphs8.clear();
    g.graphCount=0;
    if(f.contextFree&&g.context)f.contextFree(g.context,nullptr);
    if(f.deviceFree&&g.device)f.deviceFree(g.device);
    if(f.backendFree&&g.backend)f.backendFree(g.backend);
    if(f.logFree&&g.logger)f.logFree(g.logger);
    g.context=nullptr;g.device=nullptr;g.backend=nullptr;g.logger=nullptr;g.ready=false;g.api=nullptr;
    if(g.qnn)dlclose(g.qnn);
    g.qnn=nullptr;
}

}

extern "C" JNIEXPORT jboolean JNICALL Java_bslsjdk_mcnpu_NpuRuntime_nativeInit(JNIEnv* e,jclass,jstring jq,jstring jw){
    if(!jq || !jw) return JNI_FALSE;
    const char* p=e->GetStringUTFChars(jq,nullptr);
    const char* w=e->GetStringUTFChars(jw,nullptr);
    std::string qnnDir=p?p:"";
    std::string workDir=w?w:"";
    if(p) e->ReleaseStringUTFChars(jq,p);
    if(w) e->ReleaseStringUTFChars(jw,w);
    return initRuntime(qnnDir,workDir)?JNI_TRUE:JNI_FALSE;
}
extern "C" JNIEXPORT jstring JNICALL Java_bslsjdk_mcnpu_NpuRuntime_nativeGetDeviceInfo(JNIEnv* e,jclass){
    return e->NewStringUTF((g.ready?g.info:deepReport()).c_str());
}
extern "C" JNIEXPORT jboolean JNICALL Java_bslsjdk_mcnpu_NpuRuntime_nativeTest(JNIEnv*,jclass){
    float a[16],b[16];
    for(int i=0;i<16;i++){a[i]=(float)i;b[i]=2.f;}
    return runAdd(a,b,16).rfind("OK ",0)==0?JNI_TRUE:JNI_FALSE;
}
// Same self test, but returns the full reply instead of collapsing it to a bit.
// The boolean version threw away the reason: every failure became the same
// "ERR HTP_GRAPH_EXECUTE" at the client, which is why a service that connects
// fine could still fail its smoke test with nothing to go on.
extern "C" JNIEXPORT jstring JNICALL Java_bslsjdk_mcnpu_NpuRuntime_nativeTestDetail(JNIEnv* e,jclass){
    float a[16],b[16];
    for(int i=0;i<16;i++){a[i]=(float)i;b[i]=2.f;}
    std::string r=runAdd(a,b,16);
    I("SMOKE DETAIL %s",r.c_str());
    return e->NewStringUTF(r.c_str());
}
extern "C" JNIEXPORT jstring JNICALL Java_bslsjdk_mcnpu_NpuRuntime_nativeAdd(JNIEnv* e,jclass,jfloatArray ja,jfloatArray jb){
    if(!ja||!jb)return e->NewStringUTF("ERR NULL");
    jsize n=e->GetArrayLength(ja);
    if(n<=0||n!=e->GetArrayLength(jb))return e->NewStringUTF("ERR SIZE");
    const uint32_t padded=addPadSize((uint32_t)n);
    if(padded==0) return e->NewStringUTF(("ERR SIZE_UNSUPPORTED max="+std::to_string(g_addLadderMax)).c_str());
    std::vector<float>a(padded,0.f),b(padded,0.f);
    e->GetFloatArrayRegion(ja,0,n,a.data());
    e->GetFloatArrayRegion(jb,0,n,b.data());
    std::string r=runAdd(a.data(),b.data(),padded);
    // Report the padding rather than hiding it: otherwise a caller benchmarking
    // 8 values would read n=16 and believe 16 were required.
    if(padded!=(uint32_t)n && r.rfind("OK",0)==0){
        r += " req_n=" + std::to_string((unsigned)n);
        r += " padded_to=" + std::to_string(padded);
    }
    return e->NewStringUTF(r.c_str());
}
extern "C" JNIEXPORT jstring JNICALL Java_bslsjdk_mcnpu_NpuRuntime_nativeAddInto(JNIEnv* e,jclass,jfloatArray ja,jfloatArray jb,jfloatArray jo,jboolean verify){
    if(!ja||!jb||!jo)return e->NewStringUTF("ERR NULL");
    jsize n=e->GetArrayLength(ja);
    if(n<=0||n!=e->GetArrayLength(jb))return e->NewStringUTF("ERR SIZE");
    if(n>e->GetArrayLength(jo))return e->NewStringUTF("ERR OUT_TOO_SMALL");
    const uint32_t padded=addPadSize((uint32_t)n);
    if(padded==0) return e->NewStringUTF(("ERR SIZE_UNSUPPORTED max="+std::to_string(g_addLadderMax)).c_str());
    std::vector<float>a(padded,0.f),b(padded,0.f),out(padded,-999.f);
    e->GetFloatArrayRegion(ja,0,n,a.data());
    e->GetFloatArrayRegion(jb,0,n,b.data());
    std::string r=runAddEx(a.data(),b.data(),padded,out.data(),verify==JNI_TRUE);
    if(r.rfind("OK",0)==0){
        e->SetFloatArrayRegion(jo,0,n,out.data());
        if(padded!=(uint32_t)n){
            r += " req_n=" + std::to_string((unsigned)n);
            r += " padded_to=" + std::to_string(padded);
        }
    }
    return e->NewStringUTF(r.c_str());
}
extern "C" JNIEXPORT jstring JNICALL Java_bslsjdk_mcnpu_NpuRuntime_nativeAddProbe(JNIEnv* e,jclass){
    if(!g.ready || !g.api || !g.context) return e->NewStringUTF("ERR NPU_NOT_READY");
    return e->NewStringUTF(probeAddLadder().c_str());
}

// Set by the service as soon as a real request arrives. The probe yields the
// device between candidates, so an in-flight probe stops at the next one instead
// of making the request wait for a diagnostic.
extern "C" JNIEXPORT void JNICALL Java_bslsjdk_mcnpu_NpuRuntime_nativeAbortProbe(JNIEnv*,jclass){
    g_probeAbort.store(true);
}
// Drop every cached graph and rebuild the context.
//
// MAX_CACHED_GRAPHS is small on purpose, and a benchmark builds one graph per
// candidate shape. Left alone, a sweep can occupy the entire budget before the
// game submits any real work, and then the first production shape is the one
// that pays a full context teardown - every cached graph, including the warmup
// graph the hot path depends on, is destroyed underneath it. Flushing when the
// diagnostic ends hands production a clean cache instead of a landmine.
extern "C" JNIEXPORT jstring JNICALL Java_bslsjdk_mcnpu_NpuRuntime_nativeFlushGraphs(JNIEnv* e,jclass){
    std::lock_guard<std::mutex> lock(gRuntimeMutex);
    if(!g.api) return e->NewStringUTF("ERR not initialised");
    const int before=g.graphCount;
    const bool ok=resetContextLocked();
    std::string r=std::string(ok?"OK":"ERR")+" FLUSH_GRAPHS dropped="+std::to_string(before)
        +" now="+std::to_string(g.graphCount)+"/"+std::to_string(MAX_CACHED_GRAPHS);
    if(!ok) r+=" err="+g.err;
    I("%s",r.c_str());
    return e->NewStringUTF(r.c_str());
}

// ---------------------------------------------------------------------------
// Op availability probe (noise kernel feasibility)
//
// Perlin is a table lookup per corner: perm[] plus a 16x3 gradient matrix. That
// means ~14 gathers per noise evaluation, 73,728 evaluations per chunk section.
// Whether the HTP has a gather decides if noise offload is worth writing at all:
//
//   with a gather   -> roughly 14 lookups per eval, feasible
//   without one     -> the only substitute is one-hot x matmul, which needs
//                      about 528 MB of intermediate per section. Not viable.
//
// So ask the backend instead of guessing. QnnBackend_getSupportedOperations()
// enumerates every op the backend registered, including the built-in package,
// which is authoritative in a way that "we tried to build a graph and it
// failed" never is - an addNode failure cannot distinguish "op missing" from
// "we passed wrong params".
static const char* kOpProbeWanted[] = {
    // table lookup - the whole question
    "Gather", "GatherElements", "GatherNd", "ScatterNd", "ScatterElements", "OneHot",
    // index arithmetic for the perm chain
    "Cast", "Reshape", "Transpose", "Concat", "Split", "StridedSlice", "Tile",
    "Squeeze", "Unsqueeze", "Shape", "Range", "Pad",
    // elementwise: floor, fade curve, lerp, compare
    "ElementWiseUnary", "ElementWiseBinary", "ElementWiseSelect", "Where",
    "Floor", "Ceil", "Round", "Abs", "Neg", "Sub", "Add", "Mul", "Div", "Min", "Max",
    "Pow", "Exp", "Log", "Sqrt", "Rsqrt", "Sin", "Cos", "Clip", "Relu",
    "Equal", "NotEqual", "Less", "LessEqual", "Greater", "GreaterEqual",
    "And", "Or", "Not",
    // reduction / layout for the corner reduction
    "MatMul", "MatMulTranspose", "ReduceMax", "ReduceSum", "ReduceMin", "ReduceMean",
    "TopK", "Quantize", "Dequantize"
};

std::string runOpProbe(){
    if(!g.ready || !g.api || !g.backend)
        return "ERR OPPROBE NOT_READY ready=" + std::to_string((int)g.ready)
             + " api=" + std::to_string((int)(g.api!=nullptr))
             + " backend=" + std::to_string((int)(g.backend!=nullptr));
    const auto& f=g.api->QNN_INTERFACE_VER_NAME;
    if(!f.backendGetSupportedOperations)
        return "ERR OPPROBE NO_API backendGetSupportedOperations=null (SDK too old?)";

    uint32_t num=0;
    const QnnBackend_OperationName_t* ops=nullptr;
    Qnn_ErrorHandle_t rc=f.backendGetSupportedOperations(g.backend,&num,&ops);
    if(rc!=QNN_SUCCESS||!ops||num==0)
        return "ERR OPPROBE QUERY rc="+std::to_string((int)rc)
             +" num="+std::to_string((unsigned)num)+" "+verbose(rc);

    std::vector<std::string> have;
    have.reserve(num);
    for(uint32_t i=0;i<num;i++){
        std::string nm = ops[i].name ? ops[i].name : "";
        if(nm.empty()) continue;
        std::string pk = ops[i].packageName ? ops[i].packageName : "";
        have.push_back(pk.empty() ? nm : (nm+"@"+pk));
    }
    std::sort(have.begin(),have.end());
    have.erase(std::unique(have.begin(),have.end()),have.end());

    // Match on the exact op name, allowing an "@package" suffix. Prefix-only
    // matching would make "Gather" also match "GatherNd", which is a different
    // op with different semantics and would give a false green light.
    std::string present, missing;
    const size_t W=sizeof(kOpProbeWanted)/sizeof(kOpProbeWanted[0]);
    for(size_t i=0;i<W;i++){
        const std::string w(kOpProbeWanted[i]);
        bool hit=false;
        for(size_t j=0;j<have.size();j++){
            const std::string& h=have[j];
            if(h.size()>=w.size() && h.compare(0,w.size(),w)==0
               && (h.size()==w.size() || h[w.size()]=='@')){ hit=true; break; }
        }
        if(hit) present += (present.empty()?"":",")+w;
        else    missing += (missing.empty()?"":",")+w;
    }

    // Both the summary and the full list go out. The summary is for reading at a
    // glance; the full list is the ground truth, so a name we never asked about
    // can never be mistaken for one the device lacks.
    std::string all;
    for(size_t i=0;i<have.size();i++) all += (i?",":"")+have[i];
    if(all.size()>6000) all = all.substr(0,6000)+"...(truncated)";

    return "OK OPPROBE n="+std::to_string((unsigned)num)
         +" present="+present
         +" missing="+missing
         +"\nOPPROBE_ALL "+all;
}

extern "C" JNIEXPORT jstring JNICALL Java_bslsjdk_mcnpu_NpuRuntime_nativeOpProbe(JNIEnv* e,jclass){
    std::lock_guard<std::mutex> lock(gRuntimeMutex);
    std::string r=runOpProbe();
    I("OPPROBE %s",r.c_str());
    return e->NewStringUTF(r.c_str());
}


// ===========================================================================
// Noise kernel
//
// Why noise and nothing else: of everything in worldgen, noise is the only part
// that is both large and free of branches. A chunk section asks for 73,728
// evaluations. On the CPU each one costs ~80ns, and most of that is eight corner
// lookups plus a 16-way branch inside the gradient function - a branch that
// mispredicts almost every time because h is effectively random. The HTP has no
// branch predictor to miss, so that cost simply disappears there.
//
// The design hangs on one op: Gather. perm[] and the 16x3 gradient matrix are
// both tables. With a gather, an evaluation is ~14 lookups plus fused
// multiply-add. Without one the only substitute is one-hot x matmul, which needs
// roughly 528MB of intermediate per section - dead on arrival.
//
// Two paths, selected at run time from what the backend actually registered:
//
//   A  full graph   coordinates in, noise out.
//                   Needs Gather, ElementWiseUnary(Floor), ElementWiseBinary.
//   C  hybrid       the host resolves the perm chain and the gradient table -
//                   both pure lookups that the CPU does well - and the graph
//                   does only the arithmetic: dot products, the fade polynomial
//                   and the trilinear blend. Needs ElementWiseBinary alone,
//                   which is far more basic than Gather.
//
// Path C recovers less, because the host still pays for the lookups, and it is
// not obviously a win once the ~2ms round trip is paid. It exists so that a
// backend without Gather still has a measured answer instead of a guess.
// ===========================================================================

static std::vector<std::string> g_opsCache;
static bool g_opsCached = false;

static void cacheOps(){
    if(g_opsCached || !g.ready || !g.api || !g.backend) return;
    const auto& f = g.api->QNN_INTERFACE_VER_NAME;
    if(!f.backendGetSupportedOperations) return;
    uint32_t num = 0;
    const QnnBackend_OperationName_t* ops = nullptr;
    if(f.backendGetSupportedOperations(g.backend, &num, &ops) != QNN_SUCCESS || !ops) return;
    g_opsCache.clear();
    for(uint32_t i = 0; i < num; i++){
        const char* nm = ops[i].name;
        if(nm && *nm) g_opsCache.emplace_back(nm);
    }
    std::sort(g_opsCache.begin(), g_opsCache.end());
    g_opsCached = true;
}

// Exact-name match allowing an "@package" suffix. Prefix-only matching would let
// "Gather" match "GatherNd", which is a different op with different semantics
// and would hand out a false green light.
static bool hasOp(const char* want){
    cacheOps();
    const std::string w(want);
    for(size_t j = 0; j < g_opsCache.size(); j++){
        const std::string& h = g_opsCache[j];
        if(h.size() >= w.size() && h.compare(0, w.size(), w) == 0
           && (h.size() == w.size() || h[w.size()] == '@')) return true;
    }
    return false;
}

// QNN keeps the name pointer and the dimensions pointer of every tensor, so both
// have to outlive the graph. std::deque never moves an existing element on
// push_back, which is exactly the guarantee needed here - a std::vector would
// hand out dangling pointers the moment it reallocated.
struct TensorArena {
    std::deque<std::string> names;
    std::deque<uint32_t>    dims;
    const char* name(const char* prefix){
        names.push_back(std::string(prefix) + "$" + std::to_string(names.size()));
        return names.back().c_str();
    }
    uint32_t* dim(uint32_t d){ dims.push_back(d); return &dims.back(); }
};

struct PerlinGraph {
    Qnn_GraphHandle_t graph = nullptr;
    uint32_t n = 0;
    bool     fullPath = false;      // true = A, false = C
    std::string aFail;             // why path A could not be built, if it could not
    TensorArena arena;
    Qnn_Tensor_t inX{}, inY{}, inZ{}, out{};
    // Path C extra inputs: eight corner dot products plus three fade weights
    Qnn_Tensor_t inD[8]{}, inU{}, inV{}, inW{};
    std::vector<Qnn_Tensor_t> statics;
    // Constants. On this device HTP rejects a STATIC tensor as an
    // ElementWiseBinary operand: addNode returns 6005/6007 the moment one is
    // fed in, while the same op on two NATIVE tensors is accepted. Rather than
    // guess at the reason, path A is built twice - once with STATIC constants
    // and once with every constant as an APP_WRITE graph input - and whichever
    // one finalizes is the one that runs. constMode records which, because the
    // executor has to supply exactly the tensors the graph declares.
    int constMode = 0;                   // 0 = STATIC operand, 1 = APP_WRITE input
    std::vector<Qnn_Tensor_t>      cT;   // constMode 1: the constant input tensors
    std::vector<Qnn_DataType_t>    cDt;
    std::vector<float>             cFlt;
    std::vector<int32_t>           cInt;
    bool ready = false;
};

static std::map<uint32_t, PerlinGraph> g_perlinGraphs;
static uint32_t g_perlinSeq = 0;

static Qnn_Tensor_t mkT(TensorArena& A, const char* prefix, Qnn_TensorType_t type,
                        Qnn_DataType_t dt, uint32_t dim){
    Qnn_Tensor_t t = QNN_TENSOR_INIT;
    t.version = QNN_TENSOR_VERSION_1;
    t.v1.name = A.name(prefix);
    t.v1.type = type;
    t.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    t.v1.dataType = dt;
    t.v1.rank = 1;
    t.v1.dimensions = A.dim(dim);
    t.v1.memType = QNN_TENSORMEMTYPE_RAW;
    return t;
}

// A static tensor: its bytes are baked into the graph at finalize time, so the
// data buffer also has to stay alive. It lives in PerlinGraph::statics and the
// payload is appended to `keep`, which is owned by the caller's arena lifetime.
static Qnn_Tensor_t mkStatic(TensorArena& A, PerlinGraph& G, const char* prefix,
                             Qnn_DataType_t dt, uint32_t dim,
                             const void* bytes, size_t nbytes, std::deque<std::vector<uint8_t>>& keep){
    Qnn_Tensor_t t = mkT(A, prefix, QNN_TENSOR_TYPE_STATIC, dt, dim);
    keep.emplace_back((const uint8_t*)bytes, (const uint8_t*)bytes + nbytes);
    t.v1.clientBuf.data = keep.back().data();
    t.v1.clientBuf.dataSize = nbytes;
    G.statics.push_back(t);
    return t;
}

template<typename QnnApi>
static Qnn_ErrorHandle_t addNode(const QnnApi& f, Qnn_GraphHandle_t gh, const char* nm,
                                 const char* typeName, Qnn_Param_t* params, uint32_t np,
                                 Qnn_Tensor_t* ins, uint32_t ni, Qnn_Tensor_t* outs, uint32_t no){
    Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
    op.v1.name = nm;
    op.v1.packageName = "qti.aisw";
    op.v1.typeName = typeName;
    op.v1.numOfParams = np;
    op.v1.params = params;
    op.v1.numOfInputs = ni;
    op.v1.inputTensors = ins;
    op.v1.numOfOutputs = no;
    op.v1.outputTensors = outs;
    return f.graphAddNode(gh, op);
}

static Qnn_Param_t scalarParam(const char* pname, Qnn_DataType_t dt, uint32_t v){
    Qnn_Scalar_t s = QNN_SCALAR_INIT;
    s.dataType = dt;
    s.uint32Value = v;
    Qnn_Param_t p = QNN_PARAM_INIT;
    p.paramType = QNN_PARAMTYPE_SCALAR;
    p.name = pname;
    p.scalarParam = s;
    return p;
}

// ElementWiseBinary with a chosen operation code.
template<typename QnnApi>
static Qnn_ErrorHandle_t addBinary(const QnnApi& f, Qnn_GraphHandle_t gh, TensorArena& A,
                                   const char* tag, uint32_t opCode,
                                   Qnn_Tensor_t& a, Qnn_Tensor_t& b, Qnn_Tensor_t& o){
    Qnn_Param_t p = scalarParam(QNN_OP_ELEMENT_WISE_BINARY_PARAM_OPERATION,
                                QNN_DATATYPE_UINT_32, opCode);
    Qnn_Tensor_t ins[2] = {a, b};
    return addNode(f, gh, A.name(tag), QNN_OP_ELEMENT_WISE_BINARY, &p, 1, ins, 2, &o, 1);
}

template<typename QnnApi>
static Qnn_ErrorHandle_t addUnary(const QnnApi& f, Qnn_GraphHandle_t gh, TensorArena& A,
                                  const char* tag, uint32_t opCode,
                                  Qnn_Tensor_t& a, Qnn_Tensor_t& o){
    Qnn_Param_t p = scalarParam(QNN_OP_ELEMENT_WISE_UNARY_PARAM_OPERATION,
                                QNN_DATATYPE_UINT_32, opCode);
    return addNode(f, gh, A.name(tag), QNN_OP_ELEMENT_WISE_UNARY, &p, 1, &a, 1, &o, 1);
}

template<typename QnnApi>
static Qnn_ErrorHandle_t addGather(const QnnApi& f, Qnn_GraphHandle_t gh, TensorArena& A,
                                   const char* tag, Qnn_Tensor_t& table,
                                   Qnn_Tensor_t& idx, Qnn_Tensor_t& o){
    // axis=0: gather rows of the table. The table is rank 1 here, which is the
    // plain "table[i]" form the perm chain wants.
    Qnn_Param_t p = scalarParam(QNN_OP_GATHER_PARAM_AXIS, QNN_DATATYPE_UINT_32, 0);
    Qnn_Tensor_t ins[2] = {table, idx};
    return addNode(f, gh, A.name(tag), QNN_OP_GATHER, &p, 1, ins, 2, &o, 1);
}

// What the noise kernel needs, and what each path can do without.
static const char* kPerlinNeeds[] = {
    "Gather",
    "ElementWiseUnary",
    "ElementWiseBinary",
};

std::string runPerlinCap(){
    if(!g.ready || !g.api) return "ERR PERLIN_CAP NOT_READY";
    std::string present, missing;
    for(const char* w : kPerlinNeeds){
        if(hasOp(w)) present += present.empty() ? w : (std::string(",") + w);
        else         missing += missing.empty() ? w : (std::string(",") + w);
    }
    const bool gather = hasOp("Gather");
    const bool ewBin  = hasOp("ElementWiseBinary");
    const bool ewUn   = hasOp("ElementWiseUnary");
    const char* path  = (gather && ewBin && ewUn) ? "A_FULL_GRAPH"
                      : (ewBin ? "C_HYBRID" : "NONE");
    return "OK PERLIN_CAP path=" + std::string(path)
         + " present=" + present
         + " missing=" + missing
         + " ops_cached=" + std::to_string((unsigned)g_opsCache.size());
}

// ---------------------------------------------------------------------------
// Path A: the whole Perlin evaluation as one graph.
//
//   ix = floor(x) mod 256                              (per axis)
//   for each of the 8 corners (i,j,k):
//       h  = perm[ ix+i + perm[ iy+j + perm[ iz+k ] ] ] & 15
//       g  = grads[h]                                  (3 floats)
//       d  = dot(g, frac - (i,j,k))
//   u,v,w = fade(frac)                                 (t^3(t(t*6-15)+10))
//   result = trilinear blend of the eight d values
//
// perm is 512 entries and the chain indexes at most 255+1+255 = 511, so the
// "& 255" that the CPU reference needs on the outer index is unnecessary here;
// the table was already built doubled. That removes a mask op per corner.
// ---------------------------------------------------------------------------
static std::string buildPerlinFull(PerlinGraph& G, uint32_t n, int constMode){
    const auto& f = g.api->QNN_INTERFACE_VER_NAME;
    TensorArena& A = G.arena;
    // Static payloads are copied into here so they outlive this frame.
    static std::deque<std::vector<uint8_t>> g_staticKeep;

    G.n = n;
    G.constMode = constMode;
    const std::string gname = "mcnpu_perlin_" + std::to_string(++g_perlinSeq);
    Qnn_ErrorHandle_t rc = f.graphCreate(g.context, gname.c_str(), nullptr, &G.graph);
    if(rc != QNN_SUCCESS || !G.graph)
        return "ERR GRAPH_CREATE rc=" + std::to_string((int)rc) + " " + verbose(rc);

    const Qnn_DataType_t F = QNN_DATATYPE_FLOAT_32;
    const Qnn_DataType_t I = QNN_DATATYPE_INT_32;

    G.inX = mkT(A, "px", QNN_TENSOR_TYPE_APP_WRITE, F, n);
    G.inY = mkT(A, "py", QNN_TENSOR_TYPE_APP_WRITE, F, n);
    G.inZ = mkT(A, "pz", QNN_TENSOR_TYPE_APP_WRITE, F, n);
    G.out = mkT(A, "po", QNN_TENSOR_TYPE_APP_READ,  F, n);

    // Every constant is a length-n tensor, never a 1-element one. A backend is
    // not obliged to broadcast, and a 1-element operand is exactly the shape
    // that gets rejected on HTP.
    //
    // constMode 1 exists because HTP refused even the length-n STATIC form when
    // it was handed to ElementWiseBinary. Those constants become ordinary graph
    // inputs instead and the executor fills them, which trades IPC bytes for a
    // graph that builds at all.
    auto mkCI = [&](const char* tag, int32_t v)->Qnn_Tensor_t{
        if(constMode == 1){
            Qnn_Tensor_t t = mkT(A, tag, QNN_TENSOR_TYPE_APP_WRITE, I, n);
            Qnn_ErrorHandle_t r = f.tensorCreateGraphTensor(G.graph, &t);
            if(r != QNN_SUCCESS)
                return mkT(A, tag, QNN_TENSOR_TYPE_APP_WRITE, I, n);
            G.cT.push_back(t); G.cDt.push_back(I); G.cFlt.push_back(0.0f); G.cInt.push_back(v);
            return t;
        }
        std::vector<int32_t> vv((size_t)n, v);
        return mkStatic(A, G, tag, I, n, vv.data(), vv.size()*sizeof(int32_t), g_staticKeep);
    };
    auto mkCF = [&](const char* tag, float v)->Qnn_Tensor_t{
        if(constMode == 1){
            Qnn_Tensor_t t = mkT(A, tag, QNN_TENSOR_TYPE_APP_WRITE, F, n);
            Qnn_ErrorHandle_t r = f.tensorCreateGraphTensor(G.graph, &t);
            if(r != QNN_SUCCESS)
                return mkT(A, tag, QNN_TENSOR_TYPE_APP_WRITE, F, n);
            G.cT.push_back(t); G.cDt.push_back(F); G.cFlt.push_back(v); G.cInt.push_back(0);
            return t;
        }
        std::vector<float> vv((size_t)n, v);
        return mkStatic(A, G, tag, F, n, vv.data(), vv.size()*sizeof(float), g_staticKeep);
    };

    int32_t perm[512];
    for(int i = 0; i < 512; i++) perm[i] = i & 255;

    // Gradient tables indexed by p2 directly: gxT[t] = grad[t & 15]. That folds
    // the "& 15" into the table and deletes one int32 bit op per corner - the
    // least portable op in the whole graph.
    std::vector<float> tgx(256), tgy(256), tgz(256);
    for(int h = 0; h < 16; h++){
        const int s0 = (h & 1) ? -1 : 1;      // sign applied to u
        const int s1 = (h & 2) ? -1 : 1;      // sign applied to v
        float cx, cy, cz;
        if(h < 8){
            cx = (float)s0;  cy = (h < 4) ? (float)s1 : 0.0f;  cz = (h < 4) ? 0.0f : (float)s1;
        } else {
            cy = (float)s0;
            if(h == 12 || h == 14){ cx = (float)s1; cz = 0.0f; }
            else                  { cx = 0.0f;      cz = (float)s1; }
        }
        for(int t = h; t < 256; t += 16){ tgx[t] = cx; tgy[t] = cy; tgz[t] = cz; }
    }

    // perm tables with the corner offset baked in: pK[s][t] = perm[t + s].
    // That removes the three "+i/+j/+k" int32 adds per corner.
    std::vector<int32_t> pk0(512), pk1(512), pj0(512), pj1(512), pi0(512), pi1(512);
    for(int t = 0; t < 512; t++){
        pk0[t] = perm[t];  pk1[t] = perm[(t + 1) & 511];
        pj0[t] = perm[t];  pj1[t] = perm[(t + 1) & 511];
        pi0[t] = perm[t];  pi1[t] = perm[(t + 1) & 511];
    }

    Qnn_Tensor_t tPK[2] = {
        mkStatic(A, G, "pk", I, 512, pk0.data(), pk0.size()*sizeof(int32_t), g_staticKeep),
        mkStatic(A, G, "pk", I, 512, pk1.data(), pk1.size()*sizeof(int32_t), g_staticKeep)
    };
    Qnn_Tensor_t tPJ[2] = {
        mkStatic(A, G, "pj", I, 512, pj0.data(), pj0.size()*sizeof(int32_t), g_staticKeep),
        mkStatic(A, G, "pj", I, 512, pj1.data(), pj1.size()*sizeof(int32_t), g_staticKeep)
    };
    Qnn_Tensor_t tPI[2] = {
        mkStatic(A, G, "pi", I, 512, pi0.data(), pi0.size()*sizeof(int32_t), g_staticKeep),
        mkStatic(A, G, "pi", I, 512, pi1.data(), pi1.size()*sizeof(int32_t), g_staticKeep)
    };
    Qnn_Tensor_t tGX = mkStatic(A, G, "gx", F, 256, tgx.data(), tgx.size()*sizeof(float), g_staticKeep);
    Qnn_Tensor_t tGY = mkStatic(A, G, "gy", F, 256, tgy.data(), tgy.size()*sizeof(float), g_staticKeep);
    Qnn_Tensor_t tGZ = mkStatic(A, G, "gz", F, 256, tgz.data(), tgz.size()*sizeof(float), g_staticKeep);

    // 256 and 1/256 float, not int32. The lattice index used to be
    // "cast(fl) MOD 256", and MOD is the single op HTP refuses as a Gather
    // index producer: gather_nat (index straight out of Cast) builds fine
    // while gather_mod (index out of MOD) is rejected at finalize. The modulus
    // is now done in the float domain, where every op involved is confirmed
    // to build, and only the result is cast.
    Qnn_Tensor_t t256 = mkCF("c256", 256.0f);
    Qnn_Tensor_t tInv = mkCF("inv256", 1.0f/256.0f);
    Qnn_Tensor_t tOne = mkCF("f1",   1.0f);
    Qnn_Tensor_t tF6  = mkCF("f6",   6.0f);
    Qnn_Tensor_t tF15 = mkCF("f15", 15.0f);
    Qnn_Tensor_t tF10 = mkCF("f10", 10.0f);
    (void)mkCI;

    // Register everything before any node references it.
    std::vector<Qnn_Tensor_t*> reg;
    reg.push_back(&G.inX); reg.push_back(&G.inY); reg.push_back(&G.inZ); reg.push_back(&G.out);
    for(auto& t : G.statics) reg.push_back(&t);
    for(auto p : reg){
        rc = f.tensorCreateGraphTensor(G.graph, p);
        if(rc != QNN_SUCCESS)
            return "ERR TENSOR_CREATE name=" + std::string(p->v1.name)
                 + " rc=" + std::to_string((int)rc) + " " + verbose(rc);
    }

    auto mkNF = [&](const char* tag)->Qnn_Tensor_t{
        Qnn_Tensor_t t = mkT(A, tag, QNN_TENSOR_TYPE_NATIVE, F, n);
        f.tensorCreateGraphTensor(G.graph, &t);
        return t;
    };
    auto mkNI = [&](const char* tag)->Qnn_Tensor_t{
        Qnn_Tensor_t t = mkT(A, tag, QNN_TENSOR_TYPE_NATIVE, I, n);
        f.tensorCreateGraphTensor(G.graph, &t);
        return t;
    };

    // ---- per axis: floor, frac, frac-1, lattice index ----
    // Nodes are added in topological order. A tensor has to be produced by an
    // earlier addNode before another node may consume it; adding `sub` before
    // the `floor` that produces its second input is what made the first version
    // fail on the very first node.
    Qnn_Tensor_t fr[3], fm[3], idx[3];
    Qnn_Tensor_t* axesIn[3] = {&G.inX, &G.inY, &G.inZ};
    const char* axisTag[3] = {"x", "y", "z"};
    for(int a = 0; a < 3; a++){
        const std::string s(axisTag[a]);
        Qnn_Tensor_t fl = mkNF(("fl" + s).c_str());
        rc = addUnary(f, G.graph, A, "fl", QNN_OP_ELEMENT_WISE_UNARY_OPERATION_FLOOR,
                      *axesIn[a], fl);
        if(rc != QNN_SUCCESS) return "ERR NODE floor rc=" + std::to_string((int)rc) + " " + verbose(rc);

        fr[a] = mkNF(("fr" + s).c_str());
        rc = addBinary(f, G.graph, A, "sub", QNN_OP_ELEMENT_WISE_BINARY_OPERATION_SUBTRACT,
                       *axesIn[a], fl, fr[a]);
        if(rc != QNN_SUCCESS) return "ERR NODE sub rc=" + std::to_string((int)rc) + " " + verbose(rc);

        fm[a] = mkNF(("fm" + s).c_str());
        rc = addBinary(f, G.graph, A, "sub1", QNN_OP_ELEMENT_WISE_BINARY_OPERATION_SUBTRACT,
                       fr[a], tOne, fm[a]);
        if(rc != QNN_SUCCESS) return "ERR NODE sub1 rc=" + std::to_string((int)rc) + " " + verbose(rc);

        // idx = fl mod 256, in the float domain, then cast once.
        //   m = fl - 256 * floor(fl * (1/256))
        Qnn_Tensor_t sc = mkNF(("sc" + s).c_str());
        rc = addBinary(f, G.graph, A, "sc_mul", QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MULTIPLY,
                       fl, tInv, sc);
        if(rc != QNN_SUCCESS) return "ERR NODE sc_mul rc=" + std::to_string((int)rc) + " " + verbose(rc);

        Qnn_Tensor_t sq = mkNF(("sq" + s).c_str());
        rc = addUnary(f, G.graph, A, "sq_fl", QNN_OP_ELEMENT_WISE_UNARY_OPERATION_FLOOR,
                      sc, sq);
        if(rc != QNN_SUCCESS) return "ERR NODE sq_fl rc=" + std::to_string((int)rc) + " " + verbose(rc);

        Qnn_Tensor_t bk = mkNF(("bk" + s).c_str());
        rc = addBinary(f, G.graph, A, "bk_mul", QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MULTIPLY,
                       sq, t256, bk);
        if(rc != QNN_SUCCESS) return "ERR NODE bk_mul rc=" + std::to_string((int)rc) + " " + verbose(rc);

        Qnn_Tensor_t md = mkNF(("md" + s).c_str());
        rc = addBinary(f, G.graph, A, "md_sub", QNN_OP_ELEMENT_WISE_BINARY_OPERATION_SUBTRACT,
                       fl, bk, md);
        if(rc != QNN_SUCCESS) return "ERR NODE md_sub rc=" + std::to_string((int)rc) + " " + verbose(rc);

        idx[a] = mkNI(("ix" + s).c_str());
        rc = addNode(f, G.graph, A.name("cast"), QNN_OP_CAST, nullptr, 0, &md, 1, &idx[a], 1);
        if(rc != QNN_SUCCESS) return "ERR NODE cast rc=" + std::to_string((int)rc) + " " + verbose(rc);
    }

    std::string err, errAcc;

    // ---- fade(t) = t^3 * (t*(t*6 - 15) + 10) ----
    auto buildFade = [&](Qnn_Tensor_t& t, const char* tag)->Qnn_Tensor_t{
        Qnn_Tensor_t a = mkNF(tag), b = mkNF(tag), c = mkNF(tag);
        Qnn_Tensor_t t2 = mkNF(tag), t3 = mkNF(tag);
        Qnn_ErrorHandle_t r;
        r = addBinary(f, G.graph, A, "f_mul",  QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MULTIPLY, t,   tF6,  a);
        if(r != QNN_SUCCESS){ err = "ERR NODE f_mul rc="  + std::to_string((int)r) + " " + verbose(r); return a; }
        r = addBinary(f, G.graph, A, "f_sub",  QNN_OP_ELEMENT_WISE_BINARY_OPERATION_SUBTRACT, a,   tF15, b);
        if(r != QNN_SUCCESS){ err = "ERR NODE f_sub rc="  + std::to_string((int)r) + " " + verbose(r); return a; }
        r = addBinary(f, G.graph, A, "f_mul2", QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MULTIPLY, t,   b,    c);
        if(r != QNN_SUCCESS){ err = "ERR NODE f_mul2 rc=" + std::to_string((int)r) + " " + verbose(r); return a; }
        r = addBinary(f, G.graph, A, "f_add",  QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD,      c,   tF10, b);
        if(r != QNN_SUCCESS){ err = "ERR NODE f_add rc="  + std::to_string((int)r) + " " + verbose(r); return a; }
        r = addBinary(f, G.graph, A, "f_sq",   QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MULTIPLY, t,   t,    t2);
        if(r != QNN_SUCCESS){ err = "ERR NODE f_sq rc="   + std::to_string((int)r) + " " + verbose(r); return a; }
        r = addBinary(f, G.graph, A, "f_cu",   QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MULTIPLY, t2,  t,    t3);
        if(r != QNN_SUCCESS){ err = "ERR NODE f_cu rc="   + std::to_string((int)r) + " " + verbose(r); return a; }
        r = addBinary(f, G.graph, A, "f_out",  QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MULTIPLY, t3,  b,    a);
        if(r != QNN_SUCCESS){ err = "ERR NODE f_out rc="  + std::to_string((int)r) + " " + verbose(r); return a; }
        return a;
    };
    Qnn_Tensor_t u = buildFade(fr[0], "u");  if(!err.empty()) return err;
    Qnn_Tensor_t v = buildFade(fr[1], "v");  if(!err.empty()) return err;
    Qnn_Tensor_t w = buildFade(fr[2], "w");  if(!err.empty()) return err;

    // ---- eight corners ----
    Qnn_Tensor_t d[8];
    for(int c2 = 0; c2 < 8; c2++){
        const int ci = c2 & 1, cj = (c2 >> 1) & 1, ck = (c2 >> 2) & 1;
        Qnn_ErrorHandle_t r;

        Qnn_Tensor_t p0 = mkNI("p0");
        r = addGather(f, G.graph, A, "g0", tPK[ck], idx[2], p0);
        if(r != QNN_SUCCESS) errAcc += " g0=rc" + std::to_string((int)r);

        Qnn_Tensor_t q1 = mkNI("q1");
        r = addBinary(f, G.graph, A, "add_q1", QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD, idx[1], p0, q1);
        if(r != QNN_SUCCESS) errAcc += " add_q1=rc" + std::to_string((int)r);

        Qnn_Tensor_t p1 = mkNI("p1");
        r = addGather(f, G.graph, A, "g1", tPJ[cj], q1, p1);
        if(r != QNN_SUCCESS) errAcc += " g1=rc" + std::to_string((int)r);

        Qnn_Tensor_t q2 = mkNI("q2");
        r = addBinary(f, G.graph, A, "add_q2", QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD, idx[0], p1, q2);
        if(r != QNN_SUCCESS) errAcc += " add_q2=rc" + std::to_string((int)r);

        Qnn_Tensor_t p2 = mkNI("p2");
        r = addGather(f, G.graph, A, "g2", tPI[ci], q2, p2);
        if(r != QNN_SUCCESS) errAcc += " g2=rc" + std::to_string((int)r);

        Qnn_Tensor_t g0 = mkNF("ggx"), g1 = mkNF("ggy"), g2 = mkNF("ggz");
        r = addGather(f, G.graph, A, "ggx", tGX, p2, g0);
        if(r != QNN_SUCCESS) errAcc += " ggx=rc" + std::to_string((int)r);
        r = addGather(f, G.graph, A, "ggy", tGY, p2, g1);
        if(r != QNN_SUCCESS) errAcc += " ggy=rc" + std::to_string((int)r);
        r = addGather(f, G.graph, A, "ggz", tGZ, p2, g2);
        if(r != QNN_SUCCESS) errAcc += " ggz=rc" + std::to_string((int)r);

        // offset = frac - (ci,cj,ck): the 0/1 corner offset is the frac-minus-one
        // tensor when the bit is set, so no subtract node is needed here.
        Qnn_Tensor_t& ox = ci ? fm[0] : fr[0];
        Qnn_Tensor_t& oy = cj ? fm[1] : fr[1];
        Qnn_Tensor_t& oz = ck ? fm[2] : fr[2];

        Qnn_Tensor_t m0 = mkNF("m0"), m1 = mkNF("m1"), m2 = mkNF("m2");
        r = addBinary(f, G.graph, A, "m0", QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MULTIPLY, g0, ox, m0);
        if(r != QNN_SUCCESS) errAcc += " m0=rc" + std::to_string((int)r);
        r = addBinary(f, G.graph, A, "m1", QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MULTIPLY, g1, oy, m1);
        if(r != QNN_SUCCESS) errAcc += " m1=rc" + std::to_string((int)r);
        r = addBinary(f, G.graph, A, "m2", QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MULTIPLY, g2, oz, m2);
        if(r != QNN_SUCCESS) errAcc += " m2=rc" + std::to_string((int)r);

        Qnn_Tensor_t a01 = mkNF("a01");
        r = addBinary(f, G.graph, A, "a01", QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD, m0, m1, a01);
        if(r != QNN_SUCCESS) errAcc += " a01=rc" + std::to_string((int)r);

        d[c2] = mkNF("dc");
        r = addBinary(f, G.graph, A, "dc", QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD, a01, m2, d[c2]);
        if(r != QNN_SUCCESS) errAcc += " dc=rc" + std::to_string((int)r);
    }

    if(!errAcc.empty()) return "ERR NODES" + errAcc;

    // ---- trilinear blend: lerp(a,b,t) = a + t*(b - a) ----
    auto lerpInto = [&](Qnn_Tensor_t& a, Qnn_Tensor_t& b, Qnn_Tensor_t& t,
                        Qnn_Tensor_t& o, const char* tag)->bool{
        Qnn_Tensor_t df = mkNF(tag), pr = mkNF(tag);
        Qnn_ErrorHandle_t r;
        r = addBinary(f, G.graph, A, "lsub", QNN_OP_ELEMENT_WISE_BINARY_OPERATION_SUBTRACT, b, a, df);
        if(r != QNN_SUCCESS){ err = "ERR NODE lsub rc=" + std::to_string((int)r) + " " + verbose(r); return false; }
        r = addBinary(f, G.graph, A, "lmul", QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MULTIPLY, df, t, pr);
        if(r != QNN_SUCCESS){ err = "ERR NODE lmul rc=" + std::to_string((int)r) + " " + verbose(r); return false; }
        r = addBinary(f, G.graph, A, "ladd", QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD, a, pr, o);
        if(r != QNN_SUCCESS){ err = "ERR NODE ladd rc=" + std::to_string((int)r) + " " + verbose(r); return false; }
        return true;
    };
    Qnn_Tensor_t x00 = mkNF("x00"), x10 = mkNF("x10"), x01 = mkNF("x01"), x11 = mkNF("x11");
    Qnn_Tensor_t y0  = mkNF("y0"),  y1  = mkNF("y1");
    if(!lerpInto(d[0], d[1], u, x00, "x00")) return err;
    if(!lerpInto(d[2], d[3], u, x10, "x10")) return err;
    if(!lerpInto(d[4], d[5], u, x01, "x01")) return err;
    if(!lerpInto(d[6], d[7], u, x11, "x11")) return err;
    if(!lerpInto(x00, x10, v, y0, "y0")) return err;
    if(!lerpInto(x01, x11, v, y1, "y1")) return err;
    if(!lerpInto(y0, y1, w, G.out, "res")) return err;

    rc = f.graphFinalize(G.graph, nullptr, nullptr);
    if(rc != QNN_SUCCESS)
        return "ERR GRAPH_FINALIZE rc=" + std::to_string((int)rc) + " " + verbose(rc);
    G.fullPath = true;
    G.ready = true;
    return "";
}

// ---------------------------------------------------------------------------
// Path C: host resolves the tables, graph does only arithmetic.
//
// Inputs: d0..d7 (eight corner dot products) and u,v,w (fade weights).
// The graph computes the trilinear blend. This is the fallback for a backend
// without Gather - it needs ElementWiseBinary and nothing else.
// ---------------------------------------------------------------------------
static std::string buildPerlinHybrid(PerlinGraph& G, uint32_t n){
    const auto& f = g.api->QNN_INTERFACE_VER_NAME;
    TensorArena& A = G.arena;
    const Qnn_DataType_t F = QNN_DATATYPE_FLOAT_32;
    G.n = n;
    const std::string gname = "mcnpu_perlin_c_" + std::to_string(++g_perlinSeq);
    Qnn_ErrorHandle_t rc = f.graphCreate(g.context, gname.c_str(), nullptr, &G.graph);
    if(rc != QNN_SUCCESS || !G.graph)
        return "ERR GRAPH_CREATE rc=" + std::to_string((int)rc) + " " + verbose(rc);

    for(int c = 0; c < 8; c++){
        G.inD[c] = mkT(A, "d", QNN_TENSOR_TYPE_APP_WRITE, F, n);
        rc = f.tensorCreateGraphTensor(G.graph, &G.inD[c]);
        if(rc != QNN_SUCCESS) return "ERR TENSOR_CREATE d rc=" + std::to_string((int)rc);
    }
    G.inU = mkT(A, "u", QNN_TENSOR_TYPE_APP_WRITE, F, n);
    G.inV = mkT(A, "v", QNN_TENSOR_TYPE_APP_WRITE, F, n);
    G.inW = mkT(A, "w", QNN_TENSOR_TYPE_APP_WRITE, F, n);
    G.out = mkT(A, "o", QNN_TENSOR_TYPE_APP_READ,  F, n);
    Qnn_Tensor_t* rest[4] = {&G.inU, &G.inV, &G.inW, &G.out};
    for(auto p : rest){
        rc = f.tensorCreateGraphTensor(G.graph, p);
        if(rc != QNN_SUCCESS) return "ERR TENSOR_CREATE uvwo rc=" + std::to_string((int)rc);
    }

    // Every addNode return is checked. The previous version ignored them, so a
    // rejected node surfaced later as an anonymous GRAPH_FINALIZE failure that
    // pointed at nothing.
    //
    // The final lerp writes straight into G.out. The old graph ended with
    // "out = r * 1.0" purely to move a NATIVE tensor into the APP_READ output,
    // and that multiply by a STATIC constant was the one node HTP rejected -
    // which is why path C died at finalize while path A died at addNode.
    std::string herr;
    auto mkN = [&](const char* tag)->Qnn_Tensor_t{
        Qnn_Tensor_t t = mkT(A, tag, QNN_TENSOR_TYPE_NATIVE, F, n);
        rc = f.tensorCreateGraphTensor(G.graph, &t);
        if(rc != QNN_SUCCESS)
            herr = "ERR TENSOR_CREATE " + std::string(tag)
                 + " rc=" + std::to_string((int)rc);
        return t;
    };
    auto lerp = [&](Qnn_Tensor_t& a, Qnn_Tensor_t& b, Qnn_Tensor_t& t,
                    const char* tag, Qnn_Tensor_t& o) -> bool {
        Qnn_Tensor_t df = mkN(tag); if(!herr.empty()) return false;
        Qnn_ErrorHandle_t r = addBinary(f, G.graph, A, "s",
            QNN_OP_ELEMENT_WISE_BINARY_OPERATION_SUBTRACT, b, a, df);
        if(r != QNN_SUCCESS){ herr = "ERR NODE s(" + std::string(tag) + ") rc="
            + std::to_string((int)r) + " " + verbose(r); return false; }
        Qnn_Tensor_t pr = mkN(tag); if(!herr.empty()) return false;
        r = addBinary(f, G.graph, A, "m",
            QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MULTIPLY, df, t, pr);
        if(r != QNN_SUCCESS){ herr = "ERR NODE m(" + std::string(tag) + ") rc="
            + std::to_string((int)r) + " " + verbose(r); return false; }
        r = addBinary(f, G.graph, A, "a",
            QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD, a, pr, o);
        if(r != QNN_SUCCESS){ herr = "ERR NODE a(" + std::string(tag) + ") rc="
            + std::to_string((int)r) + " " + verbose(r); return false; }
        return true;
    };

    Qnn_Tensor_t x00 = mkN("x00"), x10 = mkN("x10"), x01 = mkN("x01"), x11 = mkN("x11");
    Qnn_Tensor_t y0  = mkN("y0"),  y1  = mkN("y1");
    if(!herr.empty()) return herr;
    if(!lerp(G.inD[0], G.inD[1], G.inU, "x00", x00)) return herr;
    if(!lerp(G.inD[2], G.inD[3], G.inU, "x10", x10)) return herr;
    if(!lerp(G.inD[4], G.inD[5], G.inU, "x01", x01)) return herr;
    if(!lerp(G.inD[6], G.inD[7], G.inU, "x11", x11)) return herr;
    if(!lerp(x00, x10, G.inV, "y0", y0)) return herr;
    if(!lerp(x01, x11, G.inV, "y1", y1)) return herr;
    if(!lerp(y0,  y1,  G.inW, "r",  G.out)) return herr;

    G.constMode = 0;
    rc = f.graphFinalize(G.graph, nullptr, nullptr);
    if(rc != QNN_SUCCESS)
        return "ERR GRAPH_FINALIZE rc=" + std::to_string((int)rc) + " " + verbose(rc);
    G.fullPath = false;
    G.ready = true;
    return "";
}

// ---------------------------------------------------------------------------
// One-shot graph capability diagnostic.
//
// This exists because four rounds of inferring the cause of a graph failure
// from the registered-op list produced four different wrong answers, each one
// shipped as a fix and each one wrong. Every probe below is a separate minimal
// graph isolating exactly one question and reporting the raw rc. It measures
// instead of inferring.
//
//   OK   - graphCreate, tensorCreate, addNode and graphFinalize all succeeded
//   rcN  - the QNN error handle from the stage named in the tag
// ---------------------------------------------------------------------------
static std::string runPerlinDiag(uint32_t n){
    if(!g.ready || !g.api || !g.context) return "ERR DIAG NOT_READY";
    const auto& f = g.api->QNN_INTERFACE_VER_NAME;
    if(n == 0 || n > 1024) n = 64;
    const Qnn_DataType_t F = QNN_DATATYPE_FLOAT_32;
    const Qnn_DataType_t I = QNN_DATATYPE_INT_32;
    std::string out;
    Qnn_GraphHandle_t gh = nullptr;
    int seq = 0;
    std::deque<std::vector<uint8_t>> keep;

    // One throwaway graph per probe. The budget is 8 and there is no per-graph
    // destroy in use here, so the budget check may reset the context between
    // probes. Diagnostic graphs are never executed, so that is acceptable.
    auto fresh = [&](const char* tag)->Qnn_ErrorHandle_t{
        ensureGraphBudget();
        g.graphCount++;
        const std::string nm = std::string("mcnpu_diag_") + tag + std::to_string(++seq);
        return f.graphCreate(g.context, nm.c_str(), nullptr, &gh);
    };
    auto reg = [&](TensorArena& A, const char* nm, Qnn_TensorType_t ty,
                   Qnn_DataType_t dt, uint32_t dim)->Qnn_Tensor_t{
        Qnn_Tensor_t t = mkT(A, nm, ty, dt, dim);
        f.tensorCreateGraphTensor(gh, &t);
        return t;
    };
    auto statT = [&](TensorArena& A, const char* nm, Qnn_DataType_t dt, uint32_t dim,
                     const void* bytes, size_t nbytes)->Qnn_Tensor_t{
        Qnn_Tensor_t t = mkT(A, nm, QNN_TENSOR_TYPE_STATIC, dt, dim);
        keep.emplace_back((const uint8_t*)bytes, (const uint8_t*)bytes + nbytes);
        t.v1.clientBuf.data = keep.back().data();
        t.v1.clientBuf.dataSize = nbytes;
        f.tensorCreateGraphTensor(gh, &t);
        return t;
    };
    auto rec = [&](const std::string& tag, Qnn_ErrorHandle_t r){
        out += std::string(" ") + tag + "="
             + (r == QNN_SUCCESS ? "OK" : ("rc" + std::to_string((int)r)));
    };
    auto fin = [&](const char* tag, Qnn_ErrorHandle_t r){
        if(r != QNN_SUCCESS){ rec(tag, r); return; }
        rec(tag, f.graphFinalize(gh, nullptr, nullptr));
    };
    auto lerp = [&](TensorArena& A, Qnn_Tensor_t& a, Qnn_Tensor_t& b,
                    Qnn_Tensor_t& t, Qnn_Tensor_t& o)->Qnn_ErrorHandle_t{
        Qnn_Tensor_t df = mkT(A, "df", QNN_TENSOR_TYPE_NATIVE, F, n);
        Qnn_ErrorHandle_t r = f.tensorCreateGraphTensor(gh, &df);
        if(r != QNN_SUCCESS) return r;
        r = addBinary(f, gh, A, "s", QNN_OP_ELEMENT_WISE_BINARY_OPERATION_SUBTRACT, b, a, df);
        if(r != QNN_SUCCESS) return r;
        Qnn_Tensor_t pr = mkT(A, "pr", QNN_TENSOR_TYPE_NATIVE, F, n);
        r = f.tensorCreateGraphTensor(gh, &pr);
        if(r != QNN_SUCCESS) return r;
        r = addBinary(f, gh, A, "m", QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MULTIPLY, df, t, pr);
        if(r != QNN_SUCCESS) return r;
        return addBinary(f, gh, A, "a", QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD, a, pr, o);
    };

    // p_lerp: one 3-node fp32 chain ending in APP_READ. This is exactly how
    // path C ends and how the working ADD graph ends, so a failure here means
    // the arithmetic chain itself is the problem, not Perlin.
    {
        Qnn_ErrorHandle_t r = fresh("lerp");
        if(r != QNN_SUCCESS){ rec("lerp_crt", r); return out; }
        TensorArena A;
        Qnn_Tensor_t a = reg(A, "a", QNN_TENSOR_TYPE_APP_WRITE, F, n);
        Qnn_Tensor_t b = reg(A, "b", QNN_TENSOR_TYPE_APP_WRITE, F, n);
        Qnn_Tensor_t t = reg(A, "t", QNN_TENSOR_TYPE_APP_WRITE, F, n);
        Qnn_Tensor_t o = reg(A, "o", QNN_TENSOR_TYPE_APP_READ,  F, n);
        fin("lerp", lerp(A, a, b, t, o));
    }
    // p_cast_i32: fp32 -> int32, straight into an int32 output. Everything in
    // path A's index chain depends on this producing an int32 tensor at all.
    {
        Qnn_ErrorHandle_t r = fresh("cast");
        if(r != QNN_SUCCESS){ rec("cast_crt", r); return out; }
        TensorArena A;
        Qnn_Tensor_t a = reg(A, "a", QNN_TENSOR_TYPE_APP_WRITE, F, n);
        Qnn_Tensor_t o = reg(A, "o", QNN_TENSOR_TYPE_APP_READ,  I, n);
        fin("cast_i32", addNode(f, gh, A.name("cast"), QNN_OP_CAST,
                                nullptr, 0, &a, 1, &o, 1));
    }
    // p_gather_fwd: {table, indices} - the operand order buildPerlinFull uses.
    {
        Qnn_ErrorHandle_t r = fresh("gf");
        if(r != QNN_SUCCESS){ rec("gf_crt", r); return out; }
        TensorArena A;
        std::vector<int32_t> tbl(512);
        for(size_t i = 0; i < tbl.size(); i++) tbl[i] = (int32_t)(i & 255);
        Qnn_Tensor_t T  = statT(A, "T",  I, 512, tbl.data(), tbl.size()*sizeof(int32_t));
        Qnn_Tensor_t ix = reg(A, "ix", QNN_TENSOR_TYPE_APP_WRITE, I, n);
        Qnn_Tensor_t o  = reg(A, "o",  QNN_TENSOR_TYPE_APP_READ,  I, n);
        fin("gather_fwd", addGather(f, gh, A, "g", T, ix, o));
    }
    // ---- v2: can a multi-node graph finalize at all? ----
    //
    // Every graph that has ever finalized on this device - ADD, MATMUL, the
    // smoke graph - has exactly one node and only APP_WRITE/APP_READ tensors.
    // Path C chains seven lerps through NATIVE intermediates and has never
    // finalized. The first version of this diagnostic folded addNode and
    // graphFinalize into a single rc, so "lerp=rc1002" could not say which
    // stage rejected, and it varied chain length and NATIVE together. This
    // sweeps them apart and names the stage that actually fails.
    auto chain = [&](const char* tag, int nodes, bool regNative){
        Qnn_ErrorHandle_t r = fresh(tag);
        if(r != QNN_SUCCESS){ rec(std::string(tag)+"_crt", r); return; }
        TensorArena A;
        Qnn_Tensor_t a = reg(A, "a", QNN_TENSOR_TYPE_APP_WRITE, F, n);
        Qnn_Tensor_t b = reg(A, "b", QNN_TENSOR_TYPE_APP_WRITE, F, n);
        Qnn_Tensor_t o = reg(A, "o", QNN_TENSOR_TYPE_APP_READ,  F, n);
        Qnn_Tensor_t cur = a;
        for(int i = 0; i < nodes; i++){
            const bool last = (i == nodes-1);
            Qnn_Tensor_t t = last ? o : mkT(A, "c", QNN_TENSOR_TYPE_NATIVE, F, n);
            if(!last && regNative){
                r = f.tensorCreateGraphTensor(gh, &t);
                if(r != QNN_SUCCESS){ rec(std::string(tag)+"_tcr"+std::to_string(i), r); return; }
            }
            r = addBinary(f, gh, A, "ad", QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD, cur, b, t);
            if(r != QNN_SUCCESS){ rec(std::string(tag)+"_nod"+std::to_string(i), r); return; }
            cur = t;
        }
        rec(tag, f.graphFinalize(gh, nullptr, nullptr));
    };
    chain("ch1",  1, true);
    chain("ch2r", 2, true);
    chain("ch3r", 3, true);
    // Gather accepted an int32 table and Cast produced int32, while a float
    // table was refused. Gradient components are -1/0/1 and fit int32 exactly,
    // so path A can gather ints and cast afterwards. This is that direction.
    {
        Qnn_ErrorHandle_t r = fresh("c2f");
        if(r != QNN_SUCCESS){ rec("c2f_crt", r); return out; }
        TensorArena A;
        Qnn_Tensor_t a = reg(A, "a", QNN_TENSOR_TYPE_APP_WRITE, I, n);
        Qnn_Tensor_t o = reg(A, "o", QNN_TENSOR_TYPE_APP_READ,  F, n);
        fin("cast_i32_to_f32", addNode(f, gh, A.name("cast"), QNN_OP_CAST,
                                       nullptr, 0, &a, 1, &o, 1));
    }
    // ---- v3: two confounded variables, separated ----
    //
    // v2 changed two things at once. Relative to ch3r, the p_lerp probe had
    // three graph inputs instead of two AND used SUBTRACT/MULTIPLY where ch3r
    // used only ADD. lerp failed and ch3r passed, which fits "more than two
    // graph inputs is refused" and "ops other than ADD are refused" equally
    // well. Path C has eleven inputs and uses SUBTRACT/MULTIPLY, so it cannot
    // tell them apart either. These vary one thing at a time.
    auto soloOp = [&](const char* tag, uint32_t op){
        Qnn_ErrorHandle_t r = fresh(tag);
        if(r != QNN_SUCCESS){ rec(std::string(tag)+"_crt", r); return; }
        TensorArena A;
        Qnn_Tensor_t a = reg(A, "a", QNN_TENSOR_TYPE_APP_WRITE, F, n);
        Qnn_Tensor_t b = reg(A, "b", QNN_TENSOR_TYPE_APP_WRITE, F, n);
        Qnn_Tensor_t o = reg(A, "o", QNN_TENSOR_TYPE_APP_READ,  F, n);
        r = addBinary(f, gh, A, "b0", op, a, b, o);
        if(r != QNN_SUCCESS){ rec(std::string(tag)+"_nod", r); return; }
        rec(tag, f.graphFinalize(gh, nullptr, nullptr));
    };
    soloOp("op_add", QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD);
    soloOp("op_sub", QNN_OP_ELEMENT_WISE_BINARY_OPERATION_SUBTRACT);
    soloOp("op_mul", QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MULTIPLY);

    // Same node count and same op as ch3r (three nodes, all ADD). Only the
    // number of graph inputs differs, so in3 vs ch3r isolates input count.
    auto inCount = [&](const char* tag, int nIn){
        Qnn_ErrorHandle_t r = fresh(tag);
        if(r != QNN_SUCCESS){ rec(std::string(tag)+"_crt", r); return; }
        TensorArena A;
        std::vector<Qnn_Tensor_t> ins;
        for(int i = 0; i < nIn; i++)
            ins.push_back(reg(A, "i", QNN_TENSOR_TYPE_APP_WRITE, F, n));
        Qnn_Tensor_t o = reg(A, "o", QNN_TENSOR_TYPE_APP_READ, F, n);
        Qnn_Tensor_t cur = ins[0];
        for(int i = 1; i < nIn; i++){
            const bool last = (i == nIn - 1);
            Qnn_Tensor_t t = last ? o : mkT(A, "c", QNN_TENSOR_TYPE_NATIVE, F, n);
            if(!last){
                r = f.tensorCreateGraphTensor(gh, &t);
                if(r != QNN_SUCCESS){ rec(std::string(tag)+"_tcr", r); return; }
            }
            r = addBinary(f, gh, A, "ad", QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD,
                          cur, ins[i], t);
            if(r != QNN_SUCCESS){ rec(std::string(tag)+"_nod"+std::to_string(i), r); return; }
            cur = t;
        }
        rec(tag, f.graphFinalize(gh, nullptr, nullptr));
    };
    inCount("in3",  3);
    inCount("in11", 11);

    // ---- Gather with a NATIVE index and a NATIVE output ----
    //
    // gather_fwd proved a STATIC int32 table with an APP_WRITE index into an
    // APP_READ output is accepted. buildPerlinFull hands the same node a NATIVE
    // index produced by Cast/MOD and writes into a NATIVE output, and that is
    // the only difference left when g0 fails with rc=6005. This mirrors it.
    {
        Qnn_ErrorHandle_t r = fresh("gn");
        if(r != QNN_SUCCESS){ rec("gather_nat_crt", r); return out; }
        TensorArena A;
        std::vector<int32_t> tbl(512);
        for(size_t i = 0; i < tbl.size(); i++) tbl[i] = (int32_t)(i & 255);
        Qnn_Tensor_t T   = statT(A, "T", I, 512, tbl.data(), tbl.size()*sizeof(int32_t));
        Qnn_Tensor_t ixf = reg(A, "ixf", QNN_TENSOR_TYPE_APP_WRITE, F, n);
        Qnn_Tensor_t ci  = mkT(A, "ci", QNN_TENSOR_TYPE_NATIVE, I, n);
        r = f.tensorCreateGraphTensor(gh, &ci);
        if(r != QNN_SUCCESS){ rec("gather_nat_tcr", r); return out; }
        r = addNode(f, gh, A.name("cast"), QNN_OP_CAST, nullptr, 0, &ixf, 1, &ci, 1);
        if(r != QNN_SUCCESS){ rec("gather_nat_cast", r); return out; }
        Qnn_Tensor_t go  = mkT(A, "go", QNN_TENSOR_TYPE_NATIVE, I, n);
        r = f.tensorCreateGraphTensor(gh, &go);
        if(r != QNN_SUCCESS){ rec("gather_nat_tcr2", r); return out; }
        r = addGather(f, gh, A, "g", T, ci, go);
        if(r != QNN_SUCCESS){ rec("gather_nat_nod", r); return out; }
        Qnn_Tensor_t o = reg(A, "o", QNN_TENSOR_TYPE_APP_READ, F, n);
        r = addNode(f, gh, A.name("c2"), QNN_OP_CAST, nullptr, 0, &go, 1, &o, 1);
        if(r != QNN_SUCCESS){ rec("gather_nat_c2", r); return out; }
        rec("gather_nat", f.graphFinalize(gh, nullptr, nullptr));
    }

    // ---- v4: chain shape held constant, op code varied ----
    //
    // ch3r is three ADD nodes through two registered NATIVE tensors and it
    // finalizes. p_lerp is three nodes of SUBTRACT/MULTIPLY/ADD through two
    // registered NATIVE tensors and it does not. Same node count, same tensor
    // types, same operand classes - only the op codes differ. If this fails,
    // ops other than ADD cannot write a NATIVE tensor here and path C cannot be
    // built as written.
    {
        Qnn_ErrorHandle_t r = fresh("chs");
        if(r != QNN_SUCCESS){ rec("chs3_crt", r); return out; }
        TensorArena A;
        Qnn_Tensor_t a = reg(A, "a", QNN_TENSOR_TYPE_APP_WRITE, F, n);
        Qnn_Tensor_t b = reg(A, "b", QNN_TENSOR_TYPE_APP_WRITE, F, n);
        Qnn_Tensor_t o = reg(A, "o", QNN_TENSOR_TYPE_APP_READ,  F, n);
        Qnn_Tensor_t cur = a;
        bool bad = false;
        for(int i = 0; i < 3 && !bad; i++){
            const bool last = (i == 2);
            Qnn_Tensor_t t = last ? o : mkT(A, "c", QNN_TENSOR_TYPE_NATIVE, F, n);
            if(!last){
                r = f.tensorCreateGraphTensor(gh, &t);
                if(r != QNN_SUCCESS){ rec("chs3_tcr", r); bad = true; break; }
            }
            r = addBinary(f, gh, A, "sb", QNN_OP_ELEMENT_WISE_BINARY_OPERATION_SUBTRACT, cur, b, t);
            if(r != QNN_SUCCESS){ rec("chs3_nod", r); bad = true; break; }
            cur = t;
        }
        if(!bad) rec("chs3", f.graphFinalize(gh, nullptr, nullptr));
    }
    // ---- int32 ADD writing a NATIVE int32 ----
    //
    // Path A adds the perm results in int32 between gathers (idx+p0, idx+p1).
    // Every op probe so far was float, so int32 ADD has never actually been
    // confirmed here. If this fails, the perm chain has to move into the float
    // domain or the +1 has to be folded into the tables the way the corner
    // offsets already are.
    {
        Qnn_ErrorHandle_t r = fresh("ai");
        if(r != QNN_SUCCESS){ rec("add_i32_crt", r); return out; }
        TensorArena A;
        Qnn_Tensor_t a = reg(A, "a", QNN_TENSOR_TYPE_APP_WRITE, I, n);
        Qnn_Tensor_t b = reg(A, "b", QNN_TENSOR_TYPE_APP_WRITE, I, n);
        Qnn_Tensor_t c = mkT(A, "c", QNN_TENSOR_TYPE_NATIVE, I, n);
        r = f.tensorCreateGraphTensor(gh, &c);
        if(r != QNN_SUCCESS){ rec("add_i32_tcr", r); return out; }
        r = addBinary(f, gh, A, "ai", QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD, a, b, c);
        if(r != QNN_SUCCESS){ rec("add_i32_nod", r); return out; }
        Qnn_Tensor_t o = reg(A, "o", QNN_TENSOR_TYPE_APP_READ, F, n);
        r = addNode(f, gh, A.name("c2f"), QNN_OP_CAST, nullptr, 0, &c, 1, &o, 1);
        if(r != QNN_SUCCESS){ rec("add_i32_cast", r); return out; }
        rec("add_i32", f.graphFinalize(gh, nullptr, nullptr));
    }
    // ---- Gather whose index comes from a float-domain modulus ----
    //
    // gather_mod showed HTP rejects a Gather index that MOD produced, which is
    // why g0 failed while every other probe passed. Path A now computes
    // fl mod 256 as fl - 256*floor(fl*(1/256)) using only ops already confirmed
    // to build, and casts once at the end. This replays exactly that index
    // chain into a Gather so the replacement is verified before path A depends
    // on it.
    {
        Qnn_ErrorHandle_t r = fresh("gf");
        if(r != QNN_SUCCESS){ rec("gather_fms_crt", r); return out; }
        TensorArena A;
        std::vector<int32_t> tbl(512);
        for(size_t i = 0; i < tbl.size(); i++) tbl[i] = (int32_t)(i & 255);
        Qnn_Tensor_t T    = statT(A, "T", I, 512, tbl.data(), tbl.size()*sizeof(int32_t));
        Qnn_Tensor_t x    = reg(A, "x",    QNN_TENSOR_TYPE_APP_WRITE, F, n);
        Qnn_Tensor_t cInv = reg(A, "inv",  QNN_TENSOR_TYPE_APP_WRITE, F, n);
        Qnn_Tensor_t c256 = reg(A, "c256", QNN_TENSOR_TYPE_APP_WRITE, F, n);
        Qnn_Tensor_t sc = mkT(A, "sc", QNN_TENSOR_TYPE_NATIVE, F, n);
        Qnn_Tensor_t sq = mkT(A, "sq", QNN_TENSOR_TYPE_NATIVE, F, n);
        Qnn_Tensor_t bk = mkT(A, "bk", QNN_TENSOR_TYPE_NATIVE, F, n);
        Qnn_Tensor_t md = mkT(A, "md", QNN_TENSOR_TYPE_NATIVE, F, n);
        Qnn_Tensor_t ix = mkT(A, "ix", QNN_TENSOR_TYPE_NATIVE, I, n);
        Qnn_Tensor_t go = mkT(A, "go", QNN_TENSOR_TYPE_NATIVE, I, n);
        Qnn_Tensor_t* nat[6] = {&sc, &sq, &bk, &md, &ix, &go};
        for(auto p : nat){
            r = f.tensorCreateGraphTensor(gh, p);
            if(r != QNN_SUCCESS){ rec("gather_fms_tcr", r); return out; }
        }
        r = addBinary(f, gh, A, "sc", QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MULTIPLY, x, cInv, sc);
        if(r != QNN_SUCCESS){ rec("gather_fms_sc", r); return out; }
        r = addUnary(f, gh, A, "sq", QNN_OP_ELEMENT_WISE_UNARY_OPERATION_FLOOR, sc, sq);
        if(r != QNN_SUCCESS){ rec("gather_fms_sq", r); return out; }
        r = addBinary(f, gh, A, "bk", QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MULTIPLY, sq, c256, bk);
        if(r != QNN_SUCCESS){ rec("gather_fms_bk", r); return out; }
        r = addBinary(f, gh, A, "md", QNN_OP_ELEMENT_WISE_BINARY_OPERATION_SUBTRACT, x, bk, md);
        if(r != QNN_SUCCESS){ rec("gather_fms_md", r); return out; }
        r = addNode(f, gh, A.name("cast"), QNN_OP_CAST, nullptr, 0, &md, 1, &ix, 1);
        if(r != QNN_SUCCESS){ rec("gather_fms_cast", r); return out; }
        r = addGather(f, gh, A, "g", T, ix, go);
        if(r != QNN_SUCCESS){ rec("gather_fms_nod", r); return out; }
        Qnn_Tensor_t o = reg(A, "o", QNN_TENSOR_TYPE_APP_READ, F, n);
        r = addNode(f, gh, A.name("c2"), QNN_OP_CAST, nullptr, 0, &go, 1, &o, 1);
        if(r != QNN_SUCCESS){ rec("gather_fms_c2", r); return out; }
        rec("gather_fms", f.graphFinalize(gh, nullptr, nullptr));
    }
    return out;
}

// CPU reference used for the correctness check. Deliberately written the way a
// textbook Perlin is, not the way the graph is, so that a shared bug between the
// two cannot produce a green result.
static double perlinRef(double x, double y, double z, const int* perm){
    auto fade = [](double t){ return t*t*t*(t*(t*6-15)+10); };
    auto grad = [](int h, double x, double y, double z){
        double u = h<8 ? x : y;
        double v = h<4 ? y : (h==12||h==14 ? x : z);
        return ((h&1)==0?u:-u) + ((h&2)==0?v:-v);
    };
    int X = (int)floor(x) & 255, Y = (int)floor(y) & 255, Z = (int)floor(z) & 255;
    x -= floor(x); y -= floor(y); z -= floor(z);
    double u = fade(x), v = fade(y), w = fade(z);
    int A = perm[X]+Y, AA = perm[A]+Z, AB = perm[A+1]+Z;
    int B = perm[X+1]+Y, BA = perm[B]+Z, BB = perm[B+1]+Z;
    auto lerp = [](double t,double a,double b){ return a + t*(b-a); };
    return lerp(w, lerp(v, lerp(u, grad(perm[AA],x,y,z),     grad(perm[BA],x-1,y,z)),
                           lerp(u, grad(perm[AB],x,y-1,z),   grad(perm[BB],x-1,y-1,z))),
                   lerp(v, lerp(u, grad(perm[AA+1],x,y,z-1), grad(perm[BA+1],x-1,y,z-1)),
                           lerp(u, grad(perm[AB+1],x,y-1,z-1), grad(perm[BB+1],x-1,y-1,z-1))));
}

// Build, run, and compare against the reference. n is capped by the element
// budget: path A writes 3 inputs plus one output per point.
std::string runPerlinBench(uint32_t n){
    if(!g.ready || !g.api || !g.context) return "ERR PERLIN NOT_READY";
    if(n == 0 || n > 16384) n = 4096;
    if(!ensureGraphBudget()) return "ERR GRAPH_BUDGET_EXHAUSTED";

    const bool gather = hasOp("Gather");
    const bool ewBin  = hasOp("ElementWiseBinary");
    const bool ewUn   = hasOp("ElementWiseUnary");
    if(!ewBin) return "ERR PERLIN NO_ELEMENTWISE_BINARY - neither path can be built";
    const bool useFull = gather && ewUn;

    auto it = g_perlinGraphs.find(n);
    if(it != g_perlinGraphs.end() && it->second.fullPath != useFull)
        g_perlinGraphs.erase(it);            // capability changed; rebuild
    auto found = g_perlinGraphs.find(n);
    PerlinGraph* G = nullptr;
    if(found != g_perlinGraphs.end()){
        G = &found->second;
    } else {
        std::string aFail;
        if(useFull){
            // STATIC constants first, then constants-as-graph-inputs. Both are
            // cheap to build and the second mode exists only because HTP
            // rejected the first; whichever one finalizes is the one that runs,
            // and both failure strings are carried out so a rejected op never
            // gets mistaken for "Perlin cannot be built here".
            const int modes[2] = {0, 1};
            for(int m = 0; m < 2 && !G; m++){
                PerlinGraph cg;
                std::string e = buildPerlinFull(cg, n, modes[m]);
                if(e.empty()){
                    auto ins = g_perlinGraphs.emplace(n, std::move(cg));
                    G = &ins.first->second;
                } else {
                    if(!aFail.empty()) aFail += " | ";
                    aFail += "cm" + std::to_string(modes[m]) + "=" + e;
                }
            }
        }
        if(!G){
            // Path A would not build. Fall back to C so a run still yields a
            // measured number instead of only an error, and carry the reason
            // out in the report - otherwise one rejected op hides whether the
            // arithmetic itself is any good.
            PerlinGraph cg;
            std::string cErr = buildPerlinHybrid(cg, n);
            if(!cErr.empty()){
                // Both paths refused. Run the diagnostic rather than report
                // only the failure, so the next attempt starts from a measured
                // answer instead of another hypothesis about which op is at
                // fault.
                const std::string d = runPerlinDiag(n);
                return "ERR PERLIN BUILD A=[" + aFail + "] C=[" + cErr
                     + "] DIAG=[" + d + "]";
            }
            auto ins = g_perlinGraphs.emplace(n, std::move(cg));
            G = &ins.first->second;
            G->aFail = aFail;
        }
    }

    const auto& f = g.api->QNN_INTERFACE_VER_NAME;
    std::vector<float> xs(n), ys(n), zs(n), out(n, 0.0f);
    // Non-power-of-two spacing on purpose: with 4-aligned coordinates the
    // fractional parts are always 0, which makes fade and the interpolation
    // degenerate and lets a broken kernel pass.
    for(uint32_t i = 0; i < n; i++){
        xs[i] = (float)i * (1.0f/96.0f) + 0.017f;
        ys[i] = (float)i * (1.0f/17.0f) + 0.31f;
        zs[i] = (float)i * (1.0f/5.0f)  + 0.7f;
    }

    int32_t perm[512];
    for(int i = 0; i < 512; i++) perm[i] = i & 255;

    Qnn_ErrorHandle_t rc = QNN_SUCCESS;
    auto t0 = std::chrono::steady_clock::now();
    if(G->fullPath){
        Qnn_Tensor_t ex = G->inX, ey = G->inY, ez = G->inZ, eo = G->out;
        ex.v1.clientBuf.data = xs.data(); ex.v1.clientBuf.dataSize = n*sizeof(float);
        ey.v1.clientBuf.data = ys.data(); ey.v1.clientBuf.dataSize = n*sizeof(float);
        ez.v1.clientBuf.data = zs.data(); ez.v1.clientBuf.dataSize = n*sizeof(float);
        eo.v1.clientBuf.data = out.data(); eo.v1.clientBuf.dataSize = n*sizeof(float);
        std::vector<Qnn_Tensor_t> inList;
        inList.push_back(ex); inList.push_back(ey); inList.push_back(ez);
        // constMode 1: the constants are ordinary graph inputs, so one buffer
        // per constant has to be filled before every execute.
        std::vector<std::vector<float>>   fbuf;
        std::vector<std::vector<int32_t>> ibuf;
        for(size_t k = 0; k < G->cT.size(); k++){
            Qnn_Tensor_t t = G->cT[k];
            if(G->cDt[k] == QNN_DATATYPE_FLOAT_32){
                fbuf.emplace_back((size_t)n, G->cFlt[k]);
                t.v1.clientBuf.data = fbuf.back().data();
                t.v1.clientBuf.dataSize = n*sizeof(float);
            } else {
                ibuf.emplace_back((size_t)n, G->cInt[k]);
                t.v1.clientBuf.data = ibuf.back().data();
                t.v1.clientBuf.dataSize = n*sizeof(int32_t);
            }
            inList.push_back(t);
        }
        rc = f.graphExecute(G->graph, inList.data(), (uint32_t)inList.size(),
                            &eo, 1, nullptr, nullptr);
    } else {
        // host side: perm chain, gradient table and dot products
        std::vector<float> d[8];
        for(int c = 0; c < 8; c++) d[c].resize(n);
        std::vector<float> u(n), v(n), w(n);
        for(uint32_t i = 0; i < n; i++){
            double x = xs[i], y = ys[i], z = zs[i];
            double fx = x - floor(x), fy = y - floor(y), fz = z - floor(z);
            int X = (int)floor(x) & 255, Y = (int)floor(y) & 255, Z = (int)floor(z) & 255;
            auto fadeT = [](double t){ return t*t*t*(t*(t*6-15)+10); };
            u[i] = (float)fadeT(fx); v[i] = (float)fadeT(fy); w[i] = (float)fadeT(fz);
            for(int c = 0; c < 8; c++){
                const int ci = c & 1, cj = (c>>1)&1, ck = (c>>2)&1;
                int h = perm[X + ci + perm[Y + cj + perm[Z + ck]]] & 15;
                const int s0 = (h & 1) ? -1 : 1;
                const int s1 = (h & 2) ? -1 : 1;
                double cgx, cgy, cgz;
                if(h < 8){
                    cgx = s0; cgy = (h < 4) ? s1 : 0; cgz = (h < 4) ? 0 : s1;
                } else {
                    cgy = s0;
                    if(h == 12 || h == 14){ cgx = s1; cgz = 0; }
                    else                  { cgx = 0;  cgz = s1; }
                }
                double ox = fx - ci, oy = fy - cj, oz = fz - ck;
                d[c][i] = (float)(cgx*ox + cgy*oy + cgz*oz);
            }
        }
        Qnn_Tensor_t in[11];
        for(int c = 0; c < 8; c++){
            in[c] = G->inD[c];
            in[c].v1.clientBuf.data = d[c].data();
            in[c].v1.clientBuf.dataSize = n*sizeof(float);
        }
        in[8] = G->inU; in[8].v1.clientBuf.data = u.data(); in[8].v1.clientBuf.dataSize = n*sizeof(float);
        in[9] = G->inV; in[9].v1.clientBuf.data = v.data(); in[9].v1.clientBuf.dataSize = n*sizeof(float);
        in[10]= G->inW; in[10].v1.clientBuf.data = w.data(); in[10].v1.clientBuf.dataSize = n*sizeof(float);
        Qnn_Tensor_t eo = G->out;
        eo.v1.clientBuf.data = out.data(); eo.v1.clientBuf.dataSize = n*sizeof(float);
        rc = f.graphExecute(G->graph, in, 11, &eo, 1, nullptr, nullptr);
    }
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - t0).count();
    if(rc != QNN_SUCCESS)
        return "ERR PERLIN EXECUTE rc=" + std::to_string((int)rc) + " " + verbose(rc);

    // compare
    double maxAbs = 0.0; uint32_t bad = 0;
    auto cpu0 = std::chrono::steady_clock::now();
    for(uint32_t i = 0; i < n; i++){
        double r = perlinRef(xs[i], ys[i], zs[i], perm);
        double e = fabs(r - out[i]);
        if(e > maxAbs) maxAbs = e;
        if(e > 1e-4) bad++;
    }
    auto cpuUs = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - cpu0).count();

    return "OK PERLIN path=" + std::string(G->fullPath ? "A_FULL" : "C_HYBRID")
         + " cm=" + std::to_string(G->constMode)
         + (G->aFail.empty() ? std::string("") : (" a_fail=[" + G->aFail + "]"))
         + " n=" + std::to_string((unsigned)n)
         + " bad=" + std::to_string((unsigned)bad) + "/" + std::to_string((unsigned)n)
         + " maxAbs=" + std::to_string(maxAbs)
         + " npu_us=" + std::to_string((long long)us)
         + " cpu_ref_us=" + std::to_string((long long)cpuUs);
}

extern "C" JNIEXPORT jstring JNICALL Java_bslsjdk_mcnpu_NpuRuntime_nativePerlinCap(JNIEnv* e, jclass){
    std::lock_guard<std::mutex> lock(gRuntimeMutex);
    std::string r = runPerlinCap();
    I("PERLIN_CAP %s", r.c_str());
    return e->NewStringUTF(r.c_str());
}

extern "C" JNIEXPORT jstring JNICALL Java_bslsjdk_mcnpu_NpuRuntime_nativePerlinBench(JNIEnv* e, jclass, jint n){
    std::lock_guard<std::mutex> lock(gRuntimeMutex);
    std::string r = runPerlinBench((uint32_t)n);
    I("PERLIN %s", r.c_str());
    return e->NewStringUTF(r.c_str());
}

extern "C" JNIEXPORT jstring JNICALL Java_bslsjdk_mcnpu_NpuRuntime_nativeAddMax(JNIEnv* e,jclass){
    std::lock_guard<std::mutex> lock(gRuntimeMutex);
    return e->NewStringUTF(std::to_string(g_addLadderMax).c_str());
}
extern "C" JNIEXPORT jstring JNICALL Java_bslsjdk_mcnpu_NpuRuntime_nativeMatMul(JNIEnv* e,jclass,jint m,jint k,jint n){
    if(m<=0||k<=0||n<=0) return e->NewStringUTF("ERR SIZE");
    if(!mmSizeAllowed((uint32_t)m)||!mmSizeAllowed((uint32_t)k)||!mmSizeAllowed((uint32_t)n))
        return e->NewStringUTF((std::string("ERR SIZE_UNSUPPORTED allowed=1..")
                + std::to_string((unsigned)MM_BUCKET_MAX)).c_str());
    return e->NewStringUTF(runMatMul((uint32_t)m,(uint32_t)k,(uint32_t)n,false).c_str());
}
extern "C" JNIEXPORT jstring JNICALL Java_bslsjdk_mcnpu_NpuRuntime_nativeMatMulFp16(JNIEnv* e,jclass,jint m,jint k,jint n){
    if(m<=0||k<=0||n<=0) return e->NewStringUTF("ERR SIZE");
    if(!mmSizeAllowed((uint32_t)m)||!mmSizeAllowed((uint32_t)k)||!mmSizeAllowed((uint32_t)n))
        return e->NewStringUTF((std::string("ERR SIZE_UNSUPPORTED allowed=1..")
                + std::to_string((unsigned)MM_BUCKET_MAX)).c_str());
    return e->NewStringUTF(runMatMul((uint32_t)m,(uint32_t)k,(uint32_t)n,true).c_str());
}
extern "C" JNIEXPORT jstring JNICALL Java_bslsjdk_mcnpu_NpuRuntime_nativeMatMulInt8(JNIEnv* e,jclass,jint m,jint k,jint n){
    if(m<=0||k<=0||n<=0) return e->NewStringUTF("ERR SIZE");
    if(!mmSizeAllowed((uint32_t)m)||!mmSizeAllowed((uint32_t)k)||!mmSizeAllowed((uint32_t)n))
        return e->NewStringUTF((std::string("ERR SIZE_UNSUPPORTED allowed=1..")
                + std::to_string((unsigned)MM_BUCKET_MAX)).c_str());
    return e->NewStringUTF(runMatMulInt8((uint32_t)m,(uint32_t)k,(uint32_t)n).c_str());
}
extern "C" JNIEXPORT jstring JNICALL Java_bslsjdk_mcnpu_NpuRuntime_nativeGetLastError(JNIEnv* e,jclass){
    return e->NewStringUTF(g_lastNativeError.c_str());
}

extern "C" JNIEXPORT jbyteArray JNICALL Java_bslsjdk_mcnpu_NpuRuntime_nativeMatMulInt8Buf(JNIEnv* e,jclass,jbyteArray ja,jbyteArray jb,jint m,jint k,jint n){
    // Every rejection below must set g_lastNativeError. These paths used to return
    // nullptr silently, so the service replied "ERR BIN_SUBMIT_FAILED " with an empty
    // reason and Minecraft could only print "see logcat". That is why 47 consecutive
    // failures in one session produced no usable diagnosis.
    if(!ja||!jb||m<=0||k<=0||n<=0){ g_lastNativeError="ERR BAD_ARGS null=" + std::to_string(!ja||!jb)
        + " m="+std::to_string(m)+" k="+std::to_string(k)+" n="+std::to_string(n); return nullptr; }
    if(!mmSizeAllowed((uint32_t)m)||!mmSizeAllowed((uint32_t)k)||!mmSizeAllowed((uint32_t)n)){
        g_lastNativeError="ERR SIZE_UNSUPPORTED allowed=1.."+std::to_string((unsigned)MM_BUCKET_MAX)
            +" got m="+std::to_string(m)+" k="+std::to_string(k)+" n="+std::to_string(n);
        return nullptr; }
    const jsize alen=e->GetArrayLength(ja), blen=e->GetArrayLength(jb);
    if(alen!=(jsize)((size_t)m*k) || blen!=(jsize)((size_t)k*n)){
        g_lastNativeError="ERR BIN_SIZE expect alen="+std::to_string((long long)m*(long long)k)
            +" blen="+std::to_string((long long)k*(long long)n)
            +" got alen="+std::to_string((int)alen)+" blen="+std::to_string((int)blen);
        return nullptr; }
    std::vector<int8_t> A((size_t)alen), B((size_t)blen), C((size_t)m*n,0);
    e->GetByteArrayRegion(ja,0,alen,(jbyte*)A.data());
    e->GetByteArrayRegion(jb,0,blen,(jbyte*)B.data());
    float scaleC=0.f;
    std::string r=runMatMulInt8Buf(A.data(),B.data(),C.data(),(uint32_t)m,(uint32_t)k,(uint32_t)n,scaleC);
    if(r.rfind("OK",0)!=0){ g_lastNativeError=r; E("MATMUL8BUF FAIL %s",r.c_str()); return nullptr; }
    const jsize total=(jsize)(4+C.size());
    std::vector<jbyte> tmp((size_t)total);
    std::memcpy(tmp.data(),&scaleC,4);
    std::memcpy(tmp.data()+4,C.data(),C.size());
    jbyteArray out=e->NewByteArray(total);
    if(!out){ g_lastNativeError="ERR OOM allocating result byte["+std::to_string((int)total)+"]"; return nullptr; }
    e->SetByteArrayRegion(out,0,total,tmp.data());
    return out;
}
extern "C" JNIEXPORT jstring JNICALL Java_bslsjdk_mcnpu_NpuRuntime_nativeXform(JNIEnv* e,jclass,jint op,jint n){
    if(n<=0) return e->NewStringUTF("ERR SIZE");
    if(n>65536) return e->NewStringUTF("ERR SIZE_UNSUPPORTED max=65536");
    return e->NewStringUTF(runBatchXform((uint32_t)n,(int)op).c_str());
}
extern "C" JNIEXPORT void JNICALL Java_bslsjdk_mcnpu_NpuRuntime_nativeShutdown(JNIEnv*,jclass){shutdownRuntime();}
