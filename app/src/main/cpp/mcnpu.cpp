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
#include <algorithm>
#include <unordered_map>
#include <cstdint>
#include <sched.h>
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
    std::unordered_map<uint32_t, AddGraph> addGraphs;
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

// Any reasonable shape is allowed (real workloads are not nice powers of two,
// e.g. 128 entities x 8 features x 16 outputs). Graphs are cached per shape.
static bool mmSizeAllowed(uint32_t v){ return v>=1 && v<=65536; }

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

static bool addSizeAllowed(uint32_t n){
    static const uint32_t allow[]={16,64,256,1024,4096,16384};
    for(uint32_t v:allow) if(v==n) return true;
    return false;
}

Qnn_Tensor_t makeTensor(const char* name,Qnn_TensorType_t type,Qnn_DataType_t dt,uint32_t* dims){
    Qnn_Tensor_t t=QNN_TENSOR_INIT;
    t.version=QNN_TENSOR_VERSION_1;
    t.v1.name=name;t.v1.type=type;t.v1.dataFormat=QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    t.v1.dataType=dt;t.v1.rank=1;t.v1.dimensions=dims;t.v1.memType=QNN_TENSORMEMTYPE_RAW;
    return t;
}

std::string runAdd(const float* av,const float* bv,uint32_t n){
    std::lock_guard<std::mutex> lock(gRuntimeMutex);
    const auto total0=std::chrono::steady_clock::now();
    if(!g.ready || !g.api || !g.context) return "ERR NPU_NOT_READY";
    const auto& f=g.api->QNN_INTERFACE_VER_NAME;

    Runtime::AddGraph* ag=nullptr;
    bool cached=false;
    auto found=g.addGraphs.find(n);
    if(found!=g.addGraphs.end()){
        ag=&found->second;
        cached=true;
    }

    Qnn_ErrorHandle_t rc=QNN_SUCCESS;
    long long createUs=0, finalizeUs=0;

    if(!cached){
        // Insert the cache entry BEFORE creating graph tensors so every tensor
        // descriptor points at dimensions owned by the final cached object.
        if(!ensureGraphBudget()) return "ERR GRAPH_BUDGET_EXHAUSTED";
        g.graphCount++;
        auto inserted=g.addGraphs.emplace(n, Runtime::AddGraph{});
        ag=&inserted.first->second;
        ag->dims[0]=n;

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

        ag->a=makeTensor("a",QNN_TENSOR_TYPE_APP_WRITE,QNN_DATATYPE_FLOAT_32,ag->dims);
        ag->b=makeTensor("b",QNN_TENSOR_TYPE_APP_WRITE,QNN_DATATYPE_FLOAT_32,ag->dims);
        ag->c=makeTensor("c",QNN_TENSOR_TYPE_APP_READ,QNN_DATATYPE_FLOAT_32,ag->dims);

        rc=f.tensorCreateGraphTensor(ag->graph,&ag->a);
        if(rc==QNN_SUCCESS) rc=f.tensorCreateGraphTensor(ag->graph,&ag->b);
        if(rc==QNN_SUCCESS) rc=f.tensorCreateGraphTensor(ag->graph,&ag->c);
        if(rc!=QNN_SUCCESS){
            // QNN 2.27 exposes no graphFree in QnnInterface. The graph is
            // owned by the context and is released by contextFree.
            g.addGraphs.erase(n);
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
            g.addGraphs.erase(n);
            return "ERR GRAPH_NODE rc="+std::to_string((int)rc)+" "+verbose(rc);
        }

        auto tFinalize0=std::chrono::steady_clock::now();
        rc=f.graphFinalize(ag->graph,nullptr,nullptr);
        finalizeUs=std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now()-tFinalize0).count();
        if(rc!=QNN_SUCCESS){
            g.addGraphs.erase(n);
            return "ERR GRAPH_FINALIZE rc="+std::to_string((int)rc)+
                   " create_us="+std::to_string(createUs)+
                   " finalize_us="+std::to_string(finalizeUs)+" "+verbose(rc);
        }

        I("ADD GRAPH READY n=%u a_id=%u b_id=%u c_id=%u",
          (unsigned)n,(unsigned)ag->a.v1.id,(unsigned)ag->b.v1.id,(unsigned)ag->c.v1.id);
    }

    std::vector<float> out(n,-999.f);

    // QNN graphExecute requires the same tensor IDs assigned during
    // tensorCreateGraphTensor(). Reuse the registered descriptors and only
    // replace their client buffers for each execution.
    Qnn_Tensor_t ea=ag->a, eb=ag->b, ec=ag->c;
    ea.v1.clientBuf.data=(void*)av;
    ea.v1.clientBuf.dataSize=n*sizeof(float);
    eb.v1.clientBuf.data=(void*)bv;
    eb.v1.clientBuf.dataSize=n*sizeof(float);
    ec.v1.clientBuf.data=out.data();
    ec.v1.clientBuf.dataSize=n*sizeof(float);
    Qnn_Tensor_t execIn[2]={ea,eb};
    Qnn_Tensor_t execOut[1]={ec};

    auto t0=std::chrono::steady_clock::now();
    rc=f.graphExecute(ag->graph,execIn,2,execOut,1,nullptr,nullptr);
    auto us=std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now()-t0).count();
    if(rc!=QNN_SUCCESS)
        return "ERR GRAPH_EXECUTE rc="+std::to_string((int)rc)+" "+verbose(rc);

    for(uint32_t i=0;i<n;i++)
        if(out[i] != av[i]+bv[i]) return "ERR OUTPUT_VERIFY";

    auto totalUs=std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now()-total0).count();
    char buf[512];
    std::snprintf(buf,sizeof(buf),
        "OK HTP_GRAPH_EXECUTE graph_cached=%s n=%u create_us=%lld finalize_us=%lld execute_us=%lld total_us=%lld out0=%g out_last=%g",
        cached?"true":"false",(unsigned)n,createUs,finalizeUs,(long long)us,
        (long long)totalUs,(double)out[0],(double)out[n-1]);
    return buf;
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
    if(rc!=QNN_SUCCESS) return "ERR MM_EXECUTE rc="+std::to_string((int)rc)+" "+verbose(rc);

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
    if(Nb==0) return "ERR XFORM_TOO_LARGE (max 65536)";
    const float sc=1.0f/127.0f;
    const uint64_t key=(2ULL<<60)|((uint64_t)(op&0xff)<<52)|(uint64_t)Nb;
    Runtime::MatMulGraph* mg=nullptr;
    auto found=g.matMulGraphs8.find(key);
    if(found!=g.matMulGraphs8.end()) mg=&found->second;
    Qnn_ErrorHandle_t rc=QNN_SUCCESS;
    if(!mg){
        if(!ensureGraphBudget()) return "ERR GRAPH_BUDGET_EXHAUSTED";
        g.graphCount++;
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
    if(rc!=QNN_SUCCESS) return "ERR XF_EXECUTE rc="+std::to_string((int)rc);
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
        g.graphCount++;
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
    if(rc!=QNN_SUCCESS) return "ERR MM8_EXECUTE rc="+std::to_string((int)rc)+" "+verbose(rc);

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
static const uint32_t MM_BUCKET_MAX = 65536;

static uint32_t bucketize(uint32_t v){
    for(size_t i=0;i<sizeof(MM_BUCKETS)/sizeof(MM_BUCKETS[0]);i++) if(v<=MM_BUCKETS[i]) return MM_BUCKETS[i];
    return 0;
}

// Exposed over IPC so the client can discover the real limits instead of
// hard-coding them. Previously the client guessed and silently produced
// shapes above MM_BUCKET_MAX, which always came back as ERR BUF_TOO_LARGE.
static std::string mmLimits(){
    std::string s="OK MM_LIMITS max="+std::to_string((unsigned)MM_BUCKET_MAX)+" buckets=";
    for(size_t i=0;i<sizeof(MM_BUCKETS)/sizeof(MM_BUCKETS[0]);i++){
        if(i) s+=",";
        s+=std::to_string((unsigned)MM_BUCKETS[i]);
    }
    return s;
}

std::string runMatMulInt8Buf(const int8_t* Ain,const int8_t* Bin,int8_t* Cout,uint32_t m,uint32_t k,uint32_t n,float& scaleCOut){
    std::lock_guard<std::mutex> lock(gRuntimeMutex);
    if(!g.ready || !g.api || !g.context) return "ERR NPU_NOT_READY";
    const auto& f=g.api->QNN_INTERFACE_VER_NAME;
    const uint32_t Mb=bucketize(m), Kb=bucketize(k), Nb=bucketize(n);
    if(Mb==0||Kb==0||Nb==0) return "ERR BUF_TOO_LARGE (max 65536)";
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
    auto found=g.matMulGraphs8.find(key);
    if(found!=g.matMulGraphs8.end()) mg=&found->second;
    Qnn_ErrorHandle_t rc=QNN_SUCCESS;
    if(!mg){
        if(!ensureGraphBudget()) return "ERR GRAPH_BUDGET_EXHAUSTED";
        g.graphCount++;
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
        I("MATMUL8BUF GRAPH READY bucket=%ux%ux%u scaleC=%.5f",(unsigned)Mb,(unsigned)Kb,(unsigned)Nb,scaleC);
    }

    std::vector<int8_t> Cpad((size_t)Mb*Nb, 0);
    Qnn_Tensor_t ea=mg->a, eb=mg->b, ec=mg->c;
    ea.v1.clientBuf.data=(void*)Ause; ea.v1.clientBuf.dataSize=(uint32_t)((size_t)Mb*Kb);
    eb.v1.clientBuf.data=(void*)Buse; eb.v1.clientBuf.dataSize=(uint32_t)((size_t)Kb*Nb);
    ec.v1.clientBuf.data=Cpad.data(); ec.v1.clientBuf.dataSize=(uint32_t)((size_t)Mb*Nb);
    Qnn_Tensor_t execIn[2]={ea,eb};
    Qnn_Tensor_t execOut[1]={ec};
    rc=f.graphExecute(mg->graph,execIn,2,execOut,1,nullptr,nullptr);
    if(rc!=QNN_SUCCESS) return "ERR BUF_EXECUTE rc="+std::to_string((int)rc);
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
extern "C" JNIEXPORT jstring JNICALL Java_bslsjdk_mcnpu_NpuRuntime_nativeAdd(JNIEnv* e,jclass,jfloatArray ja,jfloatArray jb){
    if(!ja||!jb)return e->NewStringUTF("ERR NULL");
    jsize n=e->GetArrayLength(ja);
    if(n<=0||n!=e->GetArrayLength(jb))return e->NewStringUTF("ERR SIZE");
    if(!addSizeAllowed((uint32_t)n))return e->NewStringUTF("ERR SIZE_UNSUPPORTED allowed=16,64,256,1024,4096,16384");
    std::vector<float>a(n),b(n);
    e->GetFloatArrayRegion(ja,0,n,a.data());
    e->GetFloatArrayRegion(jb,0,n,b.data());
    std::string r=runAdd(a.data(),b.data(),(uint32_t)n);
    return e->NewStringUTF(r.c_str());
}
extern "C" JNIEXPORT jstring JNICALL Java_bslsjdk_mcnpu_NpuRuntime_nativeMatMul(JNIEnv* e,jclass,jint m,jint k,jint n){
    if(m<=0||k<=0||n<=0) return e->NewStringUTF("ERR SIZE");
    if(!mmSizeAllowed((uint32_t)m)||!mmSizeAllowed((uint32_t)k)||!mmSizeAllowed((uint32_t)n))
        return e->NewStringUTF("ERR SIZE_UNSUPPORTED allowed=16,32,64,128,256,512");
    return e->NewStringUTF(runMatMul((uint32_t)m,(uint32_t)k,(uint32_t)n,false).c_str());
}
extern "C" JNIEXPORT jstring JNICALL Java_bslsjdk_mcnpu_NpuRuntime_nativeMatMulFp16(JNIEnv* e,jclass,jint m,jint k,jint n){
    if(m<=0||k<=0||n<=0) return e->NewStringUTF("ERR SIZE");
    if(!mmSizeAllowed((uint32_t)m)||!mmSizeAllowed((uint32_t)k)||!mmSizeAllowed((uint32_t)n))
        return e->NewStringUTF("ERR SIZE_UNSUPPORTED allowed=16,32,64,128,256,512");
    return e->NewStringUTF(runMatMul((uint32_t)m,(uint32_t)k,(uint32_t)n,true).c_str());
}
extern "C" JNIEXPORT jstring JNICALL Java_bslsjdk_mcnpu_NpuRuntime_nativeMatMulInt8(JNIEnv* e,jclass,jint m,jint k,jint n){
    if(m<=0||k<=0||n<=0) return e->NewStringUTF("ERR SIZE");
    if(!mmSizeAllowed((uint32_t)m)||!mmSizeAllowed((uint32_t)k)||!mmSizeAllowed((uint32_t)n))
        return e->NewStringUTF("ERR SIZE_UNSUPPORTED allowed=16,32,64,128,256,512");
    return e->NewStringUTF(runMatMulInt8((uint32_t)m,(uint32_t)k,(uint32_t)n).c_str());
}
extern "C" JNIEXPORT jbyteArray JNICALL Java_bslsjdk_mcnpu_NpuRuntime_nativeMatMulInt8Buf(JNIEnv* e,jclass,jbyteArray ja,jbyteArray jb,jint m,jint k,jint n){
    if(!ja||!jb||m<=0||k<=0||n<=0) return nullptr;
    if(!mmSizeAllowed((uint32_t)m)||!mmSizeAllowed((uint32_t)k)||!mmSizeAllowed((uint32_t)n)) return nullptr;
    const jsize alen=e->GetArrayLength(ja), blen=e->GetArrayLength(jb);
    if(alen!=(jsize)((size_t)m*k) || blen!=(jsize)((size_t)k*n)) return nullptr;
    std::vector<int8_t> A((size_t)alen), B((size_t)blen), C((size_t)m*n,0);
    e->GetByteArrayRegion(ja,0,alen,(jbyte*)A.data());
    e->GetByteArrayRegion(jb,0,blen,(jbyte*)B.data());
    float scaleC=0.f;
    std::string r=runMatMulInt8Buf(A.data(),B.data(),C.data(),(uint32_t)m,(uint32_t)k,(uint32_t)n,scaleC);
    if(r.rfind("OK",0)!=0){ E("MATMUL8BUF FAIL %s",r.c_str()); return nullptr; }
    const jsize total=(jsize)(4+C.size());
    std::vector<jbyte> tmp((size_t)total);
    std::memcpy(tmp.data(),&scaleC,4);
    std::memcpy(tmp.data()+4,C.data(),C.size());
    jbyteArray out=e->NewByteArray(total);
    if(!out) return nullptr;
    e->SetByteArrayRegion(out,0,total,tmp.data());
    return out;
}
extern "C" JNIEXPORT jstring JNICALL Java_bslsjdk_mcnpu_NpuRuntime_nativeXform(JNIEnv* e,jclass,jint op,jint n){
    if(n<=0) return e->NewStringUTF("ERR SIZE");
    if(n>65536) return e->NewStringUTF("ERR SIZE_UNSUPPORTED max=65536");
    return e->NewStringUTF(runBatchXform((uint32_t)n,(int)op).c_str());
}
extern "C" JNIEXPORT void JNICALL Java_bslsjdk_mcnpu_NpuRuntime_nativeShutdown(JNIEnv*,jclass){shutdownRuntime();}
