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
#include <mutex>
#include <algorithm>
#include <unordered_map>
#include <cstdint>
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
#define I(...) __android_log_print(ANDROID_LOG_INFO,TAG,"%s",__VA_ARGS__)
#define E(...) __android_log_print(ANDROID_LOG_ERROR,TAG,__VA_ARGS__)

namespace {
struct Runtime {
    uint64_t diagCount=0;
    uint64_t graphSeq=0;
    struct AddGraph {
        Qnn_GraphHandle_t graph=nullptr;
        uint32_t dims[1]={0};
        Qnn_Tensor_t a=QNN_TENSOR_INIT;
        Qnn_Tensor_t b=QNN_TENSOR_INIT;
        Qnn_Tensor_t c=QNN_TENSOR_INIT;
    };
    std::unordered_map<uint32_t, AddGraph> addGraphs;
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
            dlerror();
            void* h=dlopen(p,RTLD_NOW|RTLD_LOCAL);
            const char* e=dlerror();
            out += std::string("  dlopen=")+(h?"OK":"FAIL")+" err="+(e?e:"<none>")+"\\n";
            if(h) dlclose(h);
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
        g.err="qnnDir invalid: "+g.libDir+" errno="+std::to_string(errno)+"("+std::string(strerror(errno))+")";
        return false;
    }

    std::string adsp=g.libDir+";/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp";
    setenv("ADSP_LIBRARY_PATH",adsp.c_str(),1);
    setenv("LD_LIBRARY_PATH",(g.libDir+":/vendor/dsp/cdsp:/vendor/lib64/").c_str(),1);
    if(workDir.empty() || chdir(workDir.c_str())!=0){
        g.err="chdir workDir failed errno="+std::to_string(errno)+"("+std::string(strerror(errno))+")";
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
    g.info="QNN HTP ready backendId=6 providers="+std::to_string(count);
    g.ready=true;
    I("MCNPU HTP READY");
    return true;
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
        Runtime::AddGraph fresh;
        fresh.dims[0]=n;
        const std::string graphName = "mcnpu_add_" + std::to_string(++g.graphSeq);

        auto tCreate0=std::chrono::steady_clock::now();
        rc=f.graphCreate(g.context,graphName.c_str(),nullptr,&fresh.graph);
        createUs=std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now()-tCreate0).count();
        if(rc!=QNN_SUCCESS || !fresh.graph)
            return "ERR GRAPH_CREATE rc="+std::to_string((int)rc)+
                   " create_us="+std::to_string(createUs)+" "+verbose(rc);

        fresh.a=makeTensor("a",QNN_TENSOR_TYPE_APP_WRITE,QNN_DATATYPE_FLOAT_32,fresh.dims);
        fresh.b=makeTensor("b",QNN_TENSOR_TYPE_APP_WRITE,QNN_DATATYPE_FLOAT_32,fresh.dims);
        fresh.c=makeTensor("c",QNN_TENSOR_TYPE_APP_READ,QNN_DATATYPE_FLOAT_32,fresh.dims);

        rc=f.tensorCreateGraphTensor(fresh.graph,&fresh.a);
        if(rc==QNN_SUCCESS) rc=f.tensorCreateGraphTensor(fresh.graph,&fresh.b);
        if(rc==QNN_SUCCESS) rc=f.tensorCreateGraphTensor(fresh.graph,&fresh.c);
        if(rc!=QNN_SUCCESS)
            return "ERR TENSOR_CREATE rc="+std::to_string((int)rc)+" "+verbose(rc);

        Qnn_Scalar_t scalar=QNN_SCALAR_INIT;
        scalar.dataType=QNN_DATATYPE_UINT_32;
        scalar.uint32Value=QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD;
        Qnn_Param_t param=QNN_PARAM_INIT;
        param.paramType=QNN_PARAMTYPE_SCALAR;
        param.name=QNN_OP_ELEMENT_WISE_BINARY_PARAM_OPERATION;
        param.scalarParam=scalar;
        Qnn_Tensor_t ins[2]={fresh.a,fresh.b};
        Qnn_OpConfig_t op=QNN_OPCONFIG_INIT;
        op.v1.name="add";
        op.v1.packageName="qti.aisw";
        op.v1.typeName=QNN_OP_ELEMENT_WISE_BINARY;
        op.v1.numOfParams=1;
        op.v1.params=&param;
        op.v1.numOfInputs=2;
        op.v1.inputTensors=ins;
        op.v1.numOfOutputs=1;
        op.v1.outputTensors=&fresh.c;

        rc=f.graphAddNode(fresh.graph,op);
        if(rc!=QNN_SUCCESS)
            return "ERR GRAPH_NODE rc="+std::to_string((int)rc)+" "+verbose(rc);

        auto tFinalize0=std::chrono::steady_clock::now();
        rc=f.graphFinalize(fresh.graph,nullptr,nullptr);
        finalizeUs=std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now()-tFinalize0).count();
        if(rc!=QNN_SUCCESS)
            return "ERR GRAPH_FINALIZE rc="+std::to_string((int)rc)+
                   " create_us="+std::to_string(createUs)+
                   " finalize_us="+std::to_string(finalizeUs)+" "+verbose(rc);

        auto inserted=g.addGraphs.emplace(n,std::move(fresh));
        ag=&inserted.first->second;

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

void shutdownRuntime(){
    std::lock_guard<std::mutex> lock(gRuntimeMutex);
    if(!g.api)return;
    const auto& f=g.api->QNN_INTERFACE_VER_NAME;
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
    if(n<=0||n!=e->GetArrayLength(jb)||n>1024)return e->NewStringUTF("ERR SIZE");
    std::vector<float>a(n),b(n);
    e->GetFloatArrayRegion(ja,0,n,a.data());
    e->GetFloatArrayRegion(jb,0,n,b.data());
    std::string r=runAdd(a.data(),b.data(),(uint32_t)n);
    return e->NewStringUTF(r.c_str());
}
extern "C" JNIEXPORT void JNICALL Java_bslsjdk_mcnpu_NpuRuntime_nativeShutdown(JNIEnv*,jclass){shutdownRuntime();}
