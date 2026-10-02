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
#define E(...) __android_log_print(ANDROID_LOG_ERROR,TAG,"%s",__VA_ARGS__)

namespace {
struct Runtime {
    void* qnn=nullptr;
    const QnnInterface_t* api=nullptr;
    Qnn_BackendHandle_t backend=nullptr;
    Qnn_DeviceHandle_t device=nullptr;
    Qnn_ContextHandle_t context=nullptr;
    std::vector<void*> rpc;
    bool ready=false;
    std::string info;
    std::string err;
    Qnn_LogHandle_t logger=nullptr;
} g;

using GetProviders = Qnn_ErrorHandle_t (*)(const QnnInterface_t ***,uint32_t *);

bool loadRuntime() {
    Dl_info di{};
    if(!dladdr((void*)&loadRuntime,&di)||!di.dli_fname){g.err="dladdr failed";return false;}
    std::string dir=di.dli_fname;
    size_t slash=dir.find_last_of('/');
    if(slash==std::string::npos){g.err="library directory missing";return false;}
    dir.resize(slash);

    std::string adsp=dir+";/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp";
    setenv("ADSP_LIBRARY_PATH",adsp.c_str(),1);
    setenv("LD_LIBRARY_PATH",(dir+":/vendor/dsp/cdsp:/vendor/lib64/").c_str(),1);
    if(chdir(dir.c_str())!=0){g.err="chdir failed errno="+std::to_string(errno);return false;}

    // 与已在同一台设备成功跑通 HTP 的 npu_probe 完全对齐：
    // 先加载 QNN HTP，再让 QNN/HTP 自己解析 FastRPC；不要预先 dlopen FastRPC。
    g.qnn=dlopen((dir+"/libQnnHtp.so").c_str(),RTLD_NOW|RTLD_GLOBAL);
    if(!g.qnn){
        const char* x=dlerror();
        g.err=std::string("QNN load failed: ")+(x?x:"?");
        return false;
    }
    return true;
}

bool initRuntime(){
    if(g.ready)return true;
    if(!loadRuntime())return false;
    auto gp=(GetProviders)dlsym(g.qnn,"QnnInterface_getProviders");
    if(!gp){g.err="QnnInterface_getProviders missing";return false;}
    const QnnInterface_t** providers=nullptr; uint32_t count=0;
    Qnn_ErrorHandle_t rc=gp(&providers,&count);
    if(rc!=QNN_SUCCESS||!providers||count==0){g.err="getProviders rc="+std::to_string((int)rc);return false;}
    for(uint32_t i=0;i<count;i++) if(providers[i]&&providers[i]->backendId==HTP_ID){g.api=providers[i];break;}
    if(!g.api){g.err="HTP provider backendId=6 not found";return false;}
    const auto& f=g.api->QNN_INTERFACE_VER_NAME;
    if(f.logCreate){
        rc=f.logCreate(nullptr,QNN_LOG_LEVEL_INFO,&g.logger);
        if(rc!=QNN_SUCCESS) g.logger=nullptr;
    }
    rc=f.backendCreate(g.logger,nullptr,&g.backend);
    if(rc!=QNN_SUCCESS||!g.backend){
        const char* msg=nullptr;
        if(f.errorGetVerboseMessage) f.errorGetVerboseMessage(rc,&msg);
        g.err="backendCreate rc="+std::to_string((int)rc)+(msg?(" msg="+std::string(msg)):"");
        return false;
    }
    rc=f.deviceCreate(g.logger,nullptr,&g.device);
    if(rc!=QNN_SUCCESS||!g.device){
        const char* msg=nullptr;
        if(f.errorGetVerboseMessage) f.errorGetVerboseMessage(rc,&msg);
        g.err="deviceCreate rc="+std::to_string((int)rc)+(msg?(" msg="+std::string(msg)):"");
        return false;
    }
    rc=f.contextCreate(g.backend,g.device,nullptr,&g.context);
    if(rc!=QNN_SUCCESS||!g.context){
        const char* msg=nullptr;
        if(f.errorGetVerboseMessage) f.errorGetVerboseMessage(rc,&msg);
        g.err="contextCreate rc="+std::to_string((int)rc)+(msg?(" msg="+std::string(msg)):"");
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
    if(!g.ready)return "ERR NPU_NOT_READY";
    const auto& f=g.api->QNN_INTERFACE_VER_NAME;
    Qnn_GraphHandle_t graph=nullptr;
    if(f.graphCreate(g.context,"mcnpu_add",nullptr,&graph)!=QNN_SUCCESS)return "ERR GRAPH_CREATE";
    uint32_t dims[1]={n};
    Qnn_Tensor_t a=makeTensor("a",QNN_TENSOR_TYPE_APP_WRITE,QNN_DATATYPE_FLOAT_32,dims);
    Qnn_Tensor_t b=makeTensor("b",QNN_TENSOR_TYPE_APP_WRITE,QNN_DATATYPE_FLOAT_32,dims);
    Qnn_Tensor_t c=makeTensor("c",QNN_TENSOR_TYPE_APP_READ,QNN_DATATYPE_FLOAT_32,dims);
    Qnn_ErrorHandle_t rc=f.tensorCreateGraphTensor(graph,&a);
    if(rc==QNN_SUCCESS)rc=f.tensorCreateGraphTensor(graph,&b);
    if(rc==QNN_SUCCESS)rc=f.tensorCreateGraphTensor(graph,&c);
    if(rc!=QNN_SUCCESS)return "ERR TENSOR_CREATE rc="+std::to_string((int)rc);

    Qnn_Scalar_t scalar=QNN_SCALAR_INIT;
    scalar.dataType=QNN_DATATYPE_UINT_32;
    scalar.uint32Value=QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD;
    Qnn_Param_t param=QNN_PARAM_INIT;
    param.paramType=QNN_PARAMTYPE_SCALAR;
    param.name=QNN_OP_ELEMENT_WISE_BINARY_PARAM_OPERATION;
    param.scalarParam=scalar;
    Qnn_Tensor_t ins[2]={a,b};
    Qnn_OpConfig_t op=QNN_OPCONFIG_INIT;
    op.v1.name="add";op.v1.packageName="qti.aisw";op.v1.typeName=QNN_OP_ELEMENT_WISE_BINARY;
    op.v1.numOfParams=1;op.v1.params=&param;op.v1.numOfInputs=2;op.v1.inputTensors=ins;
    op.v1.numOfOutputs=1;op.v1.outputTensors=&c;
    rc=f.graphAddNode(graph,op);
    if(rc!=QNN_SUCCESS)return "ERR GRAPH_NODE rc="+std::to_string((int)rc);
    rc=f.graphFinalize(graph,nullptr,nullptr);
    if(rc!=QNN_SUCCESS)return "ERR GRAPH_FINALIZE rc="+std::to_string((int)rc);

    std::vector<float> out(n,-999.f);
    Qnn_Tensor_t ea=a,eb=b,ec=c;
    ea.v1.clientBuf.data=(void*)av;ea.v1.clientBuf.dataSize=n*sizeof(float);
    eb.v1.clientBuf.data=(void*)bv;eb.v1.clientBuf.dataSize=n*sizeof(float);
    ec.v1.clientBuf.data=out.data();ec.v1.clientBuf.dataSize=n*sizeof(float);
    Qnn_Tensor_t execIn[2]={ea,eb},execOut[1]={ec};
    auto t0=std::chrono::steady_clock::now();
    rc=f.graphExecute(graph,execIn,2,execOut,1,nullptr,nullptr);
    auto us=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-t0).count();
    if(rc!=QNN_SUCCESS)return "ERR GRAPH_EXECUTE rc="+std::to_string((int)rc);
    for(uint32_t i=0;i<n;i++)if(out[i] != av[i]+bv[i])return "ERR OUTPUT_VERIFY";
    char buf[256];std::snprintf(buf,sizeof(buf),"OK HTP graphExecute n=%u elapsed_us=%lld first=%g",(unsigned)n,(long long)us,(double)out[0]);
    return buf;
}

void shutdownRuntime(){
    if(!g.api)return;
    const auto& f=g.api->QNN_INTERFACE_VER_NAME;
    if(f.contextFree&&g.context)f.contextFree(g.context,nullptr);
    if(f.deviceFree&&g.device)f.deviceFree(g.device);
    if(f.backendFree&&g.backend)f.backendFree(g.backend);
    if(f.logFree&&g.logger)f.logFree(g.logger);
    g.context=nullptr;g.device=nullptr;g.backend=nullptr;g.logger=nullptr;g.ready=false;g.api=nullptr;
    if(g.qnn)dlclose(g.qnn);g.qnn=nullptr;
    for(void* h:g.rpc)if(h)dlclose(h);g.rpc.clear();
}
}

extern "C" JNIEXPORT void JNICALL Java_bslsjdk_mcnpu_NpuRuntime_nativeConfigure(JNIEnv* e,jclass,jstring s){
    if(!s)return;const char* p=e->GetStringUTFChars(s,nullptr);if(p){setenv("MCNPU_TUNING",p,1);e->ReleaseStringUTFChars(s,p);}
}
extern "C" JNIEXPORT jboolean JNICALL Java_bslsjdk_mcnpu_NpuRuntime_nativeInit(JNIEnv*,jclass){return initRuntime()?JNI_TRUE:JNI_FALSE;}
extern "C" JNIEXPORT jstring JNICALL Java_bslsjdk_mcnpu_NpuRuntime_nativeGetDeviceInfo(JNIEnv* e,jclass){return e->NewStringUTF((g.ready?g.info:g.err).c_str());}
extern "C" JNIEXPORT jboolean JNICALL Java_bslsjdk_mcnpu_NpuRuntime_nativeTest(JNIEnv*,jclass){
    float a[16],b[16];for(int i=0;i<16;i++){a[i]=(float)i;b[i]=2.f;}
    return runAdd(a,b,16).rfind("OK ",0)==0?JNI_TRUE:JNI_FALSE;
}
extern "C" JNIEXPORT jstring JNICALL Java_bslsjdk_mcnpu_NpuRuntime_nativeAdd(JNIEnv* e,jclass,jfloatArray ja,jfloatArray jb){
    if(!ja||!jb)return e->NewStringUTF("ERR NULL");
    jsize n=e->GetArrayLength(ja);
    if(n<=0||n!=e->GetArrayLength(jb)||n>1024)return e->NewStringUTF("ERR SIZE");
    std::vector<float>a(n),b(n);e->GetFloatArrayRegion(ja,0,n,a.data());e->GetFloatArrayRegion(jb,0,n,b.data());
    std::string r=runAdd(a.data(),b.data(),(uint32_t)n);return e->NewStringUTF(r.c_str());
}
extern "C" JNIEXPORT void JNICALL Java_bslsjdk_mcnpu_NpuRuntime_nativeShutdown(JNIEnv*,jclass){shutdownRuntime();}
