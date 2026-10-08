#include "ornith15_safetensors.h"
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <sstream>
#include <sys/stat.h>
#include <unistd.h>

namespace ornith15 {
namespace {
constexpr uint64_t kMaxHeaderBytes = 16ull * 1024ull * 1024ull;
constexpr uint64_t kMaxTensorCount = 20000;
bool u64(const char* p,uint64_t* o){uint64_t v=0;for(int i=0;i<8;i++)v|=uint64_t((unsigned char)p[i])<<(8*i);*o=v;return true;}
class J {
 const std::string&s; size_t p=0;
 void ws(){while(p<s.size()&&(s[p]==' '||s[p]=='\n'||s[p]=='\r'||s[p]=='\t'))p++;}
 bool c(char x){if(p<s.size()&&s[p]==x){p++;return true;}return false;}
 bool str(std::string&o){if(!c('"'))return false;o.clear();while(p<s.size()){char x=s[p++];if(x=='"')return true;if(x=='\\'){if(p>=s.size())return false;char e=s[p++];if(e=='"'||e=='\\'||e=='/'||e=='b'||e=='f'||e=='n'||e=='r'||e=='t')o.push_back(e=='n'?'\n':e=='r'?'\r':e=='t'?'\t':e=='b'?'\b':e=='f'?'\f':e);else return false;}else if((unsigned char)x<32)return false;else o.push_back(x);}return false;}
 bool nums(std::vector<int64_t>&a){if(!c('['))return false;ws();if(c(']'))return true;for(;;){ws();bool neg=c('-');if(p>=s.size()||s[p]<'0'||s[p]>'9')return false;uint64_t v=0;while(p<s.size()&&s[p]>='0'&&s[p]<='9'){uint64_t d=s[p++]-'0';if(v>(UINT64_MAX-d)/10)return false;v=v*10+d;}if(neg||v>INT64_MAX)return false;a.push_back((int64_t)v);ws();if(c(','))continue;return c(']');}}
 bool skip(){ws();if(p>=s.size())return false;if(s[p]=='"'){std::string x;return str(x);}if(s[p]=='['||s[p]=='{'){char o=s[p++],q=o=='['?']':'}';ws();if(c(q))return true;for(;;){if(o=='{' ){std::string k;if(!str(k)||!c(':'))return false;}if(!skip())return false;ws();if(c(','))continue;return c(q);}}if(s.compare(p,4,"true")==0){p+=4;return true;}if(s.compare(p,5,"false")==0){p+=5;return true;}if(s.compare(p,4,"null")==0){p+=4;return true;}size_t b=p;if(s[p]=='-'||s[p]=='+')p++;while(p<s.size()&&((s[p]>='0'&&s[p]<='9')||s[p]=='.'||s[p]=='e'||s[p]=='E'||s[p]=='+'||s[p]=='-'))p++;return p>b;}
 bool tensor(TensorView&t){if(!c('{'))return false;bool d=false,sh=false,off=false;for(;;){ws();if(c('}'))return d&&sh&&off;std::string k;if(!str(k)||!c(':'))return false;ws();if(k=="dtype"){if(!str(t.dtype))return false;d=true;}else if(k=="shape"){if(!nums(t.shape))return false;sh=true;}else if(k=="data_offsets"){std::vector<int64_t>a;if(!nums(a)||a.size()!=2||a[0]<0||a[1]<a[0])return false;t.data_begin=a[0];t.data_end=a[1];off=true;}else if(!skip())return false;ws();if(c(','))continue;return c('}')&&d&&sh&&off;}}
public:J(const std::string&x):s(x){}
 bool root(std::vector<TensorView>&v,std::string*e){ws();if(!c('{')){if(e)*e="root";return false;}for(;;){ws();if(c('}'))return true;std::string k;if(!str(k)||!c(':'))return false;ws();if(k=="__metadata__"){if(!skip())return false;}else{TensorView t;t.name=k;if(!tensor(t))return false;v.push_back(t);if(v.size()>kMaxTensorCount)return false;}ws();if(c(','))continue;return c('}');}}
};
}
SafetensorsReader::~SafetensorsReader(){Close();}
void SafetensorsReader::Close(){if(fd_>=0)::close(fd_);fd_=-1;file_bytes_=header_bytes_=data_begin_=0;tensors_.clear();}
bool SafetensorsReader::Open(const std::string&path,std::string*e){
 Close();fd_=::open(path.c_str(),O_RDONLY|O_CLOEXEC);if(fd_<0){if(e)*e=strerror(errno);return false;}
 struct stat st{};if(fstat(fd_,&st)||st.st_size<8){if(e)*e="bad file";Close();return false;}file_bytes_=st.st_size;
 char b[8];if(pread(fd_,b,8,0)!=8){if(e)*e="header read";Close();return false;}uint64_t h=0;u64(b,&h);
 if(!h||h>kMaxHeaderBytes||h>file_bytes_-8){if(e)*e="invalid header";Close();return false;}
 std::string j((size_t)h,'\0');if(pread(fd_,j.data(),j.size(),8)!=(ssize_t)j.size()){if(e)*e="short header";Close();return false;}
 header_bytes_=h;data_begin_=8+h;J parser(j);if(!parser.root(tensors_,e)){Close();return false;}
 for(auto&t:tensors_){uint64_t rb=t.data_begin,re=t.data_end;if(re<rb||re>file_bytes_-data_begin_){if(e)*e="tensor bounds: "+t.name;Close();return false;}t.data_begin=data_begin_+rb;t.data_end=data_begin_+re;}return true;
}
bool SafetensorsReader::FindTensor(const std::string&n,TensorView*o)const{if(!o)return false;for(const auto&t:tensors_)if(t.name==n){*o=t;return true;}return false;}
bool SafetensorsReader::Read(uint64_t off,void*d,size_t n,std::string*e)const{if(fd_<0||!d||off>file_bytes_||n>file_bytes_-off){if(e)*e="read bounds";return false;}size_t z=0;while(z<n){ssize_t r=pread(fd_,(char*)d+z,n-z,off+z);if(r<=0){if(e)*e=strerror(errno);return false;}z+=r;}return true;}
}
