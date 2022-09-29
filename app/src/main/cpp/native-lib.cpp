#include <jni.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <string>
#include <vector>

#include "tensorflow/lite/model.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/stderr_reporter.h"
#include "util_logging.h"
//#include "util_tflite.h"


#ifdef __cplusplus
extern "C" {
#endif
// 部署头文件
#include "tflite_classification.h"


#define testimage  "0901151828-hh-up-0-[tap]-imgenv.csv"

static AAssetManager *s_nativeasset;

bool
asset_read_file_float(AAssetManager *assetMgr, char *fname, std::vector<float> &buf);


JNIEXPORT void JNICALL
Java_com_example_gesture_1classification_MainActivity_setNativeAssetManager(
        JNIEnv * env ,
        jobject instance,
        jobject assetManager ) {

    AAssetManager *nativeasset = AAssetManager_fromJava(env, assetManager);
    //the use of nativeasset
    s_nativeasset = nativeasset;
}

JNIEXPORT jstring JNICALL
Java_com_example_gesture_1classification_MainActivity_stringFromJNI(
        JNIEnv * env,
        jobject instance) {

    std::string hello = "Hello from C++";

//    //1. 获取随机数据
//    std::vector<float> image;
//    for(int i = 0; i < 200*200; i++){
//        image.push_back(rand() / double(RAND_MAX));
//    }
    //1. 读取csv数据测试
    std::vector<float> image;
    asset_read_file_float(s_nativeasset, (char *) testimage, image);
    DBG_LOGE("=========>>>>testimage: %s", testimage);


    //0. 初始化模型文件
    int ret = tflite_init_interface(s_nativeasset);
    DBG_LOGI("native tflite_init_interface ret: %d", ret);

#if 1 /* for debug */
    // 可视化输入数据前10个
    float *pData = (float *) image.data();
    for (int i = 0; i < 5; i++) {
        DBG_LOGI("native b: %f", *pData++);
    }
#endif
    //2. 输入数据到模型文件
    tflite_feedData_interface((float *) image.data());
    DBG_LOGI("native feed data success.");

    //3. 获取模型输出结果
    classification_result_t class_result = {0};
    tflite_invoke_interface(&class_result);

#if 1 /* for debug */
    //可视化输出结果
    for (int i = 0; i < class_result.num; i++) {
        DBG_LOGI("native class_result num: %d: id: %d, score: %f, name: %s",
                 class_result.num,
                 class_result.classify[i].id,
                 class_result.classify[i].score,
                 class_result.classify[i].name);
    }
#endif

    DBG_LOGI("native output success.");
    return env->NewStringUTF(hello.c_str());
}


std::vector <std::string>
_string_split(std::string str, std::string pattern) {
    std::string::size_type pos;
    std::vector <std::string> result;
    str += pattern;//扩展字符串以方便操作
    int size = str.size();
    for (int i = 0; i < size; i++) {
        pos = str.find(pattern, i);
        if (pos < size) {
            std::string s = str.substr(i, pos - i);
            result.push_back(s);
            i = pos + pattern.size() - 1;
        }
    }
    return result;
}

inline float
_str2float(const std::string &str) {
    return stof(str);
}

bool
asset_read_file_float(AAssetManager *assetMgr, char *fname, std::vector<float> &buf) {
    AAsset *descriptor = AAssetManager_open(assetMgr, fname, AASSET_MODE_BUFFER);
    if (descriptor == NULL) {
        return false;
    }
    size_t fileLength = AAsset_getLength(descriptor);

    std::vector <uint8_t> vec_uint8;
    vec_uint8.resize(fileLength);
    int64_t readSize = AAsset_read(descriptor, vec_uint8.data(), fileLength);
    if (readSize != fileLength) {
        DBG_LOGE("asset_read_file_float readSize!=fileLength");
        return false;
    }
    std::vector <uint8_t> _data = {};
    _data.swap(vec_uint8);

    std::string str_data;
//    for(std::vector<uint8_t>::iterator iter = vec_uint8.begin(); iter!= vec_uint8.end(); ++iter){
//        if(*iter != NULL)
//            str_data += *iter;
//    }
    str_data.assign(_data.begin(), _data.end());
    std::vector <std::string> str_data_split_line = _string_split(str_data, "\n");
    std::vector <std::string> str_data_split = {};
    for (int i = 0; i < str_data_split_line.size() - 1; i++) {
        std::vector <std::string> temp = _string_split(str_data_split_line[i], ",");
        str_data_split.insert(str_data_split.end(), temp.begin(), temp.end());
    }
//    std::vector<float> data;
    transform(str_data_split.begin(), str_data_split.end(), back_inserter(buf), _str2float);

//    buf.insert(buf.end(), data.begin(), data.end());

    AAsset_close(descriptor);

    return buf.size();
}

#ifdef __cplusplus
}
#endif