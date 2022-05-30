/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 1430337978@qq.com
 * ------------------------------------------------ */
#ifndef TFLITE_CLASSIFICATION_H_
#define TFLITE_CLASSIFICATION_H_

#ifdef __cplusplus
extern "C" {
#endif

#define CLASSIFY_MODEL_PATH        "classification_model/mobilenet_v3.tflite"
#define CLASSIFY_LABEL_MAP_PATH    "classification_model/gestrue_class_label.txt"
#define MAX_CLASS_NUM  7

typedef struct _classify_t
{
    int     id;
    float   score;
    char    name[64];
} classify_t;

typedef struct _classification_result_t
{
    int num;
    classify_t classify[MAX_CLASS_NUM];
} classification_result_t;



//void * get_classification_input_buf (int *w, int *h);
//int  tflite_getInputType_interface ();

bool tflite_feedData_interface(float *pBuffer);
bool tflite_init_interface(AAssetManager *assetMgr);
int  tflite_invoke_interface(classification_result_t *class_result);
    
#ifdef __cplusplus
}
#endif

#endif /* TFLITE_CLASSIFICATION_H_ */
