/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#include "util_tflite.h"
#include "util_logging.h"
#include "tflite_classification.h"
#include <list>

#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>

static tflite_interpreter_t s_interpreter;
static tflite_tensor_t      s_tensor_input;
static tflite_tensor_t      s_tensor_output;

static char                 s_class_name [MAX_CLASS_NUM][64];

int  tflite_getInputType_interface ();
void * get_classification_input_buf (int *w, int *h);

/* -------------------------------------------------- *
 *  load class labels
 * -------------------------------------------------- */
static int
load_label_map (const char *label_buf, size_t label_size)
{
    std::string str_label (label_buf);
    std::string str_line;
    std::stringstream sstream {str_label};

    int id = 0;
    while (getline (sstream, str_line, '\n'))
    {
        memcpy (&s_class_name[id], str_line.c_str(), sizeof (s_class_name[id]));
//        LOG ("ID[%d] %s\n", id, s_class_name[id]);
        id ++;
    }
    return 0;
}



/* -------------------------------------------------- *
 *  Create TFLite Interpreter
 * -------------------------------------------------- */
int
init_tflite_classification(const char *model_buf, size_t model_size,
                           const char *label_buf, size_t label_size)
{
    tflite_create_interpreter (&s_interpreter, model_buf, model_size);
    tflite_get_tensor_by_name (&s_interpreter, 0, "serving_default_input:0",  &s_tensor_input);
    tflite_get_tensor_by_name (&s_interpreter, 1, "PartitionedCall:0", &s_tensor_output);
    load_label_map (label_buf, label_size);
    DBG_LOGI("init_tflite_classification success...");
    return 0;
}

int
tflite_getInputType_interface ()
{
    if (s_tensor_input.type == kTfLiteUInt8)
        return 1;
    else
        return 0;
}

void *
get_classification_input_buf (int *w, int *h)
{
    *w = s_tensor_input.dims[3];
    *h = s_tensor_input.dims[2];
    return s_tensor_input.ptr;
}



/* -------------------------------------------------- *
 * Invoke TensorFlow Lite
 * -------------------------------------------------- */
static float
get_scoreval (int class_id)
{
    if (s_tensor_output.type == kTfLiteFloat32)
    {
        float *val;
        if(s_tensor_output.ptr !=NULL){
            val = (float *)s_tensor_output.ptr;
        }
        return val[class_id];
    }

//    if (s_tensor_output.type == kTfLiteUInt8)
//    {
//        uint8_t *val8 = (uint8_t *)s_tensor_output.ptr;
//        float scale = s_tensor_output.quant_scale;
//        float zerop = s_tensor_output.quant_zerop;
//        float fval = (val8[class_id] - zerop) * scale;
//        return fval;
//    }

    return 0;
}

static int
push_listitem (std::list<classify_t> &class_list, classify_t &item, size_t topn)
{
    size_t idx = 0;

    /* search insert point */
    for (auto itr = class_list.begin(); itr != class_list.end(); itr ++)
    {
        if (item.score > itr->score)
        {
            class_list.insert (itr, item);
            if (class_list.size() > topn)
            {
                class_list.pop_back();
            }
            return 0;
        }

        idx ++;
        if (idx >= topn)
        {
            return 0;
        }
    }

    /* if list is not full, add item to the bottom */
    if (class_list.size() < topn)
    {
        class_list.push_back (item);
    }
    return 0;
}


int
tflite_invoke_interface (classification_result_t *class_ret)
{
    size_t topn = MAX_CLASS_NUM;

    if (s_interpreter.interpreter->Invoke() != kTfLiteOk)
    {
        fprintf (stderr, "ERR: %s(%d)\n", __FILE__, __LINE__);
        return -1;
    }

    std::list<classify_t> classify_list;
    for (int i = 0; i < MAX_CLASS_NUM; i ++)
    {
        classify_t item;
        item.id = i;
        item.score = get_scoreval(i);
        push_listitem (classify_list, item, topn);
    }
    int count = 0;
    for (auto itr = classify_list.begin(); itr != classify_list.end(); itr ++)
    {
        classify_t *item = &class_ret->classify[count];

        item->id    = itr->id;
        item->score = itr->score;
        memcpy (item->name, s_class_name[itr->id], 64);

        count ++;
        class_ret->num = count;
    }
    return 0;
}


bool
tflite_init_interface(AAssetManager *assetMgr)
{
    int ret;
    // 初始化
    std::vector<uint8_t> m_tflite_model_buf;
    std::vector<uint8_t> m_label_map_buf;

    asset_read_file(assetMgr,
                    (char *)CLASSIFY_MODEL_PATH, m_tflite_model_buf);

    asset_read_file(assetMgr,
                    (char *)CLASSIFY_LABEL_MAP_PATH, m_label_map_buf);

    ret = init_tflite_classification (
            (const char *)m_tflite_model_buf.data(), m_tflite_model_buf.size(),
            (const char *)m_label_map_buf.data(), m_label_map_buf.size());

    return ret;
}

bool
tflite_feedData_interface(float *pBuffer)
{
    int x, y, w, h;
    float *buf_fp32 = (float *) get_classification_input_buf (&w, &h);
    float *buf_fp32_vis = buf_fp32;
    static float *buf = NULL;

    // 数据拷贝
    if (buf == NULL)
        buf = (float*) malloc(w * h * 1);
    buf = pBuffer;

    //Normalization
    float mean = 0.5f;
    float std  = 0.5f;
    for (y = 0; y < h; y ++)
    {
        for (x = 0; x < w; x ++)
        {
            *buf_fp32++ = (float)(*buf ++ - mean) / std;
        }
    }
#if 1 /* for debug */
    //可视化输入数据
    for(int i =0; i < 10; i++)
    {
        DBG_LOGI("input data buf_fp32_vis: %f,", *buf_fp32_vis++);
    }
#endif

    return 0;

}

