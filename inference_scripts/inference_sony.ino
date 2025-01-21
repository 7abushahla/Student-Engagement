#include <Camera.h>
#include <malloc.h> // Include malloc.h for mallinfo()
#include <SDHCI.h>
#include <File.h>

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "fully_quantized_small_model_int8.h"  // Your model file after conversion

// Globals for TensorFlow Lite
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

constexpr int kTensorArenaSize = 300000;  // 300 KB
uint8_t tensor_arena[kTensorArenaSize];

// Image and model input size
const int capture_width = 160;
const int capture_height = 120;
const int target_w = 128;
const int target_h = 128;
const int pixfmt = CAM_IMAGE_PIX_FMT_YUV422;  // Image format

// Constants for inference loop
const int NUM_INFERENCES = 1000;
int inference_count = 0;
float total_inference_time = 0.0;
float total_end_to_end_time = 0.0;
std::vector<float> inference_times;
std::vector<float> end_to_end_times;
std::vector<float> fps_values;

SDClass SD;
File myFile;

// Function to report memory usage
void ReportMemoryUsage(const char* context) {
    struct mallinfo mi = mallinfo();
    Serial.print(context);
    Serial.print(" Free memory: ");
    Serial.println(mi.fordblks, DEC);  // Free space in the heap
}

// Error handling for the camera
void printError(enum CamErr err) {
    Serial.print("Error: ");
    switch (err) {
        case CAM_ERR_NO_DEVICE: Serial.println("No Device"); break;
        case CAM_ERR_ILLEGAL_DEVERR: Serial.println("Illegal device error"); break;
        case CAM_ERR_ALREADY_INITIALIZED: Serial.println("Already initialized"); break;
        case CAM_ERR_NOT_INITIALIZED: Serial.println("Not initialized"); break;
        case CAM_ERR_CANT_CREATE_THREAD: Serial.println("Failed to create thread"); break;
        case CAM_ERR_INVALID_PARAM: Serial.println("Invalid parameter"); break;
        case CAM_ERR_NO_MEMORY: Serial.println("No memory"); break;
        case CAM_ERR_USR_INUSED: Serial.println("Buffer already in use"); break;
        case CAM_ERR_NOT_PERMITTED: Serial.println("Operation not permitted"); break;
        default: break;
    }
}

// Callback for camera processing
void CamCB(CamImage img) {
    if (!img.isAvailable()) {
        Serial.println("Image is not available");
        return;
    }

    // Start end-to-end timer
    unsigned long start_end_to_end = micros();

    uint16_t* buf = (uint16_t*)img.getImgBuff();
    int n = 0;

    // Start preprocessing timer
    unsigned long start_preprocessing = micros();

    // Resize and normalize the image to fit the model input
    for (int y = 0; y < target_h; ++y) {
        for (int x = 0; x < target_w; ++x) {
            int downsampled_y = y * (capture_height / target_h);
            int downsampled_x = x * (capture_width / target_w);

            uint16_t value = buf[downsampled_y * capture_width + downsampled_x];
            uint16_t y_h = (value & 0xf000) >> 8;
            uint16_t y_l = (value & 0x00f0) >> 4;
            value = (y_h | y_l);

            input->data.f[n++] = ((float)(value) / 127.5) - 1.0;  // Normalize to [-1, 1]
        }
    }

    // End preprocessing timer
    unsigned long end_preprocessing = micros();
    float preprocessing_time = (end_preprocessing - start_preprocessing) / 1000.0;  // Convert to ms

    // Start inference timer
    unsigned long start_inference = micros();
    TfLiteStatus invoke_status = interpreter->Invoke();
    unsigned long end_inference = micros();

    if (invoke_status != kTfLiteOk) {
        Serial.println("Inference failed");
        return;
    }

    // Record times
    float inference_time = (end_inference - start_inference) / 1000.0;  // Convert to ms
    total_inference_time += inference_time;
    inference_times.push_back(inference_time);

    unsigned long end_end_to_end = micros();
    float end_to_end_time = (end_end_to_end - start_end_to_end) / 1000.0;  // Convert to ms
    total_end_to_end_time += end_to_end_time;
    end_to_end_times.push_back(end_to_end_time);

    float fps = 1.0 / inference_time;
    fps_values.push_back(fps);

    inference_count++;

    // Print only for every 100th inference
    if (inference_count % 100 == 0 || inference_count == NUM_INFERENCES) {
        // Get model output for 3 classes
        int8_t paper_score = output->data.int8[0];
        int8_t screen_score = output->data.int8[1];
        int8_t wander_score = output->data.int8[2];

        Serial.println("-----------");
        Serial.print("Inference ");
        Serial.print(inference_count);
        Serial.println(":");
        Serial.print("Paper Score: " + String(paper_score) + ", ");
        Serial.print("Screen Score: " + String(screen_score) + ", ");
        Serial.println("Wander Score: " + String(wander_score));
        Serial.println("Preprocessing Time: " + String(preprocessing_time, 4) + " ms");
        Serial.println("Inference Time: " + String(inference_time, 4) + " ms");
        Serial.println("End-to-End Time: " + String(end_to_end_time, 4) + " ms");
    }

    // Save all results and statistics only at the end of all inferences
    if (inference_count == NUM_INFERENCES) {
        myFile = SD.open("dir/inference_results.txt", FILE_WRITE);
        if (myFile) {
            myFile.println("Inference Count\tInference Time (ms)\tEnd-to-End Time (ms)\tFPS");
            for (int i = 0; i < inference_count; i++) {
                myFile.print(String(i + 1) + "\t");
                myFile.print(String(inference_times[i], 4) + "\t");
                myFile.print(String(end_to_end_times[i], 4) + "\t");
                myFile.println(String(fps_values[i], 4));
            }

            // Write statistics
            float avg_inference_time = total_inference_time / inference_count;
            float avg_end_to_end_time = total_end_to_end_time / inference_count;
            float avg_fps = 0.0;
            for (float fps : fps_values) {
                avg_fps += fps;
            }
            avg_fps /= inference_count;

            float sum_sq_inference = 0.0;
            float sum_sq_end_to_end = 0.0;
            float sum_sq_fps = 0.0;

            for (float t : inference_times) {
                sum_sq_inference += (t - avg_inference_time) * (t - avg_inference_time);
            }
            for (float t : end_to_end_times) {
                sum_sq_end_to_end += (t - avg_end_to_end_time) * (t - avg_end_to_end_time);
            }
            for (float fps : fps_values) {
                sum_sq_fps += (fps - avg_fps) * (fps - avg_fps);
            }

            float std_inference_time = sqrt(sum_sq_inference / inference_count);
            float std_end_to_end_time = sqrt(sum_sq_end_to_end / inference_count);
            float std_fps = sqrt(sum_sq_fps / inference_count);

            myFile.println();
            myFile.println("Statistics:");
            myFile.print("Average Inference Time:\t");
            myFile.println(String(avg_inference_time, 4));
            myFile.print("Standard Deviation Inference Time:\t");
            myFile.println(String(std_inference_time, 4));
            myFile.print("Average End-to-End Time:\t");
            myFile.println(String(avg_end_to_end_time, 4));
            myFile.print("Standard Deviation End-to-End Time:\t");
            myFile.println(String(std_end_to_end_time, 4));
            myFile.print("Average FPS:\t");
            myFile.println(String(avg_fps, 4));
            myFile.print("Standard Deviation FPS:\t");
            myFile.println(String(std_fps, 4));
            myFile.close();
            Serial.println("Inference results saved to dir/inference_results.txt");
        } else {
            Serial.println("Error opening file for writing results.");
        }

        Serial.println("Completed all inferences.");
        Serial.print("Average Inference Time: ");
        Serial.print(total_inference_time / NUM_INFERENCES, 4);
        Serial.println(" ms");
        Serial.print("Average End-to-End Time: ");
        Serial.print(total_end_to_end_time / NUM_INFERENCES, 4);
        Serial.println(" ms");
        while (true);
    }
}

void setup() {
    CamErr err;

    Serial.begin(115200);
    while (!Serial);

    Serial.print("Insert SD card.");
    while (!SD.begin()) {
        ;  // Wait until SD card is mounted
    }

    SD.mkdir("dir/");

    tflite::InitializeTarget();
    memset(tensor_arena, 0, kTensorArenaSize * sizeof(uint8_t));

    static tflite::MicroErrorReporter micro_error_reporter;
    error_reporter = &micro_error_reporter;

    model = tflite::GetModel(model_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("Model schema version mismatch!");
        return;
    }

    static tflite::AllOpsResolver resolver;

    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
    interpreter = &static_interpreter;

    ReportMemoryUsage("Before AllocateTensors(): ");
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("AllocateTensors() failed.");
        return;
    }
    ReportMemoryUsage("After AllocateTensors(): ");

    input = interpreter->input(0);
    output = interpreter->output(0);

    Serial.println("TensorFlow Lite model setup completed.");

    Serial.println("Preparing camera...");
    err = theCamera.begin(1, CAM_VIDEO_FPS_15, capture_width, capture_height, pixfmt);
    if (err != CAM_ERR_SUCCESS) {
        printError(err);
        return;
    }

    Serial.println("Starting camera streaming...");
    err = theCamera.startStreaming(true, CamCB);
    if (err != CAM_ERR_SUCCESS) {
        printError(err);
        return;
    }

    Serial.println("Camera ready.");
}

void loop() {
    // Do nothing, as the logic is handled in the CamCB function.
}
