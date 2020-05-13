#include "accelerometer_handler.h"
#include "config.h"
#include "magic_wand_model_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "mbed.h"
#include "uLCD_4DGL.h"
#include <cmath>
#include "DA7212.h"

#define bufferLength (32)
#define signalLength (106)

DA7212 audio;
int16_t waveform[kAudioTxBufferSize];
Serial pc(USBTX, USBRX);
uLCD_4DGL uLCD(D1, D0, D2);

Thread thread(osPriorityNormal, 120 * 1024 /*120K stack size*/);
Thread t;

InterruptIn button(SW2);
InterruptIn button1(SW3);

int signal1[signalLength];

char serialInBuffer[bufferLength];

int serialCount = 0;


DigitalOut green_led(LED2);

int gesture_index;

int song1[42];
int song2[34];
int song3[30];

/*int song1[42] =
    {

        261, 261, 392, 392, 440, 440, 392,

        349, 349, 330, 330, 294, 294, 261,

        392, 392, 349, 349, 330, 330, 294,

        392, 392, 349, 349, 330, 330, 294,

        261, 261, 392, 392, 440, 440, 392,

        349, 349, 330, 330, 294, 294, 0};*/

int noteLength1[42] =
    {

        1, 1, 1, 1, 1, 1, 2,

        1, 1, 1, 1, 1, 1, 2,

        1, 1, 1, 1, 1, 1, 2,

        1, 1, 1, 1, 1, 1, 2,

        1, 1, 1, 1, 1, 1, 2,

        1, 1, 1, 1, 1, 1, 2};

/*int song2[34] =
    {

        261, 293, 329, 261, 261, 293, 329, 261, 329, 349, 392, 329, 349, 392, 392, 440, 392, 349,
        329, 261, 392, 440, 392, 349, 329, 261, 293, 392, 261, 261, 293, 392, 261, 0

};*/

int noteLength2[34] = {

    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
/*
int song3[30] = {

    329, 293, 261, 329, 329, 329, 0, 293, 293, 293, 0, 329, 392, 392, 0,
    329, 293, 261, 329, 329, 329, 329, 293, 293, 329, 293, 261, 0, 0, 0};
*/

int noteLength3[30] = {

    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

void loadSignal(void)
{
    green_led = 0;

    int i = 0;

    serialCount = 0;

    audio.spk.pause();

    while (i < signalLength)
    {
        if (pc.readable())
        {
            serialInBuffer[serialCount] = pc.getc();
            serialCount++;

            if (serialCount == 3)
            {
                serialInBuffer[serialCount] = '\0';
                signal1[i] = (int)atoi(serialInBuffer);
                serialCount = 0;

                i++;
            }
        }
    }

    green_led = 1;
}

void playNote(int freq)
{
    for (int i = 0; i < kAudioTxBufferSize; i++)
    {
        waveform[i] = (int16_t)(sin((double)i * 2. * M_PI / (double)(kAudioSampleFrequency / freq)) * ((1 << 16) - 1));
    }

    audio.spk.play(waveform, kAudioTxBufferSize);
}

int play_music(int num)
{
    int len, length;

    if (num == 0)
        len = 42;
    else if (num == 1)
        len = 34;
    else
        len = 30;

    for (int i = 0; i < len; i++)
    {
        if (num == 0)
            length = noteLength1[i];
        else if (num == 1)
            length = noteLength2[i];
        else
            length = noteLength3[i];

        while (length--)
        {

            // the loop below will play the note for the duration of 1s

            for (int j = 0; j < kAudioSampleFrequency / kAudioTxBufferSize; ++j)
            {
                if (num == 0)
                    playNote(song1[i]);
                else if (num == 1)
                    playNote(song2[i]);
                else
                    playNote(song3[i]);
            }

            if (length < 1)
                wait(1.0);
        }
        playNote(0);
    }
}

int PredictGesture(float *output)
{

    // How many times the most recent gesture has been matched in a row

    static int continuous_count = 0;

    // The result of the last prediction

    static int last_predict = -1;

    // Find whichever output has a probability > 0.8 (they sum to 1)

    int this_predict = -1;

    for (int i = 0; i < label_num; i++)
    {

        if (output[i] > 0.8)
            this_predict = i;
    }

    // No gesture was detected above the threshold

    if (this_predict == -1)
    {

        continuous_count = 0;

        last_predict = label_num;

        return label_num;
    }

    if (last_predict == this_predict)
    {

        continuous_count += 1;
    }
    else
    {

        continuous_count = 0;
    }

    last_predict = this_predict;

    // If we haven't yet had enough consecutive matches for this gesture,

    // report a negative result

    if (continuous_count < config.consecutiveInferenceThresholds[this_predict])
    {

        return label_num;
    }

    // Otherwise, we've seen a positive result, so clear all our variables

    // and report it

    continuous_count = 0;

    last_predict = -1;

    return this_predict;
}

void machine_learning()
{
    // Create an area of memory to use for input, output, and intermediate arrays.

    // The size of this will depend on the model you're using, and may need to be

    // determined by experimentation.

    constexpr int kTensorArenaSize = 60 * 1024;

    uint8_t tensor_arena[kTensorArenaSize];

    // Whether we should clear the buffer next time we fetch data

    bool should_clear_buffer = false;

    bool got_data = false;

    // Set up logging.

    static tflite::MicroErrorReporter micro_error_reporter;

    tflite::ErrorReporter *error_reporter = &micro_error_reporter;

    // Map the model into a usable data structure. This doesn't involve any

    // copying or parsing, it's a very lightweight operation.

    const tflite::Model *model = tflite::GetModel(g_magic_wand_model_data);

    if (model->version() != TFLITE_SCHEMA_VERSION)
    {

        error_reporter->Report(

            "Model provided is schema version %d not equal "

            "to supported version %d.",

            model->version(), TFLITE_SCHEMA_VERSION);

        return -1;
    }

    // Pull in only the operation implementations we need.

    // This relies on a complete list of all the ops needed by this graph.

    // An easier approach is to just use the AllOpsResolver, but this will

    // incur some penalty in code space for op implementations that are not

    // needed by this graph.

    static tflite::MicroOpResolver<6> micro_op_resolver;

    micro_op_resolver.AddBuiltin(

        tflite::BuiltinOperator_DEPTHWISE_CONV_2D,

        tflite::ops::micro::Register_DEPTHWISE_CONV_2D());

    micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_MAX_POOL_2D,

                                 tflite::ops::micro::Register_MAX_POOL_2D());

    micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_CONV_2D,

                                 tflite::ops::micro::Register_CONV_2D());

    micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_FULLY_CONNECTED,

                                 tflite::ops::micro::Register_FULLY_CONNECTED());

    micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_SOFTMAX,

                                 tflite::ops::micro::Register_SOFTMAX());

    micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_RESHAPE,
                                 tflite::ops::micro::Register_RESHAPE(), 1);

    // Build an interpreter to run the model with

    static tflite::MicroInterpreter static_interpreter(

        model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);

    tflite::MicroInterpreter *interpreter = &static_interpreter;

    // Allocate memory from the tensor_arena for the model's tensors

    interpreter->AllocateTensors();

    // Obtain pointer to the model's input tensor

    TfLiteTensor *model_input = interpreter->input(0);

    if ((model_input->dims->size != 4) || (model_input->dims->data[0] != 1) ||

        (model_input->dims->data[1] != config.seq_length) ||

        (model_input->dims->data[2] != kChannelNumber) ||

        (model_input->type != kTfLiteFloat32))
    {

        error_reporter->Report("Bad input tensor parameters in model");

        return -1;
    }

    int input_length = model_input->bytes / sizeof(float);

    TfLiteStatus setup_status = SetupAccelerometer(error_reporter);

    if (setup_status != kTfLiteOk)
    {

        error_reporter->Report("Set up failed\n");

        return -1;
    }

    error_reporter->Report("Set up successful...\n");

    while (true)
    {

        // Attempt to read new data from the accelerometer

        got_data = ReadAccelerometer(error_reporter, model_input->data.f,

                                     input_length, should_clear_buffer);

        // If there was no new data,

        // don't try to clear the buffer again and wait until next time

        if (!got_data)
        {

            should_clear_buffer = false;

            continue;
        }

        // Run inference, and report any error

        TfLiteStatus invoke_status = interpreter->Invoke();

        if (invoke_status != kTfLiteOk)
        {

            error_reporter->Report("Invoke failed on index: %d\n", begin_index);

            continue;
        }

        // Analyze the results to obtain a prediction

        gesture_index = PredictGesture(interpreter->output(0)->data.f);

        // Clear the buffer next time we read data

        should_clear_buffer = gesture_index < label_num;

        // Produce an output

        if (gesture_index < label_num)
        {
            error_reporter->Report(config.output_message[gesture_index]);
        }
    }
}

int in;
void getIndex()
{
    while (1)
    {
        in = gesture_index;

        wait(.5);
    }
}

int main(int argc, char *argv[])
{
    thread.start(machine_learning);
    t.start(getIndex);

    int num = 0;

    loadSignal();
    for (int i = 0; i < 106; i++)
    {
        if (i < 42)
            song1[i] = signal1[i];
        else if (i >= 42 && i < 76)
            song2[i - 42] = signal1[i];
        else 
            song3[i - 76] = signal1[i];
    }

    while (true)
    {
        if (in == 0 && button == 0)
        {
            uLCD.reset();

            for (int i = 0; i < 2; i++)
            {
                uLCD.locate(0, 0);
                uLCD.text_width(2); //4X size text
                uLCD.text_height(2);
                uLCD.color(GREEN);
                uLCD.printf("\n mode \n\n front");

                wait(.5);
            }

            num = 1;
            uLCD.reset();
            while (button != 0)
            {
                if (in && button1 == 0)
                {
                    if (num < 3)
                        num++;
                    else
                        num = 1;

                    play_music(num - 1);

                    uLCD.locate(0, 0);
                    uLCD.text_width(2); //4X size text
                    uLCD.text_height(2);
                    uLCD.printf("\n Song %1D", num);
                }
                else
                {
                    uLCD.locate(0, 0);
                    uLCD.text_width(2); //4X size text
                    uLCD.text_height(2);
                    uLCD.printf("\n Song %1D", num);
                }
            }

            in = 3;
            uLCD.reset();
        }
        else if (in == 1 && button == 0)
        {
            uLCD.reset();

            for (int i = 0; i < 2; i++)
            {
                uLCD.locate(0, 0);
                uLCD.text_width(2); //4X size text
                uLCD.text_height(2);
                uLCD.color(GREEN);
                uLCD.printf("\n mode \n\n back");

                wait(.5);
            }

            num = 1;
            uLCD.reset();
            while (button != 0)
            {
                if (in && button1 == 0)
                {
                    if (num > 1)
                        num--;
                    else
                        num = 3;

                    play_music(num - 1);

                    uLCD.locate(0, 0);
                    uLCD.text_width(2); //4X size text
                    uLCD.text_height(2);
                    uLCD.printf("\n Song %1D", num);
                }
                else
                {
                    uLCD.locate(0, 0);
                    uLCD.text_width(2); //4X size text
                    uLCD.text_height(2);
                    uLCD.printf("\n Song %1D", num);
                }
            }

            in = 3;
            uLCD.reset();
        }
        else if (in == 2 && button == 0)
        {
            uLCD.reset();

            for (int i = 0; i < 2; i++)
            {
                uLCD.locate(0, 0);
                uLCD.text_width(2); //4X size text
                uLCD.text_height(2);
                uLCD.color(GREEN);
                uLCD.printf("\n mode \n\n change \n songs");

                wait(.5);
            }

            uLCD.reset();
            while (button != 0)
            {
                if (in == 0 && button1 == 0)
                {

                    uLCD.reset();

                    uLCD.locate(0, 0);
                    uLCD.text_width(2); //4X size text
                    uLCD.text_height(2);
                    uLCD.printf("\n Song 1");

                    play_music(0);

                    in = 3;
                    uLCD.reset();
                }
                else if (in == 1 && button1 == 0)
                {

                    uLCD.reset();

                    uLCD.locate(0, 0);
                    uLCD.text_width(2); //4X size text
                    uLCD.text_height(2);
                    uLCD.printf("\n Song 2");
                    play_music(1);

                    in = 3;
                    uLCD.reset();
                }
                else if (in == 2 && button1 == 0)
                {

                    uLCD.reset();

                    uLCD.locate(0, 0);
                    uLCD.text_width(2); //4X size text
                    uLCD.text_height(2);
                    uLCD.printf("\n Song 3");
                    play_music(2);

                    in = 3;
                    uLCD.reset();
                }
                else
                {
                    uLCD.locate(0, 0);
                    uLCD.text_width(2); //4X size text
                    uLCD.text_height(2);
                    uLCD.printf("\n choose \n Song !");
                }
            }

            in = 3;
            uLCD.reset();
        }
        else
        {
            uLCD.text_width(1); //4X size text
            uLCD.text_height(1);
            uLCD.color(BLUE);
            uLCD.locate(0, 0);
            uLCD.printf("\n107061210"); // 107061210

            uLCD.text_width(2); //4X size text
            uLCD.text_height(2);
            uLCD.color(GREEN);
            uLCD.printf("\n Music \n  Player");

            uLCD.text_width(4); //4X size text
            uLCD.text_height(4);
            uLCD.color(GREEN);

            uLCD.circle(60, 94, 25, RED);
            uLCD.triangle(50, 79, 50, 109, 80, 94, RED);
        }
    }
}
