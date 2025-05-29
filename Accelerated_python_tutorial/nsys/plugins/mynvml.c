#include <stdio.h>
#include <stdbool.h>
#include <unistd.h>

#include </usr/local/cuda-12.9/include/nvml.h>
#include <nvtx3/nvToolsExtCounters.h>
#include <nvtx3/nvToolsExtPayloadHelper.h>

#define NVML_CHECK(nvmlCall) do { \
    nvmlReturn_t result = nvmlCall; \
    if (NVML_SUCCESS != result) \
    { \
        fprintf(stderr, "NVML error at %s:%d \"%s\": %s \n", \
            __FILE__,  __LINE__, #nvmlCall,  nvmlErrorString(result));\
        shutdown(); \
        exit(1); \
    } \
} while(0)

typedef struct
{
    unsigned int gpuUtil;
    unsigned int encoderUtil;
    unsigned int decoderUtil;
    unsigned int videoClock;
} counters_t;

static nvtxDomainHandle_t NvmlDomain = 0;

static void shutdown()
{
    nvmlReturn_t result = nvmlShutdown();
    if (NVML_SUCCESS != result)
    {
        fprintf(stderr, "Failed to shutdown NVML: %s\n", nvmlErrorString(result));
    }
}

// Setup the NVTX counter group for the NVML counters and get an ID for it.
static uint64_t getNvtxCounterId(unsigned int deviceIdx)
{
    NVTX_DEFINE_SCHEMA_FOR_STRUCT_AND_REGISTER(NvmlDomain, counters_t, NULL,
        NVTX_PAYLOAD_ENTRIES(
            (gpuUtil, TYPE_UINT, "GPU Util [%]"),
            (encoderUtil, TYPE_UINT, "NvEnc Util [%]"),
            (decoderUtil, TYPE_UINT, "NvDec Util [%]"),
            (videoClock, TYPE_UINT, "Video Clock [MHz]")
        )
    )

    // Setup the counter (group) attributes.
    nvtxCounterAttr_t cntAttr;
    memset(&cntAttr, 0, sizeof(cntAttr));
    cntAttr.structSize = sizeof(nvtxCounterAttr_t);
    cntAttr.schemaId = counters_t_schemaId;
    cntAttr.scopeId = NVTX_SCOPE_CURRENT_VM;

    // Use GPU + NVML device index as counter group name.
    char gpuIdString[8];
    sprintf(gpuIdString, "GPU %u", deviceIdx);
    cntAttr.name = gpuIdString;

    return nvtxCounterRegister(NvmlDomain, &cntAttr);
}

static void queryCounters(nvmlDevice_t device, uint64_t nvtxCountersId)
{
    while (true)
    {
        counters_t counters;
        NVML_CHECK(nvmlDeviceGetClock(device, NVML_CLOCK_VIDEO,
            NVML_CLOCK_ID_CURRENT, &(counters.videoClock)));

        nvmlUtilization_t utilization;
        NVML_CHECK(nvmlDeviceGetUtilizationRates(device, &utilization));
        counters.gpuUtil = utilization.gpu;

        unsigned int intervalUs;
        NVML_CHECK(nvmlDeviceGetEncoderUtilization(device,
            &(counters.encoderUtil), &intervalUs));
        NVML_CHECK(nvmlDeviceGetDecoderUtilization(device,
            &(counters.decoderUtil), &intervalUs));

        nvtxCounterSample(NvmlDomain, nvtxCountersId, &counters, sizeof(counters_t));

        // Use a 50ms interval or shorter if encoder/decoder allows it.
        unsigned int sleeptime = (intervalUs < 50000) ? intervalUs : 50000;
        usleep(sleeptime);
    }
}

int main(int argc, char** argv)
{
    nvmlReturn_t result = nvmlInit();
    if (NVML_SUCCESS != result)
    {
        fprintf(stderr, "Failed to initialize NVML: %s\n", nvmlErrorString(result));
        return 1;
    }

    unsigned int deviceIdx = (argc > 1) ? atoi(argv[1]) : 0;

    nvmlDevice_t hDevice;
    NVML_CHECK(nvmlDeviceGetHandleByIndex(deviceIdx, &hDevice));

    NvmlDomain = nvtxDomainCreateA("MyNVML");

    uint64_t countersId = getNvtxCounterId(deviceIdx);

    queryCounters(hDevice, countersId);

    shutdown();

    return 0;
}
