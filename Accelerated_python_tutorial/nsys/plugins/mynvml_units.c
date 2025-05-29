#include <stdio.h>
#include <stdbool.h>
#include <unistd.h>

#include <nvml.h>
#include <nvtx3/nvToolsExtCounters.h>
#include <nvtx3/nvToolsExtSemanticsCounters.h>

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

// Get NVTX counter semantics with unit already set.
static nvtxSemanticsCounter_t getCounterSemantic(const char* unit)
{
    nvtxSemanticsCounter_t semantic;
    memset(&semantic, 0, sizeof(semantic));

    semantic.header.structSize = sizeof(nvtxSemanticsCounter_t);
    semantic.header.semanticId = NVTX_SEMANTIC_ID_COUNTERS_V1;
    semantic.header.version = NVTX_COUNTER_SEMANTIC_VERSION;
    semantic.unit = unit;

    return semantic;
}

// Setup the NVTX counter group for the NVML counters and get an ID for it.
static uint64_t getNvtxCounterId(unsigned int deviceIdx)
{
    nvtxSemanticsCounter_t percent = getCounterSemantic("%");
    percent.flags = NVTX_COUNTER_FLAG_LIMITS;
    percent.limitType = NVTX_COUNTER_LIMIT_U64;
    percent.min.u64 = 0;
    percent.max.u64 = 100;

    nvtxSemanticsCounter_t hz = getCounterSemantic("Hz");

    // Describe the payload's data layout (counters_t) and add semantics.
    const nvtxPayloadSchemaEntry_t schema[] = {
        {0, NVTX_PAYLOAD_ENTRY_TYPE_UINT, "GPU Util",
            NULL, 0, 0, (nvtxSemanticsHeader_t*)&percent},
        {0, NVTX_PAYLOAD_ENTRY_TYPE_UINT, "NvEnc Util",
            NULL, 0, 0, (nvtxSemanticsHeader_t*)&percent},
        {0, NVTX_PAYLOAD_ENTRY_TYPE_UINT, "NvDec Util",
            NULL, 0, 0, (nvtxSemanticsHeader_t*)&percent},
        {0, NVTX_PAYLOAD_ENTRY_TYPE_UINT, "Video Clock",
            NULL, 0, 0, (nvtxSemanticsHeader_t*)&hz}
    };

    // Boiler plate for payload schema registration.
    nvtxPayloadSchemaAttr_t schemaAttr;
    schemaAttr.fieldMask = NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_TYPE
        | NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_ENTRIES
        | NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_NUM_ENTRIES
        | NVTX_PAYLOAD_SCHEMA_ATTR_FIELD_STATIC_SIZE;
    schemaAttr.type = NVTX_PAYLOAD_SCHEMA_TYPE_STATIC;
    schemaAttr.entries = schema;
    schemaAttr.numEntries = sizeof(schema)/sizeof(schema[0]);
    schemaAttr.payloadStaticSize = sizeof(counters_t);
    const uint64_t counters_t_schemaId = nvtxPayloadSchemaRegister(NvmlDomain, &schemaAttr);

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
        counters.videoClock *= 1000000; // MHz to Hz

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
