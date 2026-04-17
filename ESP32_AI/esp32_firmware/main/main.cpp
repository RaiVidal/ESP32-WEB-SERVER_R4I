/**
 * main.cpp
 * ========
 * ESP32 firmware for loading an ESP-DL model (.espdl) from SPIFFS,
 * running inference, and reporting latency over UART.
 *
 * Supports:
 *   - Single-layer benchmarks (any exported layer)
 *   - Full TinyCNN image classification
 *   - Configurable via menuconfig (see Kconfig.projbuild)
 *
 * Build & flash:
 *   idf.py set-target esp32c3   # or esp32s3
 *   idf.py build
 *   idf.py -p /dev/ttyUSB0 flash monitor
 *
 * Board requirements:
 *   - ESP32-C3 SuperMini (or ESP32-S3)
 *   - ESP-IDF v5.3+
 *   - Component: espressif/esp-dl (via idf_component.yml)
 */

#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

#include "esp_chip_info.h"
#include "esp_private/esp_clk.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "nvs_flash.h"

// SPIFFS
#include "esp_spiffs.h"

// ESP-DL
#include "dl_model_base.hpp"
#include "dl_tensor_base.hpp"

#ifdef CONFIG_IDF_TARGET_ESP32S3
static const char *TAG = "NN_ESP32S3";
#elif CONFIG_IDF_TARGET_ESP32
static const char *TAG = "NN_ESP32";
#elif CONFIG_IDF_TARGET_ESP32C3
static const char *TAG = "NN_ESP32C3";
#endif

static constexpr uint32_t kBytesPerMb = 1024U * 1024U;
static constexpr uint32_t kIdleDelayMs = 5000U;

// ---------------------------------------------------------------------------
// Configuration (override via menuconfig / sdkconfig)
// ---------------------------------------------------------------------------

#ifndef CONFIG_MODEL_PATH
#define CONFIG_MODEL_PATH   "/spiffs/tiny_cnn.espdl"
#endif

#ifndef CONFIG_INTERNAL_RAM_MB
#define CONFIG_INTERNAL_RAM_MB  2          // MB of internal SRAM for model ops
#endif

#ifndef CONFIG_N_RUNS
#define CONFIG_N_RUNS           50         // inference repetitions for benchmark
#endif

// ---------------------------------------------------------------------------
// SPIFFS mount
// ---------------------------------------------------------------------------

static esp_err_t mount_spiffs(void)
{
    esp_vfs_spiffs_conf_t conf = {
        .base_path              = "/spiffs",
        .partition_label        = NULL,   // use default "spiffs" partition
        .max_files              = 5,
        .format_if_mount_failed = false,
    };

    esp_err_t ret = esp_vfs_spiffs_register(&conf);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "SPIFFS mount failed: %s", esp_err_to_name(ret));
        return ret;
    }

    size_t total = 0, used = 0;
    esp_spiffs_info(conf.partition_label, &total, &used);
    ESP_LOGI(TAG, "SPIFFS: %u / %u bytes used", (unsigned)used, (unsigned)total);
    return ESP_OK;
}

// ---------------------------------------------------------------------------
// Memory report helper
// ---------------------------------------------------------------------------

static void print_heap(const char *label)
{
#if CONFIG_SPIRAM
    ESP_LOGI(TAG, "[MEM] %s: internal=%u B  PSRAM=%u B",
             label,
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
#else
    ESP_LOGI(TAG, "[MEM] %s: internal=%u B  PSRAM=not present",
             label,
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
#endif
}

// ---------------------------------------------------------------------------
// SYSINFO (ESP32-C3: sysinfo-only firmware)
// ---------------------------------------------------------------------------

static void print_sysinfo_c3_and_idle(const esp_chip_info_t &chip_info)
{
    const uint32_t cpu_mhz = (uint32_t)(esp_clk_cpu_freq() / 1000000U);
    const uint32_t free_internal = (uint32_t)heap_caps_get_free_size(MALLOC_CAP_INTERNAL);

    ESP_LOGI(TAG, "[SYSINFO] Board        : %s", "esp32c3_mini");
    ESP_LOGI(TAG, "[SYSINFO] Chip         : ESP32-C3 (RISC-V, %u core%s)",
             (unsigned)chip_info.cores,
             (chip_info.cores == 1) ? "" : "s");
    ESP_LOGI(TAG, "[SYSINFO] CPU freq     : %u MHz", (unsigned)cpu_mhz);
    ESP_LOGI(TAG, "[SYSINFO] Free heap    : %u bytes (internal)", (unsigned)free_internal);
    ESP_LOGI(TAG, "[SYSINFO] PSRAM        : not present");
    ESP_LOGI(TAG, "[SYSINFO] ESP-DL       : NOT SUPPORTED on this chip");
    ESP_LOGI(TAG,
             "[SYSINFO] Note         : This board is for PC-side benchmarking only.\n"
             "                                 Flash with S3 or WROOM firmware for inference.");

    while (true) {
        vTaskDelay(pdMS_TO_TICKS(kIdleDelayMs));
    }
}

// ---------------------------------------------------------------------------
// Inference benchmark
// ---------------------------------------------------------------------------

static void run_inference_benchmark(dl::Model *model,
                                    dl::TensorBase *input_tensor)
{
    // Warm-up: 5 runs (not timed)
    for (int i = 0; i < 5; ++i) {
        model->run(input_tensor);
    }

    // Timed runs
    int64_t t_start = esp_timer_get_time();   // µs
    for (int i = 0; i < CONFIG_N_RUNS; ++i) {
        model->run(input_tensor);
    }
    int64_t t_end = esp_timer_get_time();

    float total_ms = (t_end - t_start) / 1000.0f;
    float avg_ms   = total_ms / CONFIG_N_RUNS;

    ESP_LOGI(TAG, "──────────────────────────────────────");
    ESP_LOGI(TAG, "Benchmark (%d runs):", CONFIG_N_RUNS);
    ESP_LOGI(TAG, "  Total   : %.2f ms", total_ms);
    ESP_LOGI(TAG, "  Average : %.3f ms / inference", avg_ms);
    ESP_LOGI(TAG, "  FPS     : %.1f", 1000.0f / avg_ms);
    ESP_LOGI(TAG, "──────────────────────────────────────");
}

// ---------------------------------------------------------------------------
// App main
// ---------------------------------------------------------------------------

extern "C" void app_main(void)
{
    esp_chip_info_t chip_info;
    esp_chip_info(&chip_info);

    const bool is_chip_c3 = (chip_info.model == CHIP_ESP32C3);
    if (is_chip_c3 || (strcmp(CONFIG_BOARD_TARGET, "esp32c3_mini") == 0)) {
        print_sysinfo_c3_and_idle(chip_info);
    }

    ESP_LOGI(TAG, "ESP32 Neural Network Inference Demo");
    ESP_LOGI(TAG, "Model path : %s", CONFIG_MODEL_PATH);
    ESP_LOGI(TAG, "Internal RAM for model ops: %d MB", CONFIG_INTERNAL_RAM_MB);

    const bool is_chip_esp32 = (chip_info.model == CHIP_ESP32);
    if (is_chip_esp32 || (strcmp(CONFIG_BOARD_TARGET, "esp32_wroom32") == 0)) {
        ESP_LOGW(TAG, "[WARN] WROOM-32: no SIMD, expect ~5x slower inference");
    }

    // NVS init (required by some ESP-IDF components)
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    // Mount SPIFFS
    ESP_ERROR_CHECK(mount_spiffs());
    print_heap("after SPIFFS mount");

    // ── Load model ──────────────────────────────────────────────────────────
    ESP_LOGI(TAG, "Loading model from %s …", CONFIG_MODEL_PATH);

    // Verify file exists
    struct stat st;
    if (stat(CONFIG_MODEL_PATH, &st) != 0) {
        ESP_LOGE(TAG, "Model file not found: %s", CONFIG_MODEL_PATH);
        ESP_LOGE(TAG, "Flash SPIFFS with: idf.py -p <PORT> spiffs-flash");
        return;
    }
    ESP_LOGI(TAG, "Model file size: %ld bytes", (long)st.st_size);

    // dl::Model manages memory allocation automatically.
    // CONFIG_INTERNAL_RAM_MB tells the memory planner how much internal SRAM
    // to budget — ops that fit here run ~2× faster than from PSRAM.
    // We mount SPIFFS as a filesystem and pass a file path, so use
    // MODEL_LOCATION_IN_SDCARD (path-based file loading).
    dl::Model *model = new dl::Model(
        CONFIG_MODEL_PATH,
        fbs::MODEL_LOCATION_IN_SDCARD,
        (size_t)CONFIG_INTERNAL_RAM_MB * kBytesPerMb  // bytes
    );
    if (model == nullptr) {
        ESP_LOGE(TAG, "Failed to allocate model");
        return;
    }
    print_heap("after model load");

    // ── Inspect model I/O ───────────────────────────────────────────────────
    auto inputs  = model->get_inputs();
    auto outputs = model->get_outputs();

    ESP_LOGI(TAG, "Model inputs (%zu):", inputs.size());
    for (auto &[name, t] : inputs) {
        ESP_LOGI(TAG, "  '%s'  shape=%s  dtype=%d",
                 name.c_str(), dl::vector_to_string(t->get_shape()).c_str(), (int)t->dtype);
    }
    ESP_LOGI(TAG, "Model outputs (%zu):", outputs.size());
    for (auto &[name, t] : outputs) {
        ESP_LOGI(TAG, "  '%s'  shape=%s  dtype=%d",
                 name.c_str(), dl::vector_to_string(t->get_shape()).c_str(), (int)t->dtype);
    }

    if (inputs.empty() || outputs.empty()) {
        ESP_LOGE(TAG, "Model I/O is empty. Check exported model and SPIFFS image content.");
        return;
    }

    // ── Build a synthetic input (zeros) — replace with real sensor/camera data
    dl::TensorBase *input_tensor = inputs.begin()->second;

    // Fill with a constant pattern (INT8 = 0 → float ≈ 0)
    memset(input_tensor->data, 0, input_tensor->get_size() * input_tensor->get_dtype_bytes());

    // ── Single inference (with output printing) ──────────────────────────────
    ESP_LOGI(TAG, "Running single inference …");
    model->run(input_tensor);

    // Retrieve output tensor and argmax (classification)
    dl::TensorBase *out_tensor = outputs.begin()->second;
    int n_classes = out_tensor->shape[out_tensor->shape.size() - 1];

    // Dequantise output (INT8 → float via exponent)
    float best_score = -1e9f;
    int   best_class = 0;
    int8_t *out_data = (int8_t *)out_tensor->data;

    for (int c = 0; c < n_classes; ++c) {
        // ESP-DL uses power-of-two quantisation: val = raw * 2^exponent
        float val = (float)out_data[c] * powf(2.0f, (float)out_tensor->exponent);
        if (val > best_score) {
            best_score = val;
            best_class = c;
        }
    }
    ESP_LOGI(TAG, "Predicted class : %d  (score=%.4f)", best_class, best_score);

    // ── Benchmark ───────────────────────────────────────────────────────────
    run_inference_benchmark(model, input_tensor);

    // ── Done ────────────────────────────────────────────────────────────────
    print_heap("final");
    ESP_LOGI(TAG, "Done. Entering idle loop.");

    // Keep alive — in a real application, replace with camera/sensor pipeline
    while (true) {
        vTaskDelay(pdMS_TO_TICKS(kIdleDelayMs));
    }
}