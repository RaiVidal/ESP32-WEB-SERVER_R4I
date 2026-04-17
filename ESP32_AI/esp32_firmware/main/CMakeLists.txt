idf_component_register(
    SRCS "main.cpp"
    INCLUDE_DIRS "."
    REQUIRES
        nvs_flash
        esp_timer
        spiffs
        esp_partition
)

# Build a SPIFFS image from ../spiffs and include it in flash-related targets.
spiffs_create_partition_image(spiffs ../spiffs FLASH_IN_PROJECT)