#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <getopt.h>
#include <dirent.h>
#include <sys/stat.h>
#include <math.h>
#include <unistd.h>
#include <time.h>       // For performance timing
#include <errno.h>      // For errno and error constants

// Architecture-specific SIMD includes
#if defined(__x86_64__) || defined(__i386__)
    #include <immintrin.h>  // AVX/SSE/AVX-512 intrinsics for x86/x64
    #define ARCH_X86
#elif defined(__aarch64__) || defined(__arm__)
    #include <arm_neon.h>   // NEON intrinsics for ARM
    #define ARCH_ARM
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

// ============================================================================
// CONSTANTS AND CONFIGURATION
// ============================================================================
#define NUM_ANTENNAS 96  // Number of antennas in the array
#define NUM_SUBBANDS 16  // Number of frequency subbands
#define NUM_CHANNELS_PER_SUBBAND 384 // Number of frequency channels per subband
#define NUM_POLARIZATIONS 2 // Number of polarizations (X and Y)
#define HEADER_SIZE_BYTES 4096 // Size of output file header in bytes
#define FILE_MAGIC "DEDISP" // Magic string for output file format
#define FILE_VERSION 1 // Output file format version
#define SAMPLE_TIME_SEC 0.000032768 // Time per sample in seconds (32.768 us)
#define TOTAL_BANDWIDTH_MHZ 187.5 // Total bandwidth in MHz
#define NUM_CHANNELS (NUM_SUBBANDS * NUM_CHANNELS_PER_SUBBAND) // Total number of channels
#define SAMPLES_PER_BLOCK 2 // Number of time samples per block in voltage data files
#define HIGHEST_FREQ_MHZ 1498.75 // Highest frequency (subband 0, channel 0)
#define CHANNEL_FREQ_STEP_MHZ -0.030517578125 // Channel frequency step (negative for descending)
#define DISPERSION_CONST 4.148808e3 // Dispersion constant (s⋅MHz²⋅cm³/pc)

// Subband start frequencies (MHz)
const double subband_start_freqs[NUM_SUBBANDS] = {
    1498.75, 1487.03125, 1475.3125, 1463.59375, 1451.875, 1440.15625, 1428.4375, 1416.71875,
    1405.0, 1393.28125, 1381.5625, 1369.84375, 1358.125, 1346.40625, 1334.6875, 1322.96875
};
const char *polarization_names[NUM_POLARIZATIONS] = {"X", "Y"};

// ============================================================================
// DATA STRUCTURES
// ============================================================================
// Represents a file entry for a subband voltage file
typedef struct {
    char name[512]; // Full path to the file
} SubbandFileEntry;

// Output modes
typedef enum {
    OUTPUT_MODE_TIME_SERIES = 0,  // Default: sum across frequency channels
    OUTPUT_MODE_SPECTRA = 1,      // Spectra: save power spectra (Stokes I)
    OUTPUT_MODE_RAW = 2           // Raw: save complex voltage time streams
} OutputMode;

// Configuration for the dedispersion process
typedef struct {
    double dispersion_measure; // DM value in pc/cm^3
    int enable_dedispersion;   // 1 = enable, 0 = disable
    char *input_directory;     // Directory containing input voltage files
    char *output_directory;    // Directory for output files
    char *candidate_name;      // Candidate name for automatic path generation
    int enable_median_subtraction; // 1 = enable median subtraction, 0 = disable
    int block_processing_size; // Number of blocks to process at once for optimization
    int median_window_size; // 0 = global median, >0 = sliding window size in time samples
    int num_threads;          // Number of OpenMP threads to use (0 = auto-detect)
    int max_threads;          // Maximum threads allowed (safeguard)
    OutputMode output_mode;   // Output mode: time_series or spectra
    double start_time;        // Start time in seconds (0 = beginning of file)
    double stop_time;         // Stop time in seconds (-1 = end of file)
    int start_sample_index;   // Start sample index (-1 = use start_time instead)
    int stop_sample_index;    // Stop sample index (-1 = use stop_time instead)
} DedispersionConfig;

// Dispersion parameters for all channels
typedef struct {
    double *channel_frequencies; // Array of channel frequencies [NUM_CHANNELS]
    double reference_frequency;  // Reference frequency (highest)
    int *dispersion_shifts;      // Dispersion shifts in samples [NUM_CHANNELS]
    int max_shift;               // Maximum shift in samples
    int total_time_samples;      // Total time samples detected from file size
} DispersionParams;

// Optimized Structure of Arrays (SoA) layout for better vectorization
// Separate arrays for each polarization to enable better SIMD access
typedef struct {
    float **antenna_power_pol0;  // [antenna][time] for polarization 0
    float **antenna_power_pol1;  // [antenna][time] for polarization 1
    int num_output_samples;      // Number of output time samples
} AntennaData;

// Spectra data structure for frequency-resolved output (single subband)
typedef struct {
    float ***spectra_stokes_i;   // [antenna][channel][time] for Stokes I (pol0 + pol1)
    int num_output_samples;      // Number of output time samples
    int num_channels;            // Number of frequency channels in this subband
    int subband_index;           // Which subband this data represents
} SpectraData;

// Raw voltage data structure for complex voltage time streams
typedef struct {
    float ****raw_voltages;      // [antenna][channel][pol][time] - complex voltage data (real/imag interleaved)
    int num_antennas;           // Number of antennas
    int num_channels;           // Number of frequency channels per subband
    int num_output_samples;     // Number of time samples
    int subband_index;          // Which subband this data represents
} RawVoltageData;

// Channel data organized for vectorized processing
typedef struct {
    float *channel_powers;       // Flat array: [channel * time + time_idx]
    int num_channels;           // Number of channels in this structure
    int num_time_samples;       // Number of time samples
} ChannelData;

// ============================================================================
// SYSTEM MONITORING AND THREAD MANAGEMENT
// ============================================================================
// Get system load average
double get_system_load() {
    FILE* loadavg_file = fopen("/proc/loadavg", "r");
    if (!loadavg_file) return -1.0;
    
    double load1, load5, load15;
    int result = fscanf(loadavg_file, "%lf %lf %lf", &load1, &load5, &load15);
    fclose(loadavg_file);
    
    return (result == 3) ? load1 : -1.0;
}

// Get total number of CPU cores
int get_cpu_count() {
    #ifdef _OPENMP
    return omp_get_num_procs();
    #else
    return (int)sysconf(_SC_NPROCESSORS_ONLN);
    #endif
}

// Get available memory in GB
double get_available_memory_gb() {
    FILE* meminfo = fopen("/proc/meminfo", "r");
    if (!meminfo) return -1.0;
    
    char line[256];
    long total_kb = 0, available_kb = 0, free_kb = 0, buffers_kb = 0, cached_kb = 0;
    
    while (fgets(line, sizeof(line), meminfo)) {
        if (sscanf(line, "MemTotal: %ld kB", &total_kb) == 1) continue;
        if (sscanf(line, "MemAvailable: %ld kB", &available_kb) == 1) continue;
        if (sscanf(line, "MemFree: %ld kB", &free_kb) == 1) continue;
        if (sscanf(line, "Buffers: %ld kB", &buffers_kb) == 1) continue;
        if (sscanf(line, "Cached: %ld kB", &cached_kb) == 1) continue;
    }
    fclose(meminfo);
    
    // Use MemAvailable if available (more accurate), otherwise estimate
    double available_gb;
    if (available_kb > 0) {
        available_gb = available_kb / (1024.0 * 1024.0);
    } else {
        // Fallback: Free + Buffers + Cached (conservative estimate)
        available_gb = (free_kb + buffers_kb + cached_kb) / (1024.0 * 1024.0);
    }
    
    return available_gb;
}

// Get total memory in GB
double get_total_memory_gb() {
    FILE* meminfo = fopen("/proc/meminfo", "r");
    if (!meminfo) return -1.0;
    
    char line[256];
    long total_kb = 0;
    
    while (fgets(line, sizeof(line), meminfo)) {
        if (sscanf(line, "MemTotal: %ld kB", &total_kb) == 1) {
            break;
        }
    }
    fclose(meminfo);
    
    return total_kb / (1024.0 * 1024.0);
}

// Estimate memory usage per processing unit
size_t estimate_memory_per_subband_mb(const DedispersionConfig* config, int total_time_samples) {
    size_t memory_mb = 0;
    
    if (config->enable_median_subtraction) {
        // Memory for channel data arrays: [NUM_CHANNELS_PER_SUBBAND][total_time_samples]
        size_t channel_data_bytes = (size_t)NUM_CHANNELS_PER_SUBBAND * total_time_samples * sizeof(float);
        
        // Memory for median arrays
        size_t median_data_bytes = NUM_CHANNELS_PER_SUBBAND * sizeof(float);
        
        // Add some overhead for processing buffers
        size_t overhead_bytes = channel_data_bytes * 0.1; // 10% overhead
        
        memory_mb = (channel_data_bytes + median_data_bytes + overhead_bytes) / (1024 * 1024);
    } else {
        // Much smaller memory footprint for direct processing
        // Just processing buffers and antenna data
        size_t processing_buffers = config->block_processing_size * 1024 * 1024; // 1MB per 1024 blocks
        memory_mb = processing_buffers / (1024 * 1024);
    }
    
    return memory_mb;
}

// Calculate maximum threads based on memory constraints
int calculate_memory_limited_threads(double available_gb, size_t memory_per_subband_mb, 
                                   const DedispersionConfig* config) {
    // Convert to MB for easier calculation
    double available_mb = available_gb * 1024.0;
    
    // Reserve memory for system and other processes 
    // Use a more reasonable reservation: min(25% of total, max(4GB, 10% of available))
    double total_gb = get_total_memory_gb();
    double conservative_reserve = total_gb * 0.25;  // 25% of total
    double reasonable_reserve = (available_gb * 0.1 > 4.0) ? available_gb * 0.1 : 4.0;  // 10% of available or 4GB
    double reserved_gb = (conservative_reserve < reasonable_reserve) ? conservative_reserve : reasonable_reserve;
    double usable_mb = available_mb - (reserved_gb * 1024.0);
    
    printf("Memory calculation debug:\n");
    printf("  Available: %.1f GB, Total: %.1f GB\n", available_gb, total_gb);
    printf("  Conservative reserve: %.1f GB, Reasonable reserve: %.1f GB\n", conservative_reserve, reasonable_reserve);
    printf("  Using reserved: %.1f GB, Usable: %.1f GB\n", reserved_gb, usable_mb / 1024.0);
    
    if (usable_mb <= 0) {
        printf("WARNING: Very low memory available, using single thread\n");
        return 1;
    }
    
    int max_threads;
    
    if (config->enable_median_subtraction) {
        // For median subtraction, threads don't process subbands in parallel,
        // but the total memory requirement is fixed per subband
        // Threading is at the channel/antenna level within subbands
        
        // Total memory needed is: 16 subbands * memory_per_subband
        size_t total_memory_needed_mb = NUM_SUBBANDS * memory_per_subband_mb;
        
        if (total_memory_needed_mb > usable_mb) {
            printf("WARNING: Insufficient memory for median subtraction mode\n");
            printf("  Required: %.1f GB, Available: %.1f GB\n", 
                   total_memory_needed_mb / 1024.0, usable_mb / 1024.0);
            return 1;
        }
        
        // Memory is not per-thread in this case, so don't limit threads based on memory
        // The limitation comes from the fixed memory requirement
        max_threads = 256; // Effectively unlimited from memory perspective
        
    } else {
        // For direct processing, threads could process subbands in parallel
        // Each thread might need some buffer space
        size_t buffer_per_thread_mb = 100; // Conservative estimate for buffers
        max_threads = (int)(usable_mb / buffer_per_thread_mb);
    }
    
    if (max_threads < 1) max_threads = 1;
    
    return max_threads;
}

// Smart thread count determination based on system state AND memory
int determine_optimal_threads(int requested_threads, int max_cores, 
                            const DedispersionConfig* config, int total_time_samples) {
    int total_cores = get_cpu_count();
    double system_load = get_system_load();
    double available_memory = get_available_memory_gb();
    
    // Conservative defaults: leave some cores for system
    int conservative_max = (total_cores * 3) / 4;  // Use 75% of cores
    int safe_max = total_cores - 4;  // Leave 4 cores for system
    if (safe_max < 1) safe_max = 1;
    
    // Calculate memory-limited threads
    size_t memory_per_subband = estimate_memory_per_subband_mb(config, total_time_samples);
    int memory_limited = calculate_memory_limited_threads(available_memory, memory_per_subband, config);
    
    int cpu_optimal;
    
    if (requested_threads == 0) {
        // Auto-detect mode
        if (system_load < 0) {
            // Can't read load, be conservative
            cpu_optimal = conservative_max;
        } else if (system_load < total_cores * 0.25) {
            // System is idle, can use more cores
            cpu_optimal = safe_max;
        } else if (system_load < total_cores * 0.5) {
            // System is moderately loaded
            cpu_optimal = conservative_max;
        } else {
            // System is busy, be very conservative
            cpu_optimal = total_cores / 4;
            if (cpu_optimal < 1) cpu_optimal = 1;
        }
    } else {
        // Use requested number, but cap it for safety
        cpu_optimal = requested_threads;
    }
    
    // Apply maximum limits (CPU-based)
    if (max_cores > 0 && cpu_optimal > max_cores) {
        cpu_optimal = max_cores;
    }
    if (cpu_optimal > safe_max) {
        cpu_optimal = safe_max;
    }
    
    // Apply memory-based limitation (most restrictive wins)
    int final_threads = (cpu_optimal < memory_limited) ? cpu_optimal : memory_limited;
    
    if (final_threads < 1) {
        final_threads = 1;
    }
    
    // Print memory analysis
    printf("Memory analysis:\n");
    printf("  Total RAM: %.1f GB\n", get_total_memory_gb());
    printf("  Available RAM: %.1f GB\n", available_memory);
    printf("  Memory per subband: %zu MB\n", memory_per_subband);
    if (config->enable_median_subtraction) {
        printf("  Total memory needed: %.1f GB (all subbands)\n", 
               (NUM_SUBBANDS * memory_per_subband) / 1024.0);
    }
    printf("  CPU-limited threads: %d\n", cpu_optimal);
    printf("  Memory-limited threads: %d\n", memory_limited);
    printf("  Final threads: %d\n", final_threads);
    
    if (final_threads < cpu_optimal) {
        printf("  NOTE: Thread count limited by memory constraints\n");
    }
    
    return final_threads;
}

// Set up OpenMP environment
void setup_openmp_environment(int num_threads) {
    #ifdef _OPENMP
    omp_set_num_threads(num_threads);
    omp_set_dynamic(0);  // Disable dynamic thread adjustment
    
    // Set thread affinity if possible (helps with NUMA)
    if (getenv("OMP_PROC_BIND") == NULL) {
        putenv("OMP_PROC_BIND=spread");
    }
    if (getenv("OMP_PLACES") == NULL) {
        putenv("OMP_PLACES=cores");
    }
    
    printf("OpenMP configured: %d threads (max available: %d)\n", 
           num_threads, omp_get_max_threads());
    #else
    (void)num_threads; // Suppress unused parameter warning
    printf("OpenMP not available, using single thread\n");
    #endif
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================
// Generate input and output directories from candidate name
int setup_directories_from_candidate(DedispersionConfig* config) {
    if (!config->candidate_name) {
        return 0; // No candidate name provided, use manual directories
    }
    
    // Base path for DSA110 candidates
    const char* base_path = "/dataz/dsa110/candidates";
    
    // Allocate memory for directory paths
    config->input_directory = malloc(512);
    config->output_directory = malloc(512);
    
    if (!config->input_directory || !config->output_directory) {
        fprintf(stderr, "Memory allocation failed for directory paths\n");
        return -1;
    }
    
    // Generate input directory: /dataz/dsa110/candidates/{candidate_name}/Level2/voltages
    snprintf(config->input_directory, 512, "%s/%s/Level2/voltages", base_path, config->candidate_name);
    
    // Generate output directory: {candidate_name}_out
    snprintf(config->output_directory, 512, "%s_out", config->candidate_name);
    
    // Check if input directory exists
    struct stat statbuf;
    if (stat(config->input_directory, &statbuf) != 0 || !S_ISDIR(statbuf.st_mode)) {
        fprintf(stderr, "Error: Input directory does not exist: %s\n", config->input_directory);
        fprintf(stderr, "Please check that candidate '%s' exists and has Level2/voltages data\n", config->candidate_name);
        return -1;
    }
    
    printf("Auto-generated directories from candidate '%s':\n", config->candidate_name);
    printf("  Input:  %s\n", config->input_directory);
    printf("  Output: %s\n", config->output_directory);
    
    return 1; // Success, directories were generated
}

// Cleanup dynamically allocated directory paths
void cleanup_config_directories(DedispersionConfig* config) {
    if (config->candidate_name) {
        // Only free if they were dynamically allocated by setup_directories_from_candidate
        free(config->input_directory);
        free(config->output_directory);
        config->input_directory = NULL;
        config->output_directory = NULL;
    }
}

// Ensure a directory exists, create if missing
void ensure_directory_exists(const char* directory_path) {
    struct stat statbuf = {0};
    if (stat(directory_path, &statbuf) == -1) mkdir(directory_path, 0700);
}

// Print a progress bar to the terminal with description
void print_progress_bar(const char* description, double fraction_complete, int bar_width) {
    int percent = (int)(fraction_complete * 100);
    int filled = (int)(fraction_complete * bar_width);
    printf("\r%s [", description);
    for (int i = 0; i < bar_width; ++i)
        putchar(i < filled ? '#' : '-');
    printf("] %3d%%", percent);
    fflush(stdout);
}

// Print a simple progress indicator
void print_progress_simple(const char* description, int current, int total) {
    printf("\r%s %d/%d (%d%%)", description, current, total, (current * 100) / total);
    fflush(stdout);
}

// Unpack a single 4-bit complex value into signed real and imaginary parts
void unpack_4bit_complex(uint8_t packed_value, int8_t* real, int8_t* imag) {
    int8_t r = (packed_value & 0x0F); if (r > 7) r -= 16;
    int8_t i = (packed_value >> 4) & 0x0F; if (i > 7) i -= 16;
    *real = r; *imag = i;
}

// ============================================================================
// VECTORIZED POWER CALCULATION
// ============================================================================
// Check CPU capabilities at runtime
int has_avx512() {
    #if defined(ARCH_X86) && defined(__AVX512F__)
    // Check if AVX-512 Foundation is available
    unsigned int eax, ebx, ecx, edx;
    __asm__ __volatile__ (
        "cpuid"
        : "=a" (eax), "=b" (ebx), "=c" (ecx), "=d" (edx)
        : "a" (7), "c" (0)
    );
    return (ebx & (1 << 16)) != 0; // AVX-512F bit
    #else
    return 0;
    #endif
}

int has_avx2() {
    #if defined(ARCH_X86) && defined(__AVX2__)
    return 1;
    #else
    return 0;
    #endif
}

int has_sse2() {
    #if defined(ARCH_X86) && defined(__SSE2__)
    return 1;
    #else
    return 0;
    #endif
}

int has_neon() {
    #if defined(ARCH_ARM)
    return 1;  // Most modern ARM processors have NEON
    #else
    return 0;
    #endif
}

// Vectorized power calculation using AVX-512 (processes 32 complex samples at once)
void calculate_power_avx512(const uint8_t* input, float* output, int n_samples) {
    #ifdef __AVX512F__
    const int vector_size = 32; // Process 32 bytes (32 complex samples) at once
    int vectorized_samples = (n_samples / vector_size) * vector_size;
    
    // Process vectorized portion
    for (int i = 0; i < vectorized_samples; i += vector_size) {
        // Load 32 bytes (32 complex 4-bit samples)
        __m256i packed = _mm256_loadu_si256((__m256i*)(input + i));
        
        // Split into two 16-byte chunks for processing
        __m128i packed_lo = _mm256_extracti128_si256(packed, 0);
        __m128i packed_hi = _mm256_extracti128_si256(packed, 1);
        
        // Unpack lower 16 bytes
        __m256i unpacked_lo = _mm256_cvtepu8_epi16(packed_lo);
        __m256i unpacked_hi = _mm256_cvtepu8_epi16(packed_hi);
        
        // Extract real and imaginary parts using AVX-512
        __m512i real_parts = _mm512_inserti64x4(_mm512_castsi256_si512(
            _mm256_and_si256(unpacked_lo, _mm256_set1_epi16(0x0F))), 
            _mm256_and_si256(unpacked_hi, _mm256_set1_epi16(0x0F)), 1);
        
        __m512i imag_parts = _mm512_inserti64x4(_mm512_castsi256_si512(
            _mm256_and_si256(_mm256_srli_epi16(unpacked_lo, 4), _mm256_set1_epi16(0x0F))), 
            _mm256_and_si256(_mm256_srli_epi16(unpacked_hi, 4), _mm256_set1_epi16(0x0F)), 1);
        
        // Convert to signed (subtract 8 if > 7) using mask
        __mmask32 mask_real = _mm512_cmpgt_epi16_mask(real_parts, _mm512_set1_epi16(7));
        __mmask32 mask_imag = _mm512_cmpgt_epi16_mask(imag_parts, _mm512_set1_epi16(7));
        
        real_parts = _mm512_mask_sub_epi16(real_parts, mask_real, real_parts, _mm512_set1_epi16(16));
        imag_parts = _mm512_mask_sub_epi16(imag_parts, mask_imag, imag_parts, _mm512_set1_epi16(16));
        
        // Convert to 32-bit for multiplication (split into two 256-bit vectors)
        __m512i real_32_lo = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(real_parts, 0));
        __m512i real_32_hi = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(real_parts, 1));
        __m512i imag_32_lo = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(imag_parts, 0));
        __m512i imag_32_hi = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(imag_parts, 1));
        
        // Calculate power using FMA: real² + imag² 
        __m512 real_float_lo = _mm512_cvtepi32_ps(real_32_lo);
        __m512 real_float_hi = _mm512_cvtepi32_ps(real_32_hi);
        __m512 imag_float_lo = _mm512_cvtepi32_ps(imag_32_lo);
        __m512 imag_float_hi = _mm512_cvtepi32_ps(imag_32_hi);
        
        // Use FMA for power calculation: power = real * real + imag * imag
        __m512 power_lo = _mm512_fmadd_ps(real_float_lo, real_float_lo, 
                                         _mm512_mul_ps(imag_float_lo, imag_float_lo));
        __m512 power_hi = _mm512_fmadd_ps(real_float_hi, real_float_hi, 
                                         _mm512_mul_ps(imag_float_hi, imag_float_hi));
        
        // Store results
        _mm512_storeu_ps(output + i, power_lo);
        _mm512_storeu_ps(output + i + 16, power_hi);
    }
    
    // Handle remaining samples with scalar code
    for (int i = vectorized_samples; i < n_samples; i++) {
        int8_t real, imag;
        unpack_4bit_complex(input[i], &real, &imag);
        output[i] = (float)real * (float)real + (float)imag * (float)imag;
    }
    #else
    // Fallback to scalar implementation
    for (int i = 0; i < n_samples; i++) {
        int8_t real, imag;
        unpack_4bit_complex(input[i], &real, &imag);
        output[i] = (float)real * (float)real + (float)imag * (float)imag;
    }
    #endif
}

// Vectorized power calculation using AVX2 (processes 16 complex samples at once)
void calculate_power_avx2(const uint8_t* input, float* output, int n_samples) {
    #ifdef __AVX2__
    const int vector_size = 16; // Process 16 bytes (16 complex samples) at once
    int vectorized_samples = (n_samples / vector_size) * vector_size;
    
    // Process vectorized portion
    for (int i = 0; i < vectorized_samples; i += vector_size) {
        // Load 16 bytes (16 complex 4-bit samples)
        __m128i packed = _mm_loadu_si128((__m128i*)(input + i));
        
        // Unpack lower 8 bytes
        __m128i unpacked_lo = _mm_unpacklo_epi8(packed, _mm_setzero_si128());
        __m128i unpacked_hi = _mm_unpackhi_epi8(packed, _mm_setzero_si128());
        
        // Extract real and imaginary parts for lower half
        __m128i real_lo = _mm_and_si128(unpacked_lo, _mm_set1_epi16(0x0F));
        __m128i imag_lo = _mm_and_si128(_mm_srli_epi16(unpacked_lo, 4), _mm_set1_epi16(0x0F));
        
        // Extract real and imaginary parts for upper half
        __m128i real_hi = _mm_and_si128(unpacked_hi, _mm_set1_epi16(0x0F));
        __m128i imag_hi = _mm_and_si128(_mm_srli_epi16(unpacked_hi, 4), _mm_set1_epi16(0x0F));
        
        // Convert to signed (subtract 8 if > 7)
        __m128i mask_lo_real = _mm_cmpgt_epi16(real_lo, _mm_set1_epi16(7));
        __m128i mask_lo_imag = _mm_cmpgt_epi16(imag_lo, _mm_set1_epi16(7));
        __m128i mask_hi_real = _mm_cmpgt_epi16(real_hi, _mm_set1_epi16(7));
        __m128i mask_hi_imag = _mm_cmpgt_epi16(imag_hi, _mm_set1_epi16(7));
        
        real_lo = _mm_sub_epi16(real_lo, _mm_and_si128(mask_lo_real, _mm_set1_epi16(16)));
        imag_lo = _mm_sub_epi16(imag_lo, _mm_and_si128(mask_lo_imag, _mm_set1_epi16(16)));
        real_hi = _mm_sub_epi16(real_hi, _mm_and_si128(mask_hi_real, _mm_set1_epi16(16)));
        imag_hi = _mm_sub_epi16(imag_hi, _mm_and_si128(mask_hi_imag, _mm_set1_epi16(16)));
        
        // Convert to 32-bit integers for multiplication
        __m256i real_32_lo = _mm256_cvtepi16_epi32(real_lo);
        __m256i imag_32_lo = _mm256_cvtepi16_epi32(imag_lo);
        __m256i real_32_hi = _mm256_cvtepi16_epi32(real_hi);
        __m256i imag_32_hi = _mm256_cvtepi16_epi32(imag_hi);
        
        // Calculate power: real² + imag²
        __m256i real_sq_lo = _mm256_mullo_epi32(real_32_lo, real_32_lo);
        __m256i imag_sq_lo = _mm256_mullo_epi32(imag_32_lo, imag_32_lo);
        __m256i real_sq_hi = _mm256_mullo_epi32(real_32_hi, real_32_hi);
        __m256i imag_sq_hi = _mm256_mullo_epi32(imag_32_hi, imag_32_hi);
        
        __m256i power_lo = _mm256_add_epi32(real_sq_lo, imag_sq_lo);
        __m256i power_hi = _mm256_add_epi32(real_sq_hi, imag_sq_hi);
        
        // Convert to float and store
        __m256 power_float_lo = _mm256_cvtepi32_ps(power_lo);
        __m256 power_float_hi = _mm256_cvtepi32_ps(power_hi);
        
        _mm256_storeu_ps(output + i, power_float_lo);
        _mm256_storeu_ps(output + i + 8, power_float_hi);
    }
    
    // Handle remaining samples with scalar code
    for (int i = vectorized_samples; i < n_samples; i++) {
        int8_t real, imag;
        unpack_4bit_complex(input[i], &real, &imag);
        output[i] = (float)real * (float)real + (float)imag * (float)imag;
    }
    #else
    // Fallback to scalar implementation
    for (int i = 0; i < n_samples; i++) {
        int8_t real, imag;
        unpack_4bit_complex(input[i], &real, &imag);
        output[i] = (float)real * (float)real + (float)imag * (float)imag;
    }
    #endif
}

// Vectorized power calculation using SSE2 (processes 8 complex samples at once)
void calculate_power_sse2(const uint8_t* input, float* output, int n_samples) {
    #if defined(ARCH_X86) && defined(__SSE2__)
    const int vector_size = 8; // Process 8 bytes (8 complex samples) at once
    int vectorized_samples = (n_samples / vector_size) * vector_size;
    
    // Process vectorized portion
    for (int i = 0; i < vectorized_samples; i += vector_size) {
        // Load 8 bytes (8 complex 4-bit samples)
        __m128i packed = _mm_loadl_epi64((__m128i*)(input + i));
        
        // Unpack to 16-bit
        __m128i unpacked = _mm_unpacklo_epi8(packed, _mm_setzero_si128());
        
        // Extract real and imaginary parts
        __m128i real = _mm_and_si128(unpacked, _mm_set1_epi16(0x0F));
        __m128i imag = _mm_and_si128(_mm_srli_epi16(unpacked, 4), _mm_set1_epi16(0x0F));
        
        // Convert to signed (subtract 8 if > 7)
        __m128i mask_real = _mm_cmpgt_epi16(real, _mm_set1_epi16(7));
        __m128i mask_imag = _mm_cmpgt_epi16(imag, _mm_set1_epi16(7));
        
        real = _mm_sub_epi16(real, _mm_and_si128(mask_real, _mm_set1_epi16(16)));
        imag = _mm_sub_epi16(imag, _mm_and_si128(mask_imag, _mm_set1_epi16(16)));
        
        // Calculate power using SSE2-compatible operations
        // Use _mm_madd_epi16 for squaring: madd(x, x) = x*x + 0*0
        __m128i real_sq_lo = _mm_madd_epi16(_mm_unpacklo_epi16(real, _mm_setzero_si128()),
                                           _mm_unpacklo_epi16(real, _mm_setzero_si128()));
        __m128i real_sq_hi = _mm_madd_epi16(_mm_unpackhi_epi16(real, _mm_setzero_si128()),
                                           _mm_unpackhi_epi16(real, _mm_setzero_si128()));
        __m128i imag_sq_lo = _mm_madd_epi16(_mm_unpacklo_epi16(imag, _mm_setzero_si128()),
                                           _mm_unpacklo_epi16(imag, _mm_setzero_si128()));
        __m128i imag_sq_hi = _mm_madd_epi16(_mm_unpackhi_epi16(imag, _mm_setzero_si128()),
                                           _mm_unpackhi_epi16(imag, _mm_setzero_si128()));
        
        __m128i power_lo = _mm_add_epi32(real_sq_lo, imag_sq_lo);
        __m128i power_hi = _mm_add_epi32(real_sq_hi, imag_sq_hi);
        
        // Convert to float and store
        __m128 power_float_lo = _mm_cvtepi32_ps(power_lo);
        __m128 power_float_hi = _mm_cvtepi32_ps(power_hi);
        
        _mm_storeu_ps(output + i, power_float_lo);
        _mm_storeu_ps(output + i + 4, power_float_hi);
    }
    
    // Handle remaining samples with scalar code
    for (int i = vectorized_samples; i < n_samples; i++) {
        int8_t real, imag;
        unpack_4bit_complex(input[i], &real, &imag);
        output[i] = (float)real * (float)real + (float)imag * (float)imag;
    }
    #else
    // Fallback to scalar implementation
    for (int i = 0; i < n_samples; i++) {
        int8_t real, imag;
        unpack_4bit_complex(input[i], &real, &imag);
        output[i] = (float)real * (float)real + (float)imag * (float)imag;
    }
    #endif
}

// Vectorized power calculation using ARM NEON (processes 16 complex samples at once)
void calculate_power_neon(const uint8_t* input, float* output, int n_samples) {
    #if defined(ARCH_ARM)
    const int vector_size = 16; // Process 16 bytes (16 complex samples) at once  
    int vectorized_samples = (n_samples / vector_size) * vector_size;
    
    // Process vectorized portion
    for (int i = 0; i < vectorized_samples; i += vector_size) {
        // Load 16 bytes (16 complex 4-bit samples)
        uint8x16_t packed = vld1q_u8(input + i);
        
        // Extract real and imaginary parts (4-bit each)
        uint8x16_t real_u8 = vandq_u8(packed, vdupq_n_u8(0x0F));
        uint8x16_t imag_u8 = vandq_u8(vshrq_n_u8(packed, 4), vdupq_n_u8(0x0F));
        
        // Convert to signed 8-bit (subtract 8 if > 7)
        int8x16_t real_s8 = vreinterpretq_s8_u8(vsubq_u8(real_u8, vandq_u8(vcgtq_u8(real_u8, vdupq_n_u8(7)), vdupq_n_u8(16))));
        int8x16_t imag_s8 = vreinterpretq_s8_u8(vsubq_u8(imag_u8, vandq_u8(vcgtq_u8(imag_u8, vdupq_n_u8(7)), vdupq_n_u8(16))));
        
        // Convert to 16-bit for processing
        int16x8_t real_lo = vmovl_s8(vget_low_s8(real_s8));
        int16x8_t real_hi = vmovl_s8(vget_high_s8(real_s8));
        int16x8_t imag_lo = vmovl_s8(vget_low_s8(imag_s8));
        int16x8_t imag_hi = vmovl_s8(vget_high_s8(imag_s8));
        
        // Calculate power: real² + imag²
        int32x4_t real_sq_lo = vmull_s16(vget_low_s16(real_lo), vget_low_s16(real_lo));
        int32x4_t real_sq_hi = vmull_s16(vget_high_s16(real_lo), vget_high_s16(real_lo));
        int32x4_t imag_sq_lo = vmull_s16(vget_low_s16(imag_lo), vget_low_s16(imag_lo));
        int32x4_t imag_sq_hi = vmull_s16(vget_high_s16(imag_lo), vget_high_s16(imag_lo));
        
        int32x4_t power_0_3 = vaddq_s32(real_sq_lo, imag_sq_lo);
        int32x4_t power_4_7 = vaddq_s32(real_sq_hi, imag_sq_hi);
        
        // Second half
        real_sq_lo = vmull_s16(vget_low_s16(real_hi), vget_low_s16(real_hi));
        real_sq_hi = vmull_s16(vget_high_s16(real_hi), vget_high_s16(real_hi));
        imag_sq_lo = vmull_s16(vget_low_s16(imag_hi), vget_low_s16(imag_hi));
        imag_sq_hi = vmull_s16(vget_high_s16(imag_hi), vget_high_s16(imag_hi));
        
        int32x4_t power_8_11 = vaddq_s32(real_sq_lo, imag_sq_lo);
        int32x4_t power_12_15 = vaddq_s32(real_sq_hi, imag_sq_hi);
        
        // Convert to float and store
        vst1q_f32(output + i,      vcvtq_f32_s32(power_0_3));
        vst1q_f32(output + i + 4,  vcvtq_f32_s32(power_4_7));
        vst1q_f32(output + i + 8,  vcvtq_f32_s32(power_8_11));
        vst1q_f32(output + i + 12, vcvtq_f32_s32(power_12_15));
    }
    
    // Handle remaining samples with scalar code
    for (int i = vectorized_samples; i < n_samples; i++) {
        int8_t real, imag;
        unpack_4bit_complex(input[i], &real, &imag);
        output[i] = (float)real * (float)real + (float)imag * (float)imag;
    }
    #else
    // Fallback to scalar implementation
    for (int i = 0; i < n_samples; i++) {
        int8_t real, imag;
        unpack_4bit_complex(input[i], &real, &imag);
        output[i] = (float)real * (float)real + (float)imag * (float)imag;
    }
    #endif
}

// Main vectorized power calculation function with runtime CPU detection
void calculate_power_vectorized(const uint8_t* input, float* output, int n_samples) {
    if (has_avx512()) {
        calculate_power_avx512(input, output, n_samples);
    } else if (has_avx2()) {
        calculate_power_avx2(input, output, n_samples);
    } else if (has_sse2()) {
        calculate_power_sse2(input, output, n_samples);
    } else if (has_neon()) {
        calculate_power_neon(input, output, n_samples);
    } else {
        // Fallback to scalar implementation
        for (int i = 0; i < n_samples; i++) {
            int8_t real, imag;
            unpack_4bit_complex(input[i], &real, &imag);
            output[i] = (float)real * (float)real + (float)imag * (float)imag;
        }
    }
}

// Optimized function to process a block of data with vectorization using SoA layout
void process_block_vectorized(uint8_t *block_data, int antenna_index, int time_in_block,
                              ChannelData* channel_data, int sample_count, int subband_index,
                              const DispersionParams* dispersion, AntennaData* antenna_data,
                              SpectraData* spectra_data, RawVoltageData* raw_voltage_data,
                              const DedispersionConfig* config, int global_time_index, 
                              int start_sample, int stop_sample) {
    
    // Process all channels for this antenna using vectorized operations
    const int vector_size = has_avx512() ? 16 : (has_avx2() ? 8 : (has_neon() ? 16 : 4));
    int full_vectors = NUM_CHANNELS_PER_SUBBAND / vector_size;
    
    // Pre-calculate base offset for this antenna and time
    size_t base_offset = (size_t)antenna_index * NUM_CHANNELS_PER_SUBBAND * SAMPLES_PER_BLOCK * NUM_POLARIZATIONS + 
                         (size_t)time_in_block * NUM_POLARIZATIONS;
    
    // Process channels in vectorized groups
    for (int vec = 0; vec < full_vectors; vec++) {
        int start_channel = vec * vector_size;
        
        // Prepare input data for vectorized processing
        uint8_t pol0_data[vector_size];
        uint8_t pol1_data[vector_size];
        
        // Gather data for vectorized processing
        for (int i = 0; i < vector_size; i++) {
            size_t channel_offset = base_offset + (size_t)(start_channel + i) * SAMPLES_PER_BLOCK * NUM_POLARIZATIONS;
            pol0_data[i] = block_data[channel_offset];
            pol1_data[i] = block_data[channel_offset + 1];
        }
        
        // Vectorized power calculation (skip for raw mode)
        float pol0_powers[vector_size];
        float pol1_powers[vector_size];
        if (config->output_mode != OUTPUT_MODE_RAW) {
            calculate_power_vectorized(pol0_data, pol0_powers, vector_size);
            calculate_power_vectorized(pol1_data, pol1_powers, vector_size);
        }
        
        // Store results using SoA layout
        for (int i = 0; i < vector_size; i++) {
            int channel_in_subband = start_channel + i;
            int global_channel_index = subband_index * NUM_CHANNELS_PER_SUBBAND + channel_in_subband;
            int dispersion_shift = config->enable_dedispersion ? dispersion->dispersion_shifts[global_channel_index] : 0;
            int output_time_index = global_time_index - dispersion_shift;
            
            // CRITICAL BUG FIX: Check if this time sample falls within valid dedispersed range
            // After dedispersion, samples beyond (total_samples - 1 - max_shift) are invalid
            int max_valid_time = dispersion->total_time_samples - 1 - dispersion->max_shift;
            if (config->enable_dedispersion && (output_time_index < 0 || output_time_index > max_valid_time)) {
                continue; // Skip invalid samples that fall outside dedispersed range
            }
            
            // Check if this time sample falls within our slice range
            if (output_time_index < start_sample || output_time_index > stop_sample) {
                continue; // Skip samples outside the requested time range
            }
            
            // Map global output time index to local array index
            int local_time_index = output_time_index - start_sample;
            
            if (config->enable_median_subtraction && channel_data) {
                // Store for median calculation (sum of both polarizations)
                size_t flat_idx = (size_t)channel_in_subband * channel_data->num_time_samples + sample_count;
                channel_data->channel_powers[flat_idx] += pol0_powers[i] + pol1_powers[i];
            } else if (config->output_mode == OUTPUT_MODE_SPECTRA) {
                // Store per-channel data for spectra mode (Stokes I = pol0 + pol1)
                if (spectra_data && local_time_index >= 0 && local_time_index < spectra_data->num_output_samples) {
                    spectra_data->spectra_stokes_i[antenna_index][channel_in_subband][local_time_index] += pol0_powers[i] + pol1_powers[i];
                }
            } else if (config->output_mode == OUTPUT_MODE_RAW) {
                // Store raw complex voltage data for both polarizations
                if (raw_voltage_data && local_time_index >= 0 && local_time_index < raw_voltage_data->num_output_samples) {
                    // Extract complex voltages from packed data
                    int8_t pol0_real, pol0_imag, pol1_real, pol1_imag;
                    unpack_4bit_complex(pol0_data[i], &pol0_real, &pol0_imag);
                    unpack_4bit_complex(pol1_data[i], &pol1_real, &pol1_imag);
                    
                    // Store as float complex pairs [real, imag] for each polarization
                    raw_voltage_data->raw_voltages[antenna_index][channel_in_subband][0][local_time_index * 2] = (float)pol0_real;
                    raw_voltage_data->raw_voltages[antenna_index][channel_in_subband][0][local_time_index * 2 + 1] = (float)pol0_imag;
                    raw_voltage_data->raw_voltages[antenna_index][channel_in_subband][1][local_time_index * 2] = (float)pol1_real;
                    raw_voltage_data->raw_voltages[antenna_index][channel_in_subband][1][local_time_index * 2 + 1] = (float)pol1_imag;
                }
            } else {
                // Direct accumulation to antenna data (TIME_SERIES mode)
                if (local_time_index >= 0 && local_time_index < antenna_data->num_output_samples) {
                    antenna_data->antenna_power_pol0[antenna_index][local_time_index] += pol0_powers[i];
                    antenna_data->antenna_power_pol1[antenna_index][local_time_index] += pol1_powers[i];
                }
            }
        }
    }
    
    // Handle remaining channels (scalar processing)
    for (int channel_in_subband = full_vectors * vector_size; channel_in_subband < NUM_CHANNELS_PER_SUBBAND; channel_in_subband++) {
        float pol0_power = 0.0f, pol1_power = 0.0f;
        
        // Only compute power if not in raw mode
        if (config->output_mode != OUTPUT_MODE_RAW) {
            for (int pol = 0; pol < NUM_POLARIZATIONS; pol++) {
                size_t offset = base_offset + (size_t)channel_in_subband * SAMPLES_PER_BLOCK * NUM_POLARIZATIONS + pol;
                
                int8_t real_part, imag_part;
                unpack_4bit_complex(block_data[offset], &real_part, &imag_part);
                float power = (float)real_part * (float)real_part + (float)imag_part * (float)imag_part;
                
                if (pol == 0) pol0_power = power;
                else pol1_power = power;
            }
        }
        
        int global_channel_index = subband_index * NUM_CHANNELS_PER_SUBBAND + channel_in_subband;
        int dispersion_shift = config->enable_dedispersion ? dispersion->dispersion_shifts[global_channel_index] : 0;
        int output_time_index = global_time_index - dispersion_shift;
        
        // CRITICAL BUG FIX: Check if this time sample falls within valid dedispersed range
        // After dedispersion, samples beyond (total_samples - 1 - max_shift) are invalid
        int max_valid_time = dispersion->total_time_samples - 1 - dispersion->max_shift;
        if (config->enable_dedispersion && (output_time_index < 0 || output_time_index > max_valid_time)) {
            continue; // Skip invalid samples that fall outside dedispersed range
        }
        
        // Check if this time sample falls within our slice range
        if (output_time_index < start_sample || output_time_index > stop_sample) {
            continue; // Skip samples outside the requested time range
        }
        
        // Map global output time index to local array index
        int local_time_index = output_time_index - start_sample;
        
        if (config->enable_median_subtraction && channel_data) {
            size_t flat_idx = (size_t)channel_in_subband * channel_data->num_time_samples + sample_count;
            channel_data->channel_powers[flat_idx] += pol0_power + pol1_power;
        } else if (config->output_mode == OUTPUT_MODE_SPECTRA) {
            // Store per-channel data for spectra mode (Stokes I = pol0 + pol1)
            if (spectra_data && local_time_index >= 0 && local_time_index < spectra_data->num_output_samples) {
                spectra_data->spectra_stokes_i[antenna_index][channel_in_subband][local_time_index] += pol0_power + pol1_power;
            }
        } else if (config->output_mode == OUTPUT_MODE_RAW) {
            // Store raw complex voltage data for both polarizations
            if (raw_voltage_data && local_time_index >= 0 && local_time_index < raw_voltage_data->num_output_samples) {
                // Extract complex voltages for both polarizations 
                for (int pol = 0; pol < NUM_POLARIZATIONS; pol++) {
                    size_t offset = base_offset + (size_t)channel_in_subband * SAMPLES_PER_BLOCK * NUM_POLARIZATIONS + pol;
                    int8_t real_part, imag_part;
                    unpack_4bit_complex(block_data[offset], &real_part, &imag_part);
                    
                    // Store as float complex pairs [real, imag]
                    raw_voltage_data->raw_voltages[antenna_index][channel_in_subband][pol][local_time_index * 2] = (float)real_part;
                    raw_voltage_data->raw_voltages[antenna_index][channel_in_subband][pol][local_time_index * 2 + 1] = (float)imag_part;
                }
            }
        } else {
            // Direct accumulation to antenna data (TIME_SERIES mode)
            if (local_time_index >= 0 && local_time_index < antenna_data->num_output_samples) {
                antenna_data->antenna_power_pol0[antenna_index][local_time_index] += pol0_power;
                antenna_data->antenna_power_pol1[antenna_index][local_time_index] += pol1_power;
            }
        }
    }
}


// Forward declarations
float quickselect(float *arr, int left, int right, int k);
int partition(float *arr, int left, int right);
float calculate_median(float *data, int n);
float fast_median(float *data, int n);
void vectorized_median_calculation(float* data, int n, float* result);

// Comparison function for sorting floats
int compare_floats(const void *a, const void *b) {
    float fa = *(const float*)a;
    float fb = *(const float*)b;
    return (fa > fb) - (fa < fb);
}

// Vectorized median calculation for large datasets using SIMD sorting networks
void vectorized_median_calculation(float* data, int n, float* result) {
    #ifdef __AVX512F__
    if (n >= 16 && has_avx512()) {
        // For large datasets, use vectorized sorting on chunks and combine
        const int chunk_size = 16;
        int num_chunks = n / chunk_size;
        float chunk_medians[num_chunks];
        
        for (int chunk = 0; chunk < num_chunks; chunk++) {
            float* chunk_data = data + chunk * chunk_size;
            
            // Load 16 floats into AVX-512 register
            __m512 values = _mm512_loadu_ps(chunk_data);
            
            // Use AVX-512 sorting network for 16 elements (simplified approach)
            // This is a partial implementation - full sorting network would be more complex
            __m512 sorted = values; // Placeholder - would implement full bitonic sort
            
            // Store sorted chunk back
            _mm512_storeu_ps(chunk_data, sorted);
            
            // Get median of this chunk (middle element after sort)
            chunk_medians[chunk] = chunk_data[8]; // Middle of 16 elements
        }
        
        // Calculate median of chunk medians
        *result = calculate_median(chunk_medians, num_chunks);
        
        // Handle remaining elements
        if (n % chunk_size != 0) {
            float remaining_median = calculate_median(data + num_chunks * chunk_size, n % chunk_size);
            *result = (*result + remaining_median) / 2.0f; // Simple combination
        }
        
        return;
    }
    #endif
    
    // Fallback to standard median calculation
    *result = calculate_median(data, n);
}

// Partition function for quickselect
int partition(float *arr, int left, int right) {
    float pivot = arr[right];
    int i = left;
    
    for (int j = left; j < right; j++) {
        if (arr[j] <= pivot) {
            float temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
            i++;
        }
    }
    
    float temp = arr[i];
    arr[i] = arr[right];
    arr[right] = temp;
    
    return i;
}

// Quickselect algorithm for finding kth smallest element
float quickselect(float *arr, int left, int right, int k) {
    if (left == right) return arr[left];
    
    int pivot_index = partition(arr, left, right);
    
    if (k == pivot_index) {
        return arr[k];
    } else if (k < pivot_index) {
        return quickselect(arr, left, pivot_index - 1, k);
    } else {
        return quickselect(arr, pivot_index + 1, right, k);
    }
}

// Fast median calculation using quickselect algorithm (faster than full sort)
float calculate_median(float *data, int n) {
    if (n == 0) return 0.0f;
    if (n == 1) return data[0];
    
    // Create a copy to avoid modifying original data
    float *work = malloc(n * sizeof(float));
    if (!work) return 0.0f;
    memcpy(work, data, n * sizeof(float));
    
    // Use quickselect for median (O(n) average case vs O(n log n) for sort)
    int k = n / 2;
    float median;
    
    if (n % 2 == 1) {
        median = quickselect(work, 0, n - 1, k);
    } else {
        float med1 = quickselect(work, 0, n - 1, k - 1);
        float med2 = quickselect(work, 0, n - 1, k);
        median = (med1 + med2) / 2.0f;
    }
    
    free(work);
    return median;
}

// Calculate Median Absolute Deviation (MAD)
float calculate_mad(float *data, int n, float median) {
    if (n == 0) return 0.0f;
    
    float *deviations = malloc(n * sizeof(float));
    if (!deviations) return 0.0f;
    
    // Calculate absolute deviations from median
    for (int i = 0; i < n; i++) {
        deviations[i] = fabsf(data[i] - median);
    }
    
    // Calculate median of absolute deviations
    float mad = calculate_median(deviations, n);
    free(deviations);
    
    // Scale MAD to approximate standard deviation (multiply by 1.4826)
    return mad * 1.4826f;
}

// Apply median subtraction to channel data
void apply_median_subtraction_to_channel(float *channel_data, int n_samples) {
    if (n_samples == 0) return;
    
    // Calculate median of this channel's time series
    float median = calculate_median(channel_data, n_samples);
    
    // Subtract median from all samples in this channel
    for (int t = 0; t < n_samples; t++) {
        channel_data[t] -= median;
    }
}

// ============================================================================
// FILE MANAGEMENT
// ============================================================================
// Compare two subband file entries by subband number for sorting
int compare_subband_files(const void *a, const void *b) {
    const SubbandFileEntry *file_a = (const SubbandFileEntry *)a;
    const SubbandFileEntry *file_b = (const SubbandFileEntry *)b;
    char *sb_a = strstr(file_a->name, "_sb");
    char *sb_b = strstr(file_b->name, "_sb");
    if (!sb_a || !sb_b) return 0;
    int num_a = atoi(sb_a + 3);
    int num_b = atoi(sb_b + 3);
    return num_a - num_b;
}

// List all subband files in a directory, sort by subband number
int list_subband_files(const char* directory_path, SubbandFileEntry *file_list, int max_files) {
    DIR* dir_handle = opendir(directory_path); 
    struct dirent* dir_entry; 
    int file_count = 0;
    if (!dir_handle) return 0;
    while ((dir_entry = readdir(dir_handle)) && file_count < max_files) {
        if (strstr(dir_entry->d_name, ".out")) {
            snprintf(file_list[file_count].name, 512, "%s/%s", directory_path, dir_entry->d_name); 
            file_count++;
        }
    }
    closedir(dir_handle); 
    qsort(file_list, file_count, sizeof(SubbandFileEntry), compare_subband_files);
    return file_count;
}

// Detect number of time samples in a voltage file by its size
int detect_time_samples_from_file(const char* filename) {
    FILE* file_handle = fopen(filename, "rb");
    if (!file_handle) {
        fprintf(stderr, "Cannot open %s to detect time samples\n", filename);
        return -1;
    }
    fseek(file_handle, 0, SEEK_END);
    long file_size_bytes = ftell(file_handle);
    fclose(file_handle);
    size_t block_size_bytes = (size_t)NUM_ANTENNAS * NUM_CHANNELS_PER_SUBBAND * SAMPLES_PER_BLOCK * NUM_POLARIZATIONS;
    int num_blocks = (int)(file_size_bytes / block_size_bytes);
    int num_time_samples = num_blocks * SAMPLES_PER_BLOCK;
    printf("Detected: file_size=%ld, block_size=%zu, blocks=%d, time_samples=%d\n", 
           file_size_bytes, block_size_bytes, num_blocks, num_time_samples);
    return num_time_samples;
}

// ============================================================================
// DISPERSION CALCULATION
// ============================================================================
DispersionParams* calculate_dispersion_params(double dm, int total_time_samples) {
    DispersionParams* params = malloc(sizeof(DispersionParams));
    if (!params) return NULL;
    
    // Allocate arrays
    params->channel_frequencies = malloc(NUM_CHANNELS * sizeof(double));
    params->dispersion_shifts = malloc(NUM_CHANNELS * sizeof(int));
    if (!params->channel_frequencies || !params->dispersion_shifts) {
        free(params->channel_frequencies);
        free(params->dispersion_shifts);
        free(params);
        return NULL;
    }
    
    // Calculate channel frequencies
    int idx = 0;
    for (int sb = 0; sb < NUM_SUBBANDS; ++sb) {
        for (int c = 0; c < NUM_CHANNELS_PER_SUBBAND; ++c) {
            params->channel_frequencies[idx++] = subband_start_freqs[sb] - c * fabs(CHANNEL_FREQ_STEP_MHZ);
        }
    }
    
    // Find reference frequency (highest)
    params->reference_frequency = params->channel_frequencies[0];
    for (int i = 1; i < NUM_CHANNELS; ++i) {
        if (params->channel_frequencies[i] > params->reference_frequency) {
            params->reference_frequency = params->channel_frequencies[i];
        }
    }
    
    // Calculate dispersion shifts
    params->max_shift = 0;
    for (int ch = 0; ch < NUM_CHANNELS; ++ch) {
        double f = params->channel_frequencies[ch];
        double f_ref = params->reference_frequency;
        double dt_sec = DISPERSION_CONST * dm * (1/(f*f) - 1/(f_ref*f_ref));
        int nsamp = (int)round(dt_sec / SAMPLE_TIME_SEC);
        
        if (nsamp > params->max_shift) params->max_shift = nsamp;
        params->dispersion_shifts[ch] = nsamp;
        
        // Debug output for first/last few channels
        if (ch < 3 || ch >= NUM_CHANNELS-3) {
            printf("ch%d: f=%.3f MHz, dt=%.3f ms, nsamp=%d\n", 
                   ch, f, dt_sec*1000.0, nsamp);
        }
    }
    
    params->total_time_samples = total_time_samples;
    
    printf("Frequency range: %.3f - %.3f MHz, max_shift=%d samples\n",
           params->channel_frequencies[NUM_CHANNELS-1], params->channel_frequencies[0], params->max_shift);
    
    return params;
}

void free_dispersion_params(DispersionParams* params) {
    if (params) {
        free(params->channel_frequencies);
        free(params->dispersion_shifts);
        free(params);
    }
}

// ============================================================================
// ANTENNA DATA MANAGEMENT
// ============================================================================
// Allocate antenna data using Structure of Arrays for better vectorization
AntennaData* allocate_antenna_data(int output_samples) {
    AntennaData* data = malloc(sizeof(AntennaData));
    if (!data) return NULL;
    
    data->num_output_samples = output_samples;
    
    // Allocate arrays for polarization 0
    data->antenna_power_pol0 = malloc(NUM_ANTENNAS * sizeof(float*));
    if (!data->antenna_power_pol0) {
        free(data);
        return NULL;
    }
    
    // Allocate arrays for polarization 1
    data->antenna_power_pol1 = malloc(NUM_ANTENNAS * sizeof(float*));
    if (!data->antenna_power_pol1) {
        free(data->antenna_power_pol0);
        free(data);
        return NULL;
    }
    
    // Allocate aligned memory for each antenna's time series
    for (int a = 0; a < NUM_ANTENNAS; ++a) {
        // Use aligned allocation for better SIMD performance (64-byte alignment for AVX-512)
        data->antenna_power_pol0[a] = aligned_alloc(64, output_samples * sizeof(float));
        data->antenna_power_pol1[a] = aligned_alloc(64, output_samples * sizeof(float));
        
        if (!data->antenna_power_pol0[a] || !data->antenna_power_pol1[a]) {
            // Cleanup on failure
            for (int i = 0; i <= a; ++i) {
                if (data->antenna_power_pol0[i]) free(data->antenna_power_pol0[i]);
                if (data->antenna_power_pol1[i]) free(data->antenna_power_pol1[i]);
            }
            free(data->antenna_power_pol0);
            free(data->antenna_power_pol1);
            free(data);
            return NULL;
        }
        
        // Initialize to zero
        memset(data->antenna_power_pol0[a], 0, output_samples * sizeof(float));
        memset(data->antenna_power_pol1[a], 0, output_samples * sizeof(float));
    }
    
    return data;
}

void free_antenna_data(AntennaData* data) {
    if (data) {
        if (data->antenna_power_pol0) {
            for (int a = 0; a < NUM_ANTENNAS; ++a) {
                if (data->antenna_power_pol0[a]) free(data->antenna_power_pol0[a]);
            }
            free(data->antenna_power_pol0);
        }
        if (data->antenna_power_pol1) {
            for (int a = 0; a < NUM_ANTENNAS; ++a) {
                if (data->antenna_power_pol1[a]) free(data->antenna_power_pol1[a]);
            }
            free(data->antenna_power_pol1);
        }
        free(data);
    }
}

// Allocate spectra data for frequency-resolved output (single subband)
SpectraData* allocate_spectra_data(int output_samples, int num_channels, int subband_index) {
    SpectraData* data = malloc(sizeof(SpectraData));
    if (!data) return NULL;
    
    data->num_output_samples = output_samples;
    data->num_channels = num_channels;
    data->subband_index = subband_index;
    
    // Allocate arrays for Stokes I
    data->spectra_stokes_i = malloc(NUM_ANTENNAS * sizeof(float**));
    if (!data->spectra_stokes_i) {
        free(data);
        return NULL;
    }
    
    // Allocate memory for each antenna's frequency-time matrix
    for (int a = 0; a < NUM_ANTENNAS; ++a) {
        // Allocate channel arrays for each antenna
        data->spectra_stokes_i[a] = malloc(num_channels * sizeof(float*));
        
        if (!data->spectra_stokes_i[a]) {
            // Cleanup on failure
            for (int i = 0; i < a; ++i) {
                if (data->spectra_stokes_i[i]) {
                    for (int c = 0; c < num_channels; ++c) {
                        if (data->spectra_stokes_i[i][c]) free(data->spectra_stokes_i[i][c]);
                    }
                    free(data->spectra_stokes_i[i]);
                }
            }
            free(data->spectra_stokes_i);
            free(data);
            return NULL;
        }
        
        // Allocate time series for each channel
        for (int c = 0; c < num_channels; ++c) {
            data->spectra_stokes_i[a][c] = aligned_alloc(64, output_samples * sizeof(float));
            
            if (!data->spectra_stokes_i[a][c]) {
                // Cleanup on failure
                for (int cc = 0; cc < c; ++cc) {
                    free(data->spectra_stokes_i[a][cc]);
                }
                for (int i = 0; i < a; ++i) {
                    for (int cc = 0; cc < num_channels; ++cc) {
                        free(data->spectra_stokes_i[i][cc]);
                    }
                    free(data->spectra_stokes_i[i]);
                }
                free(data->spectra_stokes_i[a]);
                free(data->spectra_stokes_i);
                free(data);
                return NULL;
            }
            
            // Initialize to zero
            memset(data->spectra_stokes_i[a][c], 0, output_samples * sizeof(float));
        }
    }
    
    return data;
}

void free_spectra_data(SpectraData* data) {
    if (data) {
        if (data->spectra_stokes_i) {
            for (int a = 0; a < NUM_ANTENNAS; ++a) {
                if (data->spectra_stokes_i[a]) {
                    for (int c = 0; c < data->num_channels; ++c) {
                        if (data->spectra_stokes_i[a][c]) free(data->spectra_stokes_i[a][c]);
                    }
                    free(data->spectra_stokes_i[a]);
                }
            }
            free(data->spectra_stokes_i);
        }
        free(data);
    }
}

/**
 * Allocate memory for raw voltage data structure
 * Storage: [antenna][channel][pol][time] with complex data (real/imag pairs)
 */
RawVoltageData* allocate_raw_voltage_data(int output_samples, int num_channels, int subband_index) {
    RawVoltageData* data = malloc(sizeof(RawVoltageData));
    if (!data) return NULL;
    
    data->num_antennas = NUM_ANTENNAS;
    data->num_channels = num_channels;
    data->num_output_samples = output_samples;
    data->subband_index = subband_index;
    
    // Allocate antenna array
    data->raw_voltages = malloc(NUM_ANTENNAS * sizeof(float***));
    if (!data->raw_voltages) {
        free(data);
        return NULL;
    }
    
    // Allocate for each antenna
    for (int a = 0; a < NUM_ANTENNAS; ++a) {
        data->raw_voltages[a] = malloc(num_channels * sizeof(float**));
        if (!data->raw_voltages[a]) {
            // Cleanup on failure
            for (int i = 0; i < a; ++i) {
                for (int c = 0; c < num_channels; ++c) {
                    for (int p = 0; p < NUM_POLARIZATIONS; ++p) {
                        if (data->raw_voltages[i][c][p]) free(data->raw_voltages[i][c][p]);
                    }
                    free(data->raw_voltages[i][c]);
                }
                free(data->raw_voltages[i]);
            }
            free(data->raw_voltages);
            free(data);
            return NULL;
        }
        
        // Allocate for each channel
        for (int c = 0; c < num_channels; ++c) {
            data->raw_voltages[a][c] = malloc(NUM_POLARIZATIONS * sizeof(float*));
            if (!data->raw_voltages[a][c]) {
                // Cleanup on failure
                for (int cc = 0; cc < c; ++cc) {
                    for (int p = 0; p < NUM_POLARIZATIONS; ++p) {
                        if (data->raw_voltages[a][cc][p]) free(data->raw_voltages[a][cc][p]);
                    }
                    free(data->raw_voltages[a][cc]);
                }
                for (int i = 0; i < a; ++i) {
                    for (int cc = 0; cc < num_channels; ++cc) {
                        for (int p = 0; p < NUM_POLARIZATIONS; ++p) {
                            if (data->raw_voltages[i][cc][p]) free(data->raw_voltages[i][cc][p]);
                        }
                        free(data->raw_voltages[i][cc]);
                    }
                    free(data->raw_voltages[i]);
                }
                free(data->raw_voltages[a]);
                free(data->raw_voltages);
                free(data);
                return NULL;
            }
            
            // Allocate for each polarization (complex data: 2 floats per sample)
            for (int p = 0; p < NUM_POLARIZATIONS; ++p) {
                data->raw_voltages[a][c][p] = aligned_alloc(64, output_samples * 2 * sizeof(float));
                if (!data->raw_voltages[a][c][p]) {
                    // Cleanup on failure
                    for (int pp = 0; pp < p; ++pp) {
                        free(data->raw_voltages[a][c][pp]);
                    }
                    for (int cc = 0; cc < c; ++cc) {
                        for (int pp = 0; pp < NUM_POLARIZATIONS; ++pp) {
                            free(data->raw_voltages[a][cc][pp]);
                        }
                        free(data->raw_voltages[a][cc]);
                    }
                    for (int i = 0; i < a; ++i) {
                        for (int cc = 0; cc < num_channels; ++cc) {
                            for (int pp = 0; pp < NUM_POLARIZATIONS; ++pp) {
                                free(data->raw_voltages[i][cc][pp]);
                            }
                            free(data->raw_voltages[i][cc]);
                        }
                        free(data->raw_voltages[i]);
                    }
                    free(data->raw_voltages[a][c]);
                    free(data->raw_voltages[a]);
                    free(data->raw_voltages);
                    free(data);
                    return NULL;
                }
                
                // Initialize to zero
                memset(data->raw_voltages[a][c][p], 0, output_samples * 2 * sizeof(float));
            }
        }
    }
    
    return data;
}

/**
 * Free raw voltage data structure
 */
void free_raw_voltage_data(RawVoltageData* data) {
    if (data) {
        if (data->raw_voltages) {
            for (int a = 0; a < data->num_antennas; ++a) {
                if (data->raw_voltages[a]) {
                    for (int c = 0; c < data->num_channels; ++c) {
                        if (data->raw_voltages[a][c]) {
                            for (int p = 0; p < NUM_POLARIZATIONS; ++p) {
                                if (data->raw_voltages[a][c][p]) free(data->raw_voltages[a][c][p]);
                            }
                            free(data->raw_voltages[a][c]);
                        }
                    }
                    free(data->raw_voltages[a]);
                }
            }
            free(data->raw_voltages);
        }
        free(data);
    }
}





// Allocate channel data using Structure of Arrays
ChannelData* allocate_channel_data(int num_channels, int num_time_samples) {
    ChannelData* data = malloc(sizeof(ChannelData));
    if (!data) return NULL;
    
    data->num_channels = num_channels;
    data->num_time_samples = num_time_samples;
    
    // Allocate flat array with alignment for vectorization
    size_t total_size = (size_t)num_channels * num_time_samples * sizeof(float);
    data->channel_powers = aligned_alloc(64, total_size);
    
    if (!data->channel_powers) {
        free(data);
        return NULL;
    }
    
    memset(data->channel_powers, 0, total_size);
    return data;
}

void free_channel_data(ChannelData* data) {
    if (data) {
        if (data->channel_powers) free(data->channel_powers);
        free(data);
    }
}

// ============================================================================
// HEADER WRITING
// ============================================================================
void write_dedisp_header(FILE *fout, int nblocks, int ntime_per_block, int npol, 
                        int ant_idx, double dm, double tsamp, int nout, int total_samples, 
                        int start_sample_index) {
    char header[4096];
    int idx = 0;
    idx += snprintf(header+idx, sizeof(header)-idx, "%s\n", FILE_MAGIC);
    idx += snprintf(header+idx, sizeof(header)-idx, "VERSION %d\n", FILE_VERSION);
    idx += snprintf(header+idx, sizeof(header)-idx, "BANDWIDTH_MHZ %.8f\n", TOTAL_BANDWIDTH_MHZ);
    idx += snprintf(header+idx, sizeof(header)-idx, "NCHAN %d\n", NUM_CHANNELS);
    idx += snprintf(header+idx, sizeof(header)-idx, "N_SAMPLES %d\n", total_samples);
    idx += snprintf(header+idx, sizeof(header)-idx, "TSAMP %.9f\n", tsamp);
    idx += snprintf(header+idx, sizeof(header)-idx, "FCH1_MHZ %.8f\n", HIGHEST_FREQ_MHZ);
    idx += snprintf(header+idx, sizeof(header)-idx, "FSTEP_MHZ %.8f\n", CHANNEL_FREQ_STEP_MHZ);
    idx += snprintf(header+idx, sizeof(header)-idx, "NBLOCKS %d\n", nblocks);
    idx += snprintf(header+idx, sizeof(header)-idx, "NTIME_PER_BLOCK %d\n", ntime_per_block);
    idx += snprintf(header+idx, sizeof(header)-idx, "NPOL %d\n", npol);
    idx += snprintf(header+idx, sizeof(header)-idx, "ANT %d\n", ant_idx);
    idx += snprintf(header+idx, sizeof(header)-idx, "NSB %d\n", NUM_SUBBANDS);
    idx += snprintf(header+idx, sizeof(header)-idx, "DM %f\n", dm);
    idx += snprintf(header+idx, sizeof(header)-idx, "NOUT %d\n", nout);
    idx += snprintf(header+idx, sizeof(header)-idx, "START_SAMPLE %d\n", start_sample_index);
    // FIXED: START_TIME should be the actual time in the original observation corresponding to start_sample_index
    // This accurately represents which slice of the original data this file contains
    idx += snprintf(header+idx, sizeof(header)-idx, "START_TIME %.8f\n", start_sample_index * tsamp);
    idx += snprintf(header+idx, sizeof(header)-idx, "POL_NAMES %s %s\n", polarization_names[0], polarization_names[1]);
    idx += snprintf(header+idx, sizeof(header)-idx, "DATA_FORMAT STOKES_I\n");
    idx += snprintf(header+idx, sizeof(header)-idx, "ENDH\n");
    int pad = HEADER_SIZE_BYTES - idx; if(pad<0) pad=0;
    memset(header+idx, ' ', pad);
    fwrite(header, 1, HEADER_SIZE_BYTES, fout);
}

// ============================================================================
// DATA PROCESSING
// ============================================================================
/**
 * Optimized processing of a single subband voltage file with global median subtraction.
 * This function implements a two-pass algorithm:
 * Pass 1: Read all data and calculate global median per frequency channel
 * Pass 2: Apply median subtraction and accumulate final antenna power
 *
 * @param filename         Path to the subband voltage file
 * @param subband_index    Index of the subband (0 to NSB-1)
 * @param dispersion       Dispersion parameters (shifts, frequencies, etc.)
 * @param antenna_data     Output buffer for accumulated antenna power
 * @param spectra_data     Output buffer for spectra data (optional, for spectra mode)
 * @param config           Configuration including median subtraction settings
 * @param start_sample     Starting sample index for time slicing
 * @param stop_sample      Stopping sample index for time slicing
 * @return                 Number of blocks processed, or -1 on error
 */
int process_subband_file(const char* filename, int subband_index, 
                        const DispersionParams* dispersion,
                        AntennaData* antenna_data, SpectraData* spectra_data, 
                        RawVoltageData* raw_voltage_data,
                        const DedispersionConfig* config, int start_sample, int stop_sample) {
    // Open the subband voltage file for reading
    FILE* file_handle = fopen(filename, "rb");
    if (!file_handle) {
        fprintf(stderr, "Cannot open %s\n", filename);
        return -1;
    }

    // Determine file size and number of blocks
    fseek(file_handle, 0, SEEK_END);
    long file_size_bytes = ftell(file_handle);
    fseek(file_handle, 0, SEEK_SET);

    size_t bytes_per_block = (size_t)NUM_ANTENNAS * NUM_CHANNELS_PER_SUBBAND * SAMPLES_PER_BLOCK * NUM_POLARIZATIONS;
    int num_blocks = (int)(file_size_bytes / bytes_per_block);
    int total_time_samples = num_blocks * SAMPLES_PER_BLOCK;

    printf("\n ... Processing subband %d/%d: %ld bytes, %d blocks", 
           subband_index + 1, NUM_SUBBANDS, file_size_bytes, num_blocks);
    
    if (config->enable_median_subtraction) {
        printf("\n");
    } else {
        printf("\n");
    }

    if (file_size_bytes % bytes_per_block != 0) {
        printf("\n    Warning: Incomplete last block\n");
    }

    // Optimize block processing size based on available memory
    int block_chunk_size = config->block_processing_size > 0 ? config->block_processing_size : 1024;
    if (block_chunk_size > num_blocks) block_chunk_size = num_blocks;

    // Allocate buffer for multiple blocks to reduce I/O overhead
    size_t chunk_buffer_size = bytes_per_block * block_chunk_size;
    uint8_t *chunk_buffer = malloc(chunk_buffer_size);
    if (!chunk_buffer) {
        fprintf(stderr, "Memory allocation failed for chunk buffer\n");
        fclose(file_handle);
        return -1;
    }

    // For global median subtraction, collect all samples per channel first
    float **global_channel_data = NULL; // [channel][samples_for_median]
    int *channel_sample_counts = NULL;   // [channel] - count of samples collected per channel
    float *channel_medians = NULL;       // [channel] - global medians
    
    if (config->enable_median_subtraction) {
        printf("\n");
        
        // Allocate storage for channel data - we'll collect densely packed samples
        global_channel_data = malloc(NUM_CHANNELS_PER_SUBBAND * sizeof(float*));
        channel_sample_counts = calloc(NUM_CHANNELS_PER_SUBBAND, sizeof(int));
        channel_medians = malloc(NUM_CHANNELS_PER_SUBBAND * sizeof(float));
        
        if (!global_channel_data || !channel_sample_counts || !channel_medians) {
            fprintf(stderr, "Memory allocation failed for median calculation\n");
            free(chunk_buffer);
            free(global_channel_data);
            free(channel_sample_counts);
            free(channel_medians);
            fclose(file_handle);
            return -1;
        }
        
        // Allocate storage for each channel's time series (sum over antennas & polarizations)
        // Allocate for worst case: all input times might be used
        for (int c = 0; c < NUM_CHANNELS_PER_SUBBAND; c++) {
            global_channel_data[c] = malloc(total_time_samples * sizeof(float));
            if (!global_channel_data[c]) {
                fprintf(stderr, "Memory allocation failed for channel %d data\n", c);
                // Cleanup on failure
                for (int i = 0; i < c; i++) {
                    free(global_channel_data[i]);
                }
                free(global_channel_data);
                free(channel_sample_counts);
                free(channel_medians);
                free(chunk_buffer);
                fclose(file_handle);
                return -1;
            }
        }
    } else {
        printf("\n");
    }

    int blocks_processed = 0;
    
    if (config->enable_median_subtraction) {
        // PASS 1: Read all data and accumulate channel power for median calculation
        printf("    Collecting channel data\n");
        
        for (int chunk_start = 0; chunk_start < num_blocks; chunk_start += block_chunk_size) {
            int blocks_in_chunk = (chunk_start + block_chunk_size > num_blocks) ? 
                                 (num_blocks - chunk_start) : block_chunk_size;
            
            size_t bytes_to_read = bytes_per_block * blocks_in_chunk;
            size_t bytes_read = fread(chunk_buffer, 1, bytes_to_read, file_handle);
            
            if (bytes_read != bytes_to_read) {
                if (feof(file_handle)) break;
                fprintf(stderr, "Read error at chunk starting block %d\n", chunk_start);
                break;
            }

            // Accumulate power per channel per time sample (sum over antennas & polarizations)
            for (int block_idx = 0; block_idx < blocks_in_chunk; block_idx++) {
                uint8_t *block_data = chunk_buffer + block_idx * bytes_per_block;
                int global_block_index = chunk_start + block_idx;
                
                for (int time_in_block = 0; time_in_block < SAMPLES_PER_BLOCK; time_in_block++) {
                    int global_time_index = global_block_index * SAMPLES_PER_BLOCK + time_in_block;
                    
                    // Parallelize channel processing within each time sample
                    #ifdef _OPENMP
                    #pragma omp parallel for schedule(static, 16)
                    #endif
                    for (int channel_in_subband = 0; channel_in_subband < NUM_CHANNELS_PER_SUBBAND; channel_in_subband++) {
                        // For median calculation, we need to determine if this input time
                        // will contribute to the output range after dedispersion
                        int global_channel_index = subband_index * NUM_CHANNELS_PER_SUBBAND + channel_in_subband;
                        int dispersion_shift = config->enable_dedispersion ? dispersion->dispersion_shifts[global_channel_index] : 0;
                        int output_time_index = global_time_index - dispersion_shift;
                        
                        // Only include samples that will appear in the output time range
                        if (output_time_index < start_sample || output_time_index > stop_sample) {
                            continue; // Skip samples that won't be in the final output
                        }
                        
                        float channel_power_sum = 0.0f;
                        
                        // Sum over all antennas and polarizations for this channel/time
                        for (int antenna_index = 0; antenna_index < NUM_ANTENNAS; antenna_index++) {
                            for (int pol = 0; pol < NUM_POLARIZATIONS; pol++) {
                                size_t offset = (size_t)antenna_index * NUM_CHANNELS_PER_SUBBAND * SAMPLES_PER_BLOCK * NUM_POLARIZATIONS + 
                                               (size_t)channel_in_subband * SAMPLES_PER_BLOCK * NUM_POLARIZATIONS + 
                                               (size_t)time_in_block * NUM_POLARIZATIONS + pol;

                                int8_t real_part, imag_part;
                                unpack_4bit_complex(block_data[offset], &real_part, &imag_part);
                                float power = (float)real_part * (float)real_part + (float)imag_part * (float)imag_part;
                                channel_power_sum += power;
                            }
                        }
                        
                        // Store summed power densely for median calculation
                        // Use atomic increment to get next index (thread-safe)
                        int idx;
                        #ifdef _OPENMP
                        #pragma omp atomic capture
                        #endif
                        idx = channel_sample_counts[channel_in_subband]++;
                        
                        global_channel_data[channel_in_subband][idx] = channel_power_sum;
                    }
                }
            }
            
            blocks_processed += blocks_in_chunk;
            
            // Update progress bar
            double progress = (double)(chunk_start + blocks_in_chunk) / num_blocks;
            print_progress_bar("    Data collection", progress, 40);
        }
        printf("\n");
        
        // Calculate median for each channel using only the samples actually collected
        printf("    Computing channel medians\n");
        
        #ifdef _OPENMP
        int progress_counter = 0;
        #pragma omp parallel for schedule(dynamic, 8)
        #endif
        for (int c = 0; c < NUM_CHANNELS_PER_SUBBAND; c++) {
            int n_samples = channel_sample_counts[c];
            if (n_samples > 0) {
                channel_medians[c] = calculate_median(global_channel_data[c], n_samples);
            } else {
                // No samples for this channel (shouldn't happen, but handle gracefully)
                channel_medians[c] = 0.0f;
            }
            
            // Update progress with thread-safe counter
            #ifdef _OPENMP
            #pragma omp atomic
            progress_counter++;
            
            // Only update display occasionally to avoid too many updates
            if (progress_counter % 8 == 0 || progress_counter == NUM_CHANNELS_PER_SUBBAND) {
                #pragma omp critical
                {
                    double progress = (double)progress_counter / NUM_CHANNELS_PER_SUBBAND;
                    print_progress_bar("    Median calculation", progress, 40);
                }
            }
            #else
            double progress = (double)(c + 1) / NUM_CHANNELS_PER_SUBBAND;
            print_progress_bar("    Median calculation", progress, 40);
            #endif
        }
        printf("\n");
        
        // PASS 2: Re-read file, apply median subtraction, and accumulate final results
        printf("    Applying median subtraction\n");
        
        // Reset file position for second pass
        fseek(file_handle, 0, SEEK_SET);
        blocks_processed = 0;
        
        for (int chunk_start = 0; chunk_start < num_blocks; chunk_start += block_chunk_size) {
            int blocks_in_chunk = (chunk_start + block_chunk_size > num_blocks) ? 
                                 (num_blocks - chunk_start) : block_chunk_size;
            
            size_t bytes_to_read = bytes_per_block * blocks_in_chunk;
            size_t bytes_read = fread(chunk_buffer, 1, bytes_to_read, file_handle);
            
            if (bytes_read != bytes_to_read) {
                if (feof(file_handle)) break;
                fprintf(stderr, "Read error at chunk starting block %d\n", chunk_start);
                break;
            }

            // Process with median subtraction and accumulate final results
            for (int block_idx = 0; block_idx < blocks_in_chunk; block_idx++) {
                uint8_t *block_data = chunk_buffer + block_idx * bytes_per_block;
                int global_block_index = chunk_start + block_idx;
                
                // Parallelize antenna processing within each block
                #ifdef _OPENMP
                #pragma omp parallel for schedule(static, 4) collapse(2)
                #endif
                for (int antenna_index = 0; antenna_index < NUM_ANTENNAS; antenna_index++) {
                    for (int time_in_block = 0; time_in_block < SAMPLES_PER_BLOCK; time_in_block++) {
                        int global_time_index = global_block_index * SAMPLES_PER_BLOCK + time_in_block;
                        
                        for (int channel_in_subband = 0; channel_in_subband < NUM_CHANNELS_PER_SUBBAND; channel_in_subband++) {
                            int global_channel_index = subband_index * NUM_CHANNELS_PER_SUBBAND + channel_in_subband;
                            int dispersion_shift = config->enable_dedispersion ? dispersion->dispersion_shifts[global_channel_index] : 0;
                            int output_time_index = global_time_index - dispersion_shift;
                            
                            // Check if this time sample falls within our slice range
                            if (output_time_index < start_sample || output_time_index > stop_sample) {
                                continue; // Skip samples outside the requested time range
                            }
                            
                            // Map global output time index to local array index
                            int local_time_index = output_time_index - start_sample;
                            
                            if (local_time_index >= 0 && local_time_index < antenna_data->num_output_samples) {
                                for (int pol = 0; pol < NUM_POLARIZATIONS; pol++) {
                                    size_t offset = (size_t)antenna_index * NUM_CHANNELS_PER_SUBBAND * SAMPLES_PER_BLOCK * NUM_POLARIZATIONS + 
                                                   (size_t)channel_in_subband * SAMPLES_PER_BLOCK * NUM_POLARIZATIONS + 
                                                   (size_t)time_in_block * NUM_POLARIZATIONS + pol;

                                    int8_t real_part, imag_part;
                                    unpack_4bit_complex(block_data[offset], &real_part, &imag_part);
                                    float power = (float)real_part * (float)real_part + (float)imag_part * (float)imag_part;
                                    
                                    // Apply median subtraction (subtract proportional to antenna/pol contribution)
                                    float median_contribution = channel_medians[channel_in_subband] / (NUM_ANTENNAS * NUM_POLARIZATIONS);
                                    power -= median_contribution;
                                    
                                    if (pol == 0) {
                                        antenna_data->antenna_power_pol0[antenna_index][local_time_index] += power;
                                    } else {
                                        antenna_data->antenna_power_pol1[antenna_index][local_time_index] += power;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            
            blocks_processed += blocks_in_chunk;
            
            // Update progress bar
            double progress = (double)(chunk_start + blocks_in_chunk) / num_blocks;
            print_progress_bar("    Median subtraction", progress, 40);
        }
        printf("\n");
        
        // Free global channel data
        for (int c = 0; c < NUM_CHANNELS_PER_SUBBAND; c++) {
            free(global_channel_data[c]);
        }
        free(global_channel_data);
        free(channel_sample_counts);
        free(channel_medians);
        
    } else {
        // Single pass: direct processing without median subtraction
        
        for (int chunk_start = 0; chunk_start < num_blocks; chunk_start += block_chunk_size) {
            int blocks_in_chunk = (chunk_start + block_chunk_size > num_blocks) ? 
                                 (num_blocks - chunk_start) : block_chunk_size;
            
            size_t bytes_to_read = bytes_per_block * blocks_in_chunk;
            size_t bytes_read = fread(chunk_buffer, 1, bytes_to_read, file_handle);
            
            if (bytes_read != bytes_to_read) {
                if (feof(file_handle)) break;
                fprintf(stderr, "Read error at chunk starting block %d\n", chunk_start);
                break;
            }

            // Direct processing without median subtraction using vectorized processing
            for (int block_idx = 0; block_idx < blocks_in_chunk; block_idx++) {
                uint8_t *block_data = chunk_buffer + block_idx * bytes_per_block;
                int global_block_index = chunk_start + block_idx;
                
                // Parallelize antenna processing for better multi-core utilization
                #ifdef _OPENMP
                #pragma omp parallel for schedule(static, 4) collapse(2)
                #endif
                for (int antenna_index = 0; antenna_index < NUM_ANTENNAS; antenna_index++) {
                    for (int time_in_block = 0; time_in_block < SAMPLES_PER_BLOCK; time_in_block++) {
                        int global_time_index = global_block_index * SAMPLES_PER_BLOCK + time_in_block;
                        
                        // Use the new vectorized processing function
                        process_block_vectorized(block_data, antenna_index, time_in_block,
                                               NULL, 0, subband_index, dispersion,
                                               antenna_data, spectra_data, raw_voltage_data, config, global_time_index,
                                               start_sample, stop_sample);
                    }
                }
            }
            
            blocks_processed += blocks_in_chunk;
            
            // Update progress bar
            double progress = (double)(chunk_start + blocks_in_chunk) / num_blocks;
            print_progress_bar("    Data processing", progress, 40);
        }
        printf("\n");
    }

    // Cleanup
    free(chunk_buffer);
    fclose(file_handle);
    
    printf("    Completed: %d blocks processed\n", blocks_processed);

    return blocks_processed;
}


// ============================================================================
// OUTPUT WRITING
// ============================================================================
/**
 * Write output files for each antenna after dedispersion and accumulation.
 * Each file contains a header and Stokes I data (sum of both polarizations).
 *
 * @param output_dir      Directory to write output files
 * @param antenna_data    Accumulated antenna power data
 * @param disp_params     Dispersion parameters
 * @param config          Program configuration
 * @return                0 on success, -1 on error
 */
int write_output_files(const char* output_dir, const AntennaData* antenna_data,
                      const DispersionParams* disp_params, const DedispersionConfig* config, 
                      int start_sample) {
    // Create time_series subdirectory
    char time_series_dir[512];
    snprintf(time_series_dir, sizeof(time_series_dir), "%s/time_series", output_dir);
    
    // Create directory if it doesn't exist
    if (mkdir(time_series_dir, 0755) != 0 && errno != EEXIST) {
        fprintf(stderr, "Failed to create time_series directory: %s\n", time_series_dir);
        perror("Error");
        return -1;
    }
    
    printf("Writing output files to directory: %s\n", time_series_dir);
    printf("Number of output samples: %d\n", antenna_data->num_output_samples);
    
    for (int antenna_index = 0; antenna_index < NUM_ANTENNAS; ++antenna_index) {
        // Build output filename for this antenna
        char filename[512];
        snprintf(filename, sizeof(filename), "%s/ant_%02d.dedisp", time_series_dir, antenna_index);
        
        FILE* fout = fopen(filename, "wb");
        if (!fout) {
            fprintf(stderr, "Cannot open %s for writing\n", filename);
            perror("Error");
            return -1;
        }
        
        // Write header with all metadata including start sample for time slicing
        write_dedisp_header(fout, 1, disp_params->total_time_samples, NUM_POLARIZATIONS, antenna_index, 
                           config->dispersion_measure, SAMPLE_TIME_SEC, antenna_data->num_output_samples, 
                           disp_params->total_time_samples, start_sample);
        
        // Allocate buffer for Stokes I (sum of both polarizations)
        float *stokes_i = malloc(antenna_data->num_output_samples * sizeof(float));
        if (!stokes_i) {
            fprintf(stderr, "Memory allocation failed for antenna %d\n", antenna_index);
            fclose(fout);
            return -1;
        }
        
        // Sum both polarizations for each time sample and check for non-zero data
        float total_power = 0.0f;
        for (int t = 0; t < antenna_data->num_output_samples; ++t) {
            stokes_i[t] = antenna_data->antenna_power_pol0[antenna_index][t] + antenna_data->antenna_power_pol1[antenna_index][t];
            total_power += stokes_i[t];
        }
        
        // Write Stokes I data to file
        size_t written = fwrite(stokes_i, sizeof(float), antenna_data->num_output_samples, fout);
        if (written != (size_t)antenna_data->num_output_samples) {
            fprintf(stderr, "Write error for antenna %d: wrote %zu of %d samples\n", 
                    antenna_index, written, antenna_data->num_output_samples);
            free(stokes_i);
            fclose(fout);
            return -1;
        }
        
        free(stokes_i);
        fclose(fout);
        
        // Update progress bar
        double progress = (double)(antenna_index + 1) / NUM_ANTENNAS;
        char progress_msg[64];
        snprintf(progress_msg, sizeof(progress_msg), "Writing files (%.1f MB/ant)", 
                (antenna_data->num_output_samples * sizeof(float)) / (1024.0 * 1024.0));
        print_progress_bar(progress_msg, progress, 40);
    }
    printf("\n");
    return 0;
}

/**
 * Write spectra data files to the spectra/ subdirectory
 * Creates binary files with [time][channel] layout for each antenna and polarization
 * 
 * @param output_dir      Base output directory
 * @param spectra_data    Spectra data structure  
 * @param disp_params     Dispersion parameters
 * @param config          Program configuration
 * @return                0 on success, -1 on error
 */
int write_spectra_files(const char* output_dir, const SpectraData* spectra_data,
                        const DispersionParams* disp_params, const DedispersionConfig* config) {
    // Suppress unused parameter warnings
    (void)disp_params;
    (void)config;
    
    // Create spectra subdirectory
    char spectra_dir[512];
    snprintf(spectra_dir, sizeof(spectra_dir), "%s/spectra", output_dir);
    
    // Create directory if it doesn't exist
    if (mkdir(spectra_dir, 0755) != 0 && errno != EEXIST) {
        fprintf(stderr, "Failed to create spectra directory: %s\n", spectra_dir);
        perror("Error");
        return -1;
    }
    
    printf("Writing spectra files to directory: %s\n", spectra_dir);
    printf("Number of channels: %d, time samples: %d\n", spectra_data->num_channels, spectra_data->num_output_samples);
    
    for (int antenna_index = 0; antenna_index < NUM_ANTENNAS; ++antenna_index) {
        // Write Stokes I data with Option B naming: ant_YY_subband_XX_stokes_i.spec
        char filename[512];
        snprintf(filename, sizeof(filename), "%s/ant_%02d_subband_%02d_stokes_i.spec", 
                spectra_dir, antenna_index, spectra_data->subband_index);
        
        FILE* fout = fopen(filename, "wb");
        if (!fout) {
            fprintf(stderr, "Cannot open %s for writing\n", filename);
            perror("Error");
            return -1;
        }
        
        // Write header information with timing metadata
        // Convert start time to microseconds for integer storage
        uint32_t start_time_us = (uint32_t)(config->start_time * 1000000.0);
        // Store sample time in nanoseconds for precision
        uint32_t sample_time_ns = (uint32_t)(SAMPLE_TIME_SEC * 1000000000.0);
        
        uint32_t header[6] = {
            (uint32_t)spectra_data->num_output_samples,  // [0] Number of time samples
            (uint32_t)spectra_data->num_channels,        // [1] Number of frequency channels
            (uint32_t)antenna_index,                     // [2] Antenna index
            (uint32_t)spectra_data->subband_index,       // [3] Subband index (Option B)
            start_time_us,                               // [4] Start time in microseconds
            sample_time_ns                               // [5] Sample time in nanoseconds
        };
        
        fwrite(header, sizeof(uint32_t), 6, fout);
        
        // Write data in [time][channel] order for efficient time-domain access
        for (int t = 0; t < spectra_data->num_output_samples; t++) {
            // Allocate temporary buffer for this time sample
            float *stokes_i_buffer = malloc(spectra_data->num_channels * sizeof(float));
            
            if (!stokes_i_buffer) {
                fprintf(stderr, "Memory allocation failed for antenna %d\n", antenna_index);
                fclose(fout);
                return -1;
            }
            
            // Copy data for this time sample across all channels
            for (int ch = 0; ch < spectra_data->num_channels; ch++) {
                stokes_i_buffer[ch] = spectra_data->spectra_stokes_i[antenna_index][ch][t];
            }
            
            // Write this time sample's data
            size_t written = fwrite(stokes_i_buffer, sizeof(float), spectra_data->num_channels, fout);
            
            if (written != (size_t)spectra_data->num_channels) {
                fprintf(stderr, "Write error for antenna %d at time %d\n", antenna_index, t);
                free(stokes_i_buffer);
                fclose(fout);
                return -1;
            }
            
            free(stokes_i_buffer);
        }
        
        fclose(fout);
        
        // Update progress bar
        double progress = (double)(antenna_index + 1) / NUM_ANTENNAS;
        char progress_msg[64];
        snprintf(progress_msg, sizeof(progress_msg), "Writing spectra (%.1f MB/ant)", 
                (spectra_data->num_output_samples * spectra_data->num_channels * sizeof(float)) / (1024.0 * 1024.0));
        print_progress_bar(progress_msg, progress, 40);
    }
    printf("\n");
    return 0;
}

int write_raw_voltage_files(const char* output_dir, const RawVoltageData* raw_voltage_data,
                           const DispersionParams* disp_params, const DedispersionConfig* config, int subband_index) {
    // Suppress unused parameter warnings
    (void)disp_params;
    
    // Create raw_voltages subdirectory
    char raw_dir[512];
    snprintf(raw_dir, sizeof(raw_dir), "%s/raw_voltages", output_dir);
    
    // Create directory if it doesn't exist
    if (mkdir(raw_dir, 0755) != 0 && errno != EEXIST) {
        fprintf(stderr, "Failed to create raw voltages directory: %s\n", raw_dir);
        perror("Error");
        return -1;
    }
    
    printf("Writing raw voltage files to directory: %s\n", raw_dir);
    printf("Number of channels: %d, time samples: %d\n", raw_voltage_data->num_channels, raw_voltage_data->num_output_samples);
    
    for (int antenna_index = 0; antenna_index < NUM_ANTENNAS; ++antenna_index) {
        for (int pol = 0; pol < 2; pol++) {
            // Write raw voltage data: matching spectra naming but with .raw extension
            char filename[512];
            const char* pol_name = (pol == 0) ? "pol_x" : "pol_y";
            snprintf(filename, sizeof(filename), "%s/ant_%02d_subband_%02d_%s.raw", 
                    raw_dir, antenna_index, subband_index, pol_name);
            
            FILE* fout = fopen(filename, "wb");
            if (!fout) {
                fprintf(stderr, "Cannot open %s for writing\n", filename);
                perror("Error");
                return -1;
            }
            
            // Write header information with timing metadata
            // Convert start time to microseconds for integer storage
            uint32_t start_time_us = (uint32_t)(config->start_time * 1000000.0);
            // Store sample time in nanoseconds for precision
            uint32_t sample_time_ns = (uint32_t)(SAMPLE_TIME_SEC * 1000000000.0);
            
            uint32_t header[6] = {
                (uint32_t)raw_voltage_data->num_output_samples,  // [0] Number of time samples
                (uint32_t)raw_voltage_data->num_channels,        // [1] Number of frequency channels
                (uint32_t)antenna_index,                         // [2] Antenna index
                (uint32_t)subband_index,                         // [3] Subband index
                start_time_us,                                   // [4] Start time in microseconds
                sample_time_ns                                   // [5] Sample time in nanoseconds
            };
            
            fwrite(header, sizeof(uint32_t), 6, fout);
            
            // Write data in [time][channel] order for efficient time-domain access
            for (int t = 0; t < raw_voltage_data->num_output_samples; t++) {
                // Allocate temporary buffer for this time sample (complex values)
                float *voltage_buffer = malloc(raw_voltage_data->num_channels * 2 * sizeof(float));
                
                if (!voltage_buffer) {
                    fprintf(stderr, "Memory allocation failed for antenna %d pol %d\n", antenna_index, pol);
                    fclose(fout);
                    return -1;
                }
                
                // Copy data for this time sample across all channels
                for (int ch = 0; ch < raw_voltage_data->num_channels; ch++) {
                    // Raw voltages are stored as real/imaginary interleaved
                    voltage_buffer[ch * 2] = raw_voltage_data->raw_voltages[antenna_index][ch][pol][t * 2];     // Real part
                    voltage_buffer[ch * 2 + 1] = raw_voltage_data->raw_voltages[antenna_index][ch][pol][t * 2 + 1]; // Imaginary part
                }
                
                // Write this time sample's data (complex values as real/imag pairs)
                size_t written = fwrite(voltage_buffer, sizeof(float), raw_voltage_data->num_channels * 2, fout);
                
                if (written != (size_t)(raw_voltage_data->num_channels * 2)) {
                    fprintf(stderr, "Write error for antenna %d pol %d at time %d\n", antenna_index, pol, t);
                    free(voltage_buffer);
                    fclose(fout);
                    return -1;
                }
                
                free(voltage_buffer);
            }
            
            fclose(fout);
        }
        
        // Update progress bar
        double progress = (double)(antenna_index + 1) / NUM_ANTENNAS;
        char progress_msg[64];
        snprintf(progress_msg, sizeof(progress_msg), "Writing raw voltages (%.1f MB/ant)", 
                (raw_voltage_data->num_output_samples * raw_voltage_data->num_channels * 2 * 2 * sizeof(float)) / (1024.0 * 1024.0));
        print_progress_bar(progress_msg, progress, 40);
    }
    printf("\n");
    return 0;
}

// ============================================================================
// TIME SLICING UTILITIES
// ============================================================================

/**
 * Calculate the output time range for time slicing after dedispersion
 * 
 * @param config          Configuration with start_time and stop_time
 * @param total_samples   Total number of input time samples
 * @param max_shift       Maximum dispersion shift in samples
 * @param start_sample    Output: starting sample index in output array
 * @param stop_sample     Output: stopping sample index in output array  
 * @param output_samples  Output: number of output samples needed
 */
void calculate_time_slice_range(const DedispersionConfig* config, int total_samples, int max_shift,
                               int* start_sample, int* stop_sample, int* output_samples) {
    int start_sample_requested, stop_sample_requested;
    
    // Calculate the valid output range after dedispersion FIRST
    // When we shift lower frequencies backwards in time by max_shift samples,
    // the valid output range becomes: 0 to (total_samples - 1 - max_shift)
    // because at times > (total_samples - 1 - max_shift), not all frequencies have valid data
    int max_valid_output_time = total_samples - 1 - max_shift;  // Latest time where all frequencies have valid data
    int min_valid_output_time = 0;                              // Earliest valid output time
    
    // Determine slicing mode: index-based or time-based
    if (config->start_sample_index >= 0) {
        // Index-based slicing (takes precedence)
        start_sample_requested = config->start_sample_index;
        printf("Using index-based start: sample %d\n", start_sample_requested);
    } else if (config->start_time > 0.0) {
        // Time-based slicing (user specified a start time)
        start_sample_requested = (int)(config->start_time / SAMPLE_TIME_SEC);
    } else {
        // DEFAULT: Start from beginning of valid dedispersed range
        start_sample_requested = min_valid_output_time;
    }
    
    if (config->stop_sample_index >= 0) {
        // Index-based slicing (takes precedence)
        stop_sample_requested = config->stop_sample_index;
        printf("Using index-based stop: sample %d\n", stop_sample_requested);
    } else if (config->stop_time >= 0.0) {
        // Time-based slicing (user specified a stop time)
        stop_sample_requested = (int)(config->stop_time / SAMPLE_TIME_SEC);
    } else {
        // DEFAULT: End at last valid dedispersed sample
        stop_sample_requested = max_valid_output_time;
    }
    
    // Ensure requested range is within bounds
    start_sample_requested = (start_sample_requested < 0) ? 0 : start_sample_requested;
    stop_sample_requested = (stop_sample_requested >= total_samples) ? total_samples - 1 : stop_sample_requested;
    
    // Apply valid dedispersed range constraints
    *start_sample = (start_sample_requested < min_valid_output_time) ? min_valid_output_time : start_sample_requested;
    *stop_sample = (stop_sample_requested > max_valid_output_time) ? max_valid_output_time : stop_sample_requested;
    
    // Ensure valid range
    if (*start_sample > *stop_sample) {
        fprintf(stderr, "Warning: Invalid time range after dedispersion correction\n");
        *start_sample = min_valid_output_time;
        *stop_sample = (*start_sample + 1000 > max_valid_output_time) ? max_valid_output_time : *start_sample + 1000; // Minimal range within bounds
    }
    
    *output_samples = *stop_sample - *start_sample + 1;
    
    printf("Time slicing configuration:\n");
    printf("  Requested time range: %.6f - %.6f seconds\n", config->start_time, config->stop_time);
    printf("  Input samples: %d - %d (of %d total)\n", start_sample_requested, stop_sample_requested, total_samples);
    printf("  Output samples after dedispersion: %d - %d (%d samples)\n", *start_sample, *stop_sample, *output_samples);
    printf("  Actual time range: %.6f - %.6f seconds\n", 
           (*start_sample) * SAMPLE_TIME_SEC, (*stop_sample) * SAMPLE_TIME_SEC);
}

// ============================================================================
// MAIN PROGRAM
// ============================================================================
/**
 * Entry point for the dedispersion program.
 * Parses command line arguments, sets up configuration, processes all subband files,
 * performs dedispersion and writes output files for each antenna.
 */
int main(int argc, char* argv[]) {
    // Configuration struct holds all runtime parameters
    DedispersionConfig config = {0};
    config.enable_dedispersion = 1; // Enable by default
    config.enable_median_subtraction = 0; // Disable by default
    config.block_processing_size = 1024; // Process 1024 blocks at once for better I/O efficiency
    config.num_threads = 0; // Auto-detect by default
    config.max_threads = 35; // Safe maximum for 40-core system
    config.candidate_name = NULL; // No candidate name by default
    config.output_mode = OUTPUT_MODE_TIME_SERIES; // Default to time series mode
    config.start_time = 0.0; // Start from beginning by default
    config.stop_time = -1.0; // Process until end by default
    config.start_sample_index = -1; // Use time-based slicing by default
    config.stop_sample_index = -1; // Use time-based slicing by default
    
    // Parse command line arguments for input/output directories and DM value
    int opt;
    while ((opt = getopt(argc, argv, "i:o:c:d:t:T:s:e:I:J:DMSPR")) != -1) {
        switch (opt) {
            case 'i': config.input_directory = optarg; break; // Input directory
            case 'o': config.output_directory = optarg; break; // Output directory
            case 'c': config.candidate_name = optarg; break; // Candidate name
            case 'd': config.dispersion_measure = atof(optarg); break; // DM value
            case 'D': config.enable_dedispersion = 0; break; // Disable dedispersion
            case 'M': config.enable_median_subtraction = 1; break; // Enable median subtraction
            case 't': config.num_threads = atoi(optarg); break; // Number of threads
            case 'T': config.max_threads = atoi(optarg); break; // Maximum threads allowed
            case 'S': config.output_mode = OUTPUT_MODE_SPECTRA; break; // Enable spectra mode
            case 'R': config.output_mode = OUTPUT_MODE_RAW; break; // Enable raw voltage mode
            case 'P': config.output_mode = OUTPUT_MODE_TIME_SERIES; break; // Enable time series mode (Power sum)
            case 's': config.start_time = atof(optarg); break; // Start time in seconds
            case 'e': config.stop_time = atof(optarg); break; // Stop time in seconds
            case 'I': config.start_sample_index = atoi(optarg); break; // Start sample index
            case 'J': config.stop_sample_index = atoi(optarg); break; // Stop sample index
            default:
                fprintf(stderr, "Usage: %s [MODE 1: -c <candidate>] [MODE 2: -i <input_dir> -o <output_dir>] -d <DM> [OPTIONS]\n", argv[0]);
                fprintf(stderr, "\nMODE 1 - Candidate name (automatic paths):\n");
                fprintf(stderr, "  -c <candidate>  Candidate name (auto-generates paths)\n");
                fprintf(stderr, "                  Input:  /dataz/dsa110/candidates/<candidate>/Level2/voltages\n");
                fprintf(stderr, "                  Output: <candidate>_out\n");
                fprintf(stderr, "\nMODE 2 - Manual paths:\n");
                fprintf(stderr, "  -i <input_dir>  Input directory containing voltage files\n");
                fprintf(stderr, "  -o <output_dir> Output directory for processed files\n");
                fprintf(stderr, "\nRequired:\n");
                fprintf(stderr, "  -d <DM>         Dispersion measure in pc/cm³\n");
                fprintf(stderr, "\nOptions:\n");
                fprintf(stderr, "  -D              Disable dedispersion\n");
                fprintf(stderr, "  -M              Enable median subtraction per frequency channel\n");
                fprintf(stderr, "  -t <threads>    Number of threads (0=auto-detect, default: auto)\n");
                fprintf(stderr, "  -T <max_threads> Maximum threads allowed (default: 35 for safety)\n");
                fprintf(stderr, "\nOutput Modes:\n");
                fprintf(stderr, "  -P              Time series mode (default): sum across frequency channels\n");
                fprintf(stderr, "  -S              Spectra mode: save all frequency channels\n");
                fprintf(stderr, "  -R              Raw voltage mode: save complex voltages per polarization\n");
                fprintf(stderr, "\nTime Slicing (optional):\n");
                fprintf(stderr, "  -s <seconds>    Start time in seconds (default: 0.0)\n");
                fprintf(stderr, "  -e <seconds>    End time in seconds (default: process until end)\n");
                fprintf(stderr, "  -I <index>      Start sample index (overrides -s if specified)\n");
                fprintf(stderr, "  -J <index>      Stop sample index (overrides -e if specified)\n");
                fprintf(stderr, "\nExamples:\n");
                fprintf(stderr, "  %s -c 250914hbqw -d 123.45                    # Use candidate name, time series\n", argv[0]);
                fprintf(stderr, "  %s -c 250914hbqw -d 123.45 -S                # Use candidate name, spectra mode\n", argv[0]);
                fprintf(stderr, "  %s -c 250914hbqw -d 123.45 -s 0.5 -e 0.6     # Time slice: 0.5s to 0.6s\n", argv[0]);
                fprintf(stderr, "  %s -c 250914hbqw -d 123.45 -I 15000 -J 18000 # Index slice: samples 15000-18000\n", argv[0]);
                fprintf(stderr, "  %s -i /path/to/voltages -o /path/to/output -d 123.45  # Manual paths\n", argv[0]);
                return 1;
        }
    }
    
    // Check if DM value was specified
    if (config.dispersion_measure == 0.0) {
        fprintf(stderr, "Error: Dispersion measure (-d) is required\n\n");
        fprintf(stderr, "Usage: %s [MODE 1: -c <candidate>] [MODE 2: -i <input_dir> -o <output_dir>] -d <DM> [OPTIONS]\n", argv[0]);
        fprintf(stderr, "\nMODE 1 - Candidate name (automatic paths):\n");
        fprintf(stderr, "  -c <candidate>  Candidate name (auto-generates paths)\n");
        fprintf(stderr, "                  Input:  /dataz/dsa110/candidates/<candidate>/Level2/voltages\n");
        fprintf(stderr, "                  Output: <candidate>_out\n");
        fprintf(stderr, "\nMODE 2 - Manual paths:\n");
        fprintf(stderr, "  -i <input_dir>  Input directory containing voltage files\n");
        fprintf(stderr, "  -o <output_dir> Output directory for processed files\n");
        fprintf(stderr, "\nRequired:\n");
        fprintf(stderr, "  -d <DM>         Dispersion measure in pc/cm³\n");
        fprintf(stderr, "\nOptions:\n");
        fprintf(stderr, "  -D              Disable dedispersion\n");
        fprintf(stderr, "  -M              Enable median subtraction per frequency channel\n");
        fprintf(stderr, "  -t <threads>    Number of threads (0=auto-detect, default: auto)\n");
        fprintf(stderr, "  -T <max_threads> Maximum threads allowed (default: 35 for safety)\n");
        fprintf(stderr, "\nOutput Modes:\n");
        fprintf(stderr, "  -P              Time series mode (default): sum across frequency channels\n");
        fprintf(stderr, "  -S              Spectra mode: save all frequency channels\n");
        fprintf(stderr, "  -R              Raw voltage mode: save complex voltages per polarization\n");
        fprintf(stderr, "\nTime Slicing (optional):\n");
        fprintf(stderr, "  -s <seconds>    Start time in seconds (default: 0.0)\n");
        fprintf(stderr, "  -e <seconds>    End time in seconds (default: process until end)\n");
        fprintf(stderr, "  -I <index>      Start sample index (overrides -s if specified)\n");
        fprintf(stderr, "  -J <index>      Stop sample index (overrides -e if specified)\n");
        fprintf(stderr, "\nExamples:\n");
        fprintf(stderr, "  %s -c 250914hbqw -d 123.45                    # Use candidate name, time series\n", argv[0]);
        fprintf(stderr, "  %s -c 250914hbqw -d 123.45 -S                # Use candidate name, spectra mode\n", argv[0]);
        fprintf(stderr, "  %s -c 250914hbqw -d 123.45 -s 0.5 -e 0.6     # Time slice: 0.5s to 0.6s\n", argv[0]);
        fprintf(stderr, "  %s -c 250914hbqw -d 123.45 -I 15000 -J 18000 # Index slice: samples 15000-18000\n", argv[0]);
        fprintf(stderr, "  %s -i /path/to/voltages -o /path/to/output -d 123.45  # Manual paths\n", argv[0]);
        return 1;
    }
    
    // Handle directory setup: either from candidate name or manual specification
    int dir_setup_result = setup_directories_from_candidate(&config);
    if (dir_setup_result < 0) {
        // Error in candidate directory setup
        return 1;
    } else if (dir_setup_result == 0) {
        // No candidate name provided, check for manual directories
        if (!config.input_directory || !config.output_directory) {
            fprintf(stderr, "Error: Must specify either candidate name (-c) OR both input (-i) and output (-o) directories\n\n");
            fprintf(stderr, "Usage: %s [MODE 1: -c <candidate>] [MODE 2: -i <input_dir> -o <output_dir>] -d <DM> [OPTIONS]\n", argv[0]);
            fprintf(stderr, "\nMODE 1 - Candidate name (automatic paths):\n");
            fprintf(stderr, "  -c <candidate>  Candidate name (auto-generates paths)\n");
            fprintf(stderr, "                  Input:  /dataz/dsa110/candidates/<candidate>/Level2/voltages\n");
            fprintf(stderr, "                  Output: <candidate>_out\n");
            fprintf(stderr, "\nMODE 2 - Manual paths:\n");
            fprintf(stderr, "  -i <input_dir>  Input directory containing voltage files\n");
            fprintf(stderr, "  -o <output_dir> Output directory for processed files\n");
            fprintf(stderr, "\nRequired:\n");
            fprintf(stderr, "  -d <DM>         Dispersion measure in pc/cm³\n");
            fprintf(stderr, "\nOptions:\n");
            fprintf(stderr, "  -D              Disable dedispersion\n");
            fprintf(stderr, "  -M              Enable median subtraction per frequency channel\n");
            fprintf(stderr, "  -t <threads>    Number of threads (0=auto-detect, default: auto)\n");
            fprintf(stderr, "  -T <max_threads> Maximum threads allowed (default: 35 for safety)\n");
            fprintf(stderr, "\nExamples:\n");
            fprintf(stderr, "  %s -c 250111wwww -d 123.45         # Use candidate name\n", argv[0]);
            fprintf(stderr, "  %s -i /path/to/voltages -o /path/to/output -d 123.45  # Manual paths\n", argv[0]);
            return 1;
        }
        printf("Using manual directory specification:\n");
        printf("  Input:  %s\n", config.input_directory);
        printf("  Output: %s\n", config.output_directory);
    }
    
    // Check for conflicting arguments
    if (config.candidate_name && (config.input_directory || config.output_directory)) {
        // Only warn if the user manually specified directories that differ from auto-generated ones
        char expected_input[512], expected_output[512];
        snprintf(expected_input, 512, "/dataz/dsa110/candidates/%s/Level2/voltages", config.candidate_name);
        snprintf(expected_output, 512, "%s_out", config.candidate_name);
        
        if ((config.input_directory && strcmp(config.input_directory, expected_input) != 0) ||
            (config.output_directory && strcmp(config.output_directory, expected_output) != 0)) {
            fprintf(stderr, "Warning: Candidate name (-c) overrides manual directory specifications (-i/-o)\n");
        }
    }
    
    // Check for incompatible mode combinations
    if (config.enable_median_subtraction && (config.output_mode == OUTPUT_MODE_SPECTRA || config.output_mode == OUTPUT_MODE_RAW)) {
        fprintf(stderr, "Error: Median subtraction (-M) cannot be used with spectra mode (-S) or raw voltage mode (-R).\n");
        fprintf(stderr, "This is because median subtraction processes data differently and conflicts with spectra/voltage accumulation.\n");
        fprintf(stderr, "Please choose one mode:\n");
        fprintf(stderr, "  For spectra output: Use -S without -M\n");
        fprintf(stderr, "  For raw voltage output: Use -R without -M\n");
        fprintf(stderr, "  For median-subtracted time series: Use -M without -S or -R\n");
        cleanup_config_directories(&config);
        return 1;
    }
    
    // Ensure output directory exists
    ensure_directory_exists(config.output_directory);
    
    // List and sort all subband files in the input directory
    SubbandFileEntry sb_files[NUM_SUBBANDS];
    int n_sb = list_subband_files(config.input_directory, sb_files, NUM_SUBBANDS);
    if (n_sb != NUM_SUBBANDS) {
        fprintf(stderr, "Expected %d subband files, found %d in %s\n", NUM_SUBBANDS, n_sb, config.input_directory);
        cleanup_config_directories(&config);
        return 1;
    }
    printf("Found %d subband files:\n", n_sb);
    for (int i = 0; i < n_sb; ++i) {
        // Extract just the filename from the full path for cleaner display
        char *filename = strrchr(sb_files[i].name, '/');
        if (filename) {
            filename++; // Skip the '/' character
        } else {
            filename = sb_files[i].name; // No path separator found, use full string
        }
        printf("  %d: %s\n", i, filename);
    }
    
    // Detect number of time samples from the first subband file
    int total_time_samples = detect_time_samples_from_file(sb_files[0].name);
    if (total_time_samples <= 0) {
        fprintf(stderr, "Cannot determine time samples from files\n");
        cleanup_config_directories(&config);
        return 1;
    }
    
    // Set up multi-threading with system load and memory awareness
    int optimal_threads = determine_optimal_threads(config.num_threads, config.max_threads, &config, total_time_samples);
    setup_openmp_environment(optimal_threads);
    
    // Calculate dispersion parameters for all channels
    DispersionParams* disp_params = calculate_dispersion_params(config.dispersion_measure, total_time_samples);
    if (!disp_params) {
        fprintf(stderr, "Failed to calculate dispersion parameters\n");
        cleanup_config_directories(&config);
        return 1;
    }
    // If DM is zero or max_shift is zero, disable dedispersion
    if (disp_params->max_shift == 0 && config.enable_dedispersion) {
        printf("WARNING: Max dispersion shift is 0 samples - disabling dedispersion\n");
        config.enable_dedispersion = 0;
    }
    // Calculate time slicing range and output samples
    int start_sample, stop_sample, output_samples;
    calculate_time_slice_range(&config, total_time_samples, disp_params->max_shift,
                              &start_sample, &stop_sample, &output_samples);
    
    if (output_samples <= 0) {
        fprintf(stderr, "Invalid time range - no output samples remain\n");
        free_dispersion_params(disp_params);
        cleanup_config_directories(&config);
        return 1;
    }
    printf("Processing configuration:\n");
    printf("  DM: %.2f pc/cm³\n", config.dispersion_measure);
    printf("  Dedispersion: %s\n", config.enable_dedispersion ? "enabled" : "disabled");
    printf("  Median subtraction: %s\n", config.enable_median_subtraction ? "enabled" : "disabled");
    printf("  Max dispersion shift: %d samples\n", disp_params->max_shift);
    printf("  Output samples: %d\n", output_samples);
    printf("  Threads: %d\n", optimal_threads);
    
    // Display SIMD capabilities
    printf("Optimization features:\n");
    printf("  SIMD: ");
    if (has_avx512()) {
        printf("AVX-512 (32 samples/vector, FMA)\n");
    } else if (has_avx2()) {
        printf("AVX2 (16 samples/vector)\n");
    } else if (has_sse2()) {
        printf("SSE2 (8 samples/vector)\n");
    } else if (has_neon()) {
        printf("ARM NEON (16 samples/vector)\n");
    } else {
        printf("Scalar (no vectorization)\n");
    }
    
    #ifdef _OPENMP
    printf("  Multi-threading: OpenMP with %d threads\n", optimal_threads);
    #else
    printf("  Multi-threading: Disabled (OpenMP not available)\n");
    #endif
    
    printf("\n");
    
    // Allocate memory for antenna data accumulation
    AntennaData* antenna_data = allocate_antenna_data(output_samples);
    if (!antenna_data) {
        fprintf(stderr, "Failed to allocate antenna data buffers\n");
        free_dispersion_params(disp_params);
        cleanup_config_directories(&config);
        return 1;
    }
    
    // For subband-by-subband processing, we'll allocate spectra_data inside the loop
    SpectraData* spectra_data = NULL;
    
    // Process each subband file and accumulate power for all antennas
    printf("Processing %d subband files:\n", n_sb);
    
    // Performance timing
    #ifdef _OPENMP
    double start_time = omp_get_wtime();
    #else
    double start_time = 0.0;
    (void)start_time; // Suppress unused variable warning
    #endif
    
    // Subband-by-subband processing for maximum memory efficiency
    if (config.output_mode == OUTPUT_MODE_SPECTRA) {
        // Process each subband individually with 16× memory reduction
        for (int sb = 0; sb < n_sb; ++sb) {
            // Allocate memory for single subband (384 channels instead of 6,144)
            spectra_data = allocate_spectra_data(output_samples, NUM_CHANNELS_PER_SUBBAND, sb);
            if (!spectra_data) {
                fprintf(stderr, "Failed to allocate spectra data for subband %d\n", sb);
                free_antenna_data(antenna_data);
                free_dispersion_params(disp_params);
                cleanup_config_directories(&config);
                return 1;
            }
            
            // Process this subband
            int result = process_subband_file(sb_files[sb].name, sb, disp_params, 
                                            antenna_data, spectra_data, NULL, &config, 
                                            start_sample, stop_sample);
            if (result < 0) {
                fprintf(stderr, "Failed to process subband %d\n", sb);
                free_spectra_data(spectra_data);
                free_antenna_data(antenna_data);
                free_dispersion_params(disp_params);
                cleanup_config_directories(&config);
                return 1;
            }
            
            // Write spectra files for this subband (Option B: ant_YY_subband_XX_stokes_i.spec)
            if (write_spectra_files(config.output_directory, spectra_data, disp_params, &config) < 0) {
                fprintf(stderr, "Failed to write spectra files for subband %d\n", sb);
                free_spectra_data(spectra_data);
                free_antenna_data(antenna_data);
                free_dispersion_params(disp_params);
                cleanup_config_directories(&config);
                return 1;
            }
            
            // Free memory for this subband before processing next
            free_spectra_data(spectra_data);
            spectra_data = NULL;
            
            // Update overall progress
            double progress = (double)(sb + 1) / n_sb;
            char progress_msg[64];
            snprintf(progress_msg, sizeof(progress_msg), "Subband-by-subband progress (%d/%d)", sb + 1, n_sb);
            print_progress_bar(progress_msg, progress, 50);
        }
    } else if (config.output_mode == OUTPUT_MODE_RAW) {
        // RAW voltage processing (subband-by-subband like SPECTRA)
        for (int sb = 0; sb < n_sb; ++sb) {
            // Allocate memory for single subband raw voltage data
            RawVoltageData* raw_voltage_data = allocate_raw_voltage_data(output_samples, NUM_CHANNELS_PER_SUBBAND, sb);
            if (!raw_voltage_data) {
                fprintf(stderr, "Failed to allocate raw voltage data for subband %d\n", sb);
                free_antenna_data(antenna_data);
                free_dispersion_params(disp_params);
                cleanup_config_directories(&config);
                return 1;
            }
            
            // Process this subband
            int result = process_subband_file(sb_files[sb].name, sb, disp_params, 
                                            antenna_data, NULL, raw_voltage_data, &config, 
                                            start_sample, stop_sample);
            if (result < 0) {
                fprintf(stderr, "Failed to process subband %d\n", sb);
                free_raw_voltage_data(raw_voltage_data);
                free_antenna_data(antenna_data);
                free_dispersion_params(disp_params);
                cleanup_config_directories(&config);
                return 1;
            }
            
            // Write raw voltage files for this subband
            if (write_raw_voltage_files(config.output_directory, raw_voltage_data, disp_params, &config, sb) < 0) {
                fprintf(stderr, "Failed to write raw voltage files for subband %d\n", sb);
                free_raw_voltage_data(raw_voltage_data);
                free_antenna_data(antenna_data);
                free_dispersion_params(disp_params);
                cleanup_config_directories(&config);
                return 1;
            }
            
            // Free memory for this subband before processing next
            free_raw_voltage_data(raw_voltage_data);
            
            // Update overall progress
            double progress = (double)(sb + 1) / n_sb;
            char progress_msg[64];
            snprintf(progress_msg, sizeof(progress_msg), "Raw voltage subband progress (%d/%d)", sb + 1, n_sb);
            print_progress_bar(progress_msg, progress, 50);
        }
    } else {
        // Standard time-series processing (all subbands together)
        for (int sb = 0; sb < n_sb; ++sb) {
            int result = process_subband_file(sb_files[sb].name, sb, disp_params, 
                                            antenna_data, spectra_data, NULL, &config, 
                                            start_sample, stop_sample);
            if (result < 0) {
                fprintf(stderr, "Failed to process subband %d\n", sb);
                free_antenna_data(antenna_data);
                free_dispersion_params(disp_params);
                cleanup_config_directories(&config);
                return 1;
            }
            
            // Update overall progress
            double progress = (double)(sb + 1) / n_sb;
            char progress_msg[64];
            snprintf(progress_msg, sizeof(progress_msg), "Overall progress (%d/%d subbands)", sb + 1, n_sb);
            print_progress_bar(progress_msg, progress, 50);
        }
    }
    printf("\n");
    
    #ifdef _OPENMP
    double processing_time = omp_get_wtime() - start_time;
    #else
    double processing_time = 0.0;
    #endif
    
    // Write output files for each antenna
    #ifdef _OPENMP
    double write_start = omp_get_wtime();
    #else
    double write_start = 0.0;
    (void)write_start; // Suppress unused variable warning
    #endif
    // Write output files based on mode
    if (config.output_mode == OUTPUT_MODE_SPECTRA) {
        // Spectra files already written in subband-by-subband mode
        printf("Spectra files written (subband-by-subband processing complete)\n");
    } else if (config.output_mode == OUTPUT_MODE_RAW) {
        // Raw voltage files already written in subband-by-subband mode
        printf("Raw voltage files written (subband-by-subband processing complete)\n");
    } else {
        if (write_output_files(config.output_directory, antenna_data, disp_params, &config, start_sample) < 0) {
            fprintf(stderr, "Failed to write output files\n");
            free_antenna_data(antenna_data);
            if (spectra_data) free_spectra_data(spectra_data);
            free_dispersion_params(disp_params);
            cleanup_config_directories(&config);
            return 1;
        }
    }
    #ifdef _OPENMP
    double write_time = omp_get_wtime() - write_start;
    double total_time = omp_get_wtime() - start_time;
    #else
    double write_time = 0.0;
    double total_time = 0.0;
    #endif
    
    // Performance summary
    printf("\nPerformance Summary:\n");
    printf("  Processing time: %.2f seconds\n", processing_time);
    printf("  Output writing: %.2f seconds\n", write_time);
    printf("  Total time: %.2f seconds\n", total_time);
    printf("  Data rate: %.2f MB/s\n", (n_sb * 16 * 1024.0) / total_time); // Rough estimate
    printf("  Threads utilized: %d/%d cores\n", optimal_threads, get_cpu_count());
    
    printf("Success! Processed %d subbands, %d antennas, %d time samples\n", 
           n_sb, NUM_ANTENNAS, total_time_samples);
    
    // Free all allocated memory
    free_antenna_data(antenna_data);
    if (spectra_data) free_spectra_data(spectra_data);
    free_dispersion_params(disp_params);
    cleanup_config_directories(&config);
    return 0;
}
