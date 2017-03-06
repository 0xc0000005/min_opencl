#![allow(non_camel_case_types,
         dead_code,
         unused_variables,
         improper_ctypes,
         non_upper_case_globals)]

use libc::{c_void, size_t, c_char};

pub type cl_char = ::std::os::raw::c_char;
pub type cl_uchar = ::std::os::raw::c_uchar;
pub type cl_short = ::std::os::raw::c_short;
pub type cl_ushort = ::std::os::raw::c_ushort;
pub type cl_int = ::std::os::raw::c_int;
pub type cl_uint = ::std::os::raw::c_uint;
pub type cl_long = ::std::os::raw::c_longlong;
pub type cl_ulong = ::std::os::raw::c_ulonglong;
pub type cl_half = ::std::os::raw::c_ushort;
pub type cl_float = ::std::os::raw::c_float;
pub type cl_double = ::std::os::raw::c_double;
pub type intptr_t = ::std::os::raw::c_longlong;
pub type ptrdiff_t = ::std::os::raw::c_longlong;
pub type cl_bool = cl_uint;
pub type cl_bitfield = cl_ulong;
pub type cl_device_type = cl_bitfield;
pub type cl_platform_info = cl_uint;
pub type cl_device_info = cl_uint;
pub type cl_device_fp_config = cl_bitfield;
pub type cl_device_mem_cache_type = cl_uint;
pub type cl_device_local_mem_type = cl_uint;
pub type cl_device_exec_capabilities = cl_bitfield;
pub type cl_device_svm_capabilities = cl_bitfield;
pub type cl_command_queue_properties = cl_bitfield;
pub type cl_device_partition_property = intptr_t;
pub type cl_device_affinity_domain = cl_bitfield;
pub type cl_context_properties = intptr_t;
pub type cl_context_info = cl_uint;
pub type cl_queue_properties = cl_bitfield;
pub type cl_command_queue_info = cl_uint;
pub type cl_channel_order = cl_uint;
pub type cl_channel_type = cl_uint;
pub type cl_mem_flags = cl_bitfield;
pub type cl_svm_mem_flags = cl_bitfield;
pub type cl_mem_object_type = cl_uint;
pub type cl_mem_info = cl_uint;
pub type cl_mem_migration_flags = cl_bitfield;
pub type cl_image_info = cl_uint;
pub type cl_buffer_create_type = cl_uint;
pub type cl_addressing_mode = cl_uint;
pub type cl_filter_mode = cl_uint;
pub type cl_sampler_info = cl_uint;
pub type cl_map_flags = cl_bitfield;
pub type cl_pipe_properties = intptr_t;
pub type cl_pipe_info = cl_uint;
pub type cl_program_info = cl_uint;
pub type cl_program_build_info = cl_uint;
pub type cl_program_binary_type = cl_uint;
pub type cl_build_status = cl_int;
pub type cl_kernel_info = cl_uint;
pub type cl_kernel_arg_info = cl_uint;
pub type cl_kernel_arg_address_qualifier = cl_uint;
pub type cl_kernel_arg_access_qualifier = cl_uint;
pub type cl_kernel_arg_type_qualifier = cl_bitfield;
pub type cl_kernel_work_group_info = cl_uint;
pub type cl_event_info = cl_uint;
pub type cl_command_type = cl_uint;
pub type cl_profiling_info = cl_uint;
pub type cl_sampler_properties = cl_bitfield;
pub type cl_kernel_exec_info = cl_uint;

pub type cl_platform_id = *mut c_void;
pub type cl_device_id = *mut c_void;
pub type cl_context = *mut c_void;
pub type cl_command_queue = *mut c_void;
pub type cl_mem = *mut c_void;
pub type cl_program = *mut c_void;
pub type cl_kernel = *mut c_void;
pub type cl_event = *mut c_void;
pub type cl_sampler = *mut c_void;

pub type fn_notify = Option<unsafe extern "C" fn(arg1: *const c_char,
                                                 arg2: *const c_void,
                                                 arg3: size_t,
                                                 arg4: *mut c_void)
                                                >;

#[repr(C)]
pub struct cl_image_format {
    pub image_channel_order: cl_channel_order,
    pub image_channel_data_type: cl_channel_type,
}
#[repr(C)]
pub struct cl_image_desc {
    pub image_type: cl_mem_object_type,
    pub image_width: size_t,
    pub image_height: size_t,
    pub image_depth: size_t,
    pub image_array_size: size_t,
    pub image_row_pitch: size_t,
    pub image_slice_pitch: size_t,
    pub num_mip_levels: cl_uint,
    pub num_samples: cl_uint,
    pub _bindgen_data_1_: [u64; 1usize],
}

// Error Codes
pub const CL_SUCCESS: cl_int = 0;
pub const CL_DEVICE_NOT_FOUND: cl_int = -1;
pub const CL_DEVICE_NOT_AVAILABLE: cl_int = -2;
pub const CL_COMPILER_NOT_AVAILABLE: cl_int = -3;
pub const CL_MEM_OBJECT_ALLOCATION_FAILURE: cl_int = -4;
pub const CL_OUT_OF_RESOURCES: cl_int = -5;
pub const CL_OUT_OF_HOST_MEMORY: cl_int = -6;
pub const CL_PROFILING_INFO_NOT_AVAILABLE: cl_int = -7;
pub const CL_MEM_COPY_OVERLAP: cl_int = -8;
pub const CL_IMAGE_FORMAT_MISMATCH: cl_int = -9;
pub const CL_IMAGE_FORMAT_NOT_SUPPORTED: cl_int = -10;
pub const CL_BUILD_PROGRAM_FAILURE: cl_int = -11;
pub const CL_MAP_FAILURE: cl_int = -12;
pub const CL_MISALIGNED_SUB_BUFFER_OFFSET: cl_int = -13;
pub const CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST: cl_int = -14;
pub const CL_COMPILE_PROGRAM_FAILURE: cl_int = -15;
pub const CL_LINKER_NOT_AVAILABLE: cl_int = -16;
pub const CL_LINK_PROGRAM_FAILURE: cl_int = -17;
pub const CL_DEVICE_PARTITION_FAILED: cl_int = -18;
pub const CL_KERNEL_ARG_INFO_NOT_AVAILABLE: cl_int = -19;

pub const CL_INVALID_VALUE: cl_int = -30;
pub const CL_INVALID_DEVICE_TYPE: cl_int = -31;
pub const CL_INVALID_PLATFORM: cl_int = -32;
pub const CL_INVALID_DEVICE: cl_int = -33;
pub const CL_INVALID_CONTEXT: cl_int = -34;
pub const CL_INVALID_QUEUE_PROPERTIES: cl_int = -35;
pub const CL_INVALID_COMMAND_QUEUE: cl_int = -36;
pub const CL_INVALID_HOST_PTR: cl_int = -37;
pub const CL_INVALID_MEM_OBJECT: cl_int = -38;
pub const CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: cl_int = -39;
pub const CL_INVALID_IMAGE_SIZE: cl_int = -40;
pub const CL_INVALID_SAMPLER: cl_int = -41;
pub const CL_INVALID_BINARY: cl_int = -42;
pub const CL_INVALID_BUILD_OPTIONS: cl_int = -43;
pub const CL_INVALID_PROGRAM: cl_int = -44;
pub const CL_INVALID_PROGRAM_EXECUTABLE: cl_int = -45;
pub const CL_INVALID_KERNEL_NAME: cl_int = -46;
pub const CL_INVALID_KERNEL_DEFINITION: cl_int = -47;
pub const CL_INVALID_KERNEL: cl_int = -48;
pub const CL_INVALID_ARG_INDEX: cl_int = -49;
pub const CL_INVALID_ARG_VALUE: cl_int = -50;
pub const CL_INVALID_ARG_SIZE: cl_int = -51;
pub const CL_INVALID_KERNEL_ARGS: cl_int = -52;
pub const CL_INVALID_WORK_DIMENSION: cl_int = -53;
pub const CL_INVALID_WORK_GROUP_SIZE: cl_int = -54;
pub const CL_INVALID_WORK_ITEM_SIZE: cl_int = -55;
pub const CL_INVALID_GLOBAL_OFFSET: cl_int = -56;
pub const CL_INVALID_EVENT_WAIT_LIST: cl_int = -57;
pub const CL_INVALID_EVENT: cl_int = -58;
pub const CL_INVALID_OPERATION: cl_int = -59;
pub const CL_INVALID_GL_OBJECT: cl_int = -60;
pub const CL_INVALID_BUFFER_SIZE: cl_int = -61;
pub const CL_INVALID_MIP_LEVEL: cl_int = -62;
pub const CL_INVALID_GLOBAL_WORK_SIZE: cl_int = -63;
pub const CL_INVALID_PROPERTY: cl_int = -64;
pub const CL_INVALID_IMAGE_DESCRIPTOR: cl_int = -65;
pub const CL_INVALID_COMPILER_OPTIONS: cl_int = -66;
pub const CL_INVALID_LINKER_OPTIONS: cl_int = -67;
pub const CL_INVALID_DEVICE_PARTITION_COUNT: cl_int = -68;
pub const CL_INVALID_PIPE_SIZE: cl_int = -69;
pub const CL_INVALID_DEVICE_QUEUE: cl_int = -70;

// cl_bool
pub const CL_FALSE: cl_bool = 0;
pub const CL_TRUE: cl_bool = 1;
pub const CL_BLOCKING: cl_bool = CL_TRUE;
pub const CL_NON_BLOCKING: cl_bool = CL_FALSE;

// cl_platform_info
pub const CL_PLATFORM_PROFILE: cl_platform_info = 0x0900;
pub const CL_PLATFORM_VERSION: cl_platform_info = 0x0901;
pub const CL_PLATFORM_NAME: cl_platform_info = 0x0902;
pub const CL_PLATFORM_VENDOR: cl_platform_info = 0x0903;
pub const CL_PLATFORM_EXTENSIONS: cl_platform_info = 0x0904;

// cl_device_type - bitfield
pub const CL_DEVICE_TYPE_DEFAULT: cl_device_type = (1 << 0);
pub const CL_DEVICE_TYPE_CPU: cl_device_type = (1 << 1);
pub const CL_DEVICE_TYPE_GPU: cl_device_type = (1 << 2);
pub const CL_DEVICE_TYPE_ACCELERATOR: cl_device_type = (1 << 3);
pub const CL_DEVICE_TYPE_CUSTOM: cl_device_type = (1 << 4);
pub const CL_DEVICE_TYPE_ALL: cl_device_type = 0xFFFFFFFF;

// cl_device_info
pub const CL_DEVICE_TYPE: cl_device_info = 0x1000;
pub const CL_DEVICE_VENDOR_ID: cl_device_info = 0x1001;
pub const CL_DEVICE_MAX_COMPUTE_UNITS: cl_device_info = 0x1002;
pub const CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS: cl_device_info = 0x1003;
pub const CL_DEVICE_MAX_WORK_GROUP_SIZE: cl_device_info = 0x1004;
pub const CL_DEVICE_MAX_WORK_ITEM_SIZES: cl_device_info = 0x1005;
pub const CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR: cl_device_info = 0x1006;
pub const CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT: cl_device_info = 0x1007;
pub const CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT: cl_device_info = 0x1008;
pub const CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG: cl_device_info = 0x1009;
pub const CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT: cl_device_info = 0x100A;
pub const CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE: cl_device_info = 0x100B;
pub const CL_DEVICE_MAX_CLOCK_FREQUENCY: cl_device_info = 0x100C;
pub const CL_DEVICE_ADDRESS_BITS: cl_device_info = 0x100D;
pub const CL_DEVICE_MAX_READ_IMAGE_ARGS: cl_device_info = 0x100E;
pub const CL_DEVICE_MAX_WRITE_IMAGE_ARGS: cl_device_info = 0x100F;
pub const CL_DEVICE_MAX_MEM_ALLOC_SIZE: cl_device_info = 0x1010;
pub const CL_DEVICE_IMAGE2D_MAX_WIDTH: cl_device_info = 0x1011;
pub const CL_DEVICE_IMAGE2D_MAX_HEIGHT: cl_device_info = 0x1012;
pub const CL_DEVICE_IMAGE3D_MAX_WIDTH: cl_device_info = 0x1013;
pub const CL_DEVICE_IMAGE3D_MAX_HEIGHT: cl_device_info = 0x1014;
pub const CL_DEVICE_IMAGE3D_MAX_DEPTH: cl_device_info = 0x1015;
pub const CL_DEVICE_IMAGE_SUPPORT: cl_device_info = 0x1016;
pub const CL_DEVICE_MAX_PARAMETER_SIZE: cl_device_info = 0x1017;
pub const CL_DEVICE_MAX_SAMPLERS: cl_device_info = 0x1018;
pub const CL_DEVICE_MEM_BASE_ADDR_ALIGN: cl_device_info = 0x1019;
pub const CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE: cl_device_info = 0x101A;
pub const CL_DEVICE_SINGLE_FP_CONFIG: cl_device_info = 0x101B;
pub const CL_DEVICE_GLOBAL_MEM_CACHE_TYPE: cl_device_info = 0x101C;
pub const CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE: cl_device_info = 0x101D;
pub const CL_DEVICE_GLOBAL_MEM_CACHE_SIZE: cl_device_info = 0x101E;
pub const CL_DEVICE_GLOBAL_MEM_SIZE: cl_device_info = 0x101F;
pub const CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE: cl_device_info = 0x1020;
pub const CL_DEVICE_MAX_CONSTANT_ARGS: cl_device_info = 0x1021;
pub const CL_DEVICE_LOCAL_MEM_TYPE: cl_device_info = 0x1022;
pub const CL_DEVICE_LOCAL_MEM_SIZE: cl_device_info = 0x1023;
pub const CL_DEVICE_ERROR_CORRECTION_SUPPORT: cl_device_info = 0x1024;
pub const CL_DEVICE_PROFILING_TIMER_RESOLUTION: cl_device_info = 0x1025;
pub const CL_DEVICE_ENDIAN_LITTLE: cl_device_info = 0x1026;
pub const CL_DEVICE_AVAILABLE: cl_device_info = 0x1027;
pub const CL_DEVICE_COMPILER_AVAILABLE: cl_device_info = 0x1028;
pub const CL_DEVICE_EXECUTION_CAPABILITIES: cl_device_info = 0x1029;
pub const CL_DEVICE_QUEUE_PROPERTIES: cl_device_info = 0x102A;  // deprecated
pub const CL_DEVICE_QUEUE_ON_HOST_PROPERTIES: cl_device_info = 0x102A;
pub const CL_DEVICE_NAME: cl_device_info = 0x102B;
pub const CL_DEVICE_VENDOR: cl_device_info = 0x102C;
pub const CL_DRIVER_VERSION: cl_device_info = 0x102D;
pub const CL_DEVICE_PROFILE: cl_device_info = 0x102E;
pub const CL_DEVICE_VERSION: cl_device_info = 0x102F;
pub const CL_DEVICE_EXTENSIONS: cl_device_info = 0x1030;
pub const CL_DEVICE_PLATFORM: cl_device_info = 0x1031;
pub const CL_DEVICE_DOUBLE_FP_CONFIG: cl_device_info = 0x1032;
// 0x1033 reserved for CL_DEVICE_HALF_FP_CONFIG
pub const CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF: cl_device_info = 0x1034;
pub const CL_DEVICE_HOST_UNIFIED_MEMORY: cl_device_info = 0x1035; // deprecated
pub const CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR: cl_device_info = 0x1036;
pub const CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT: cl_device_info = 0x1037;
pub const CL_DEVICE_NATIVE_VECTOR_WIDTH_INT: cl_device_info = 0x1038;
pub const CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG: cl_device_info = 0x1039;
pub const CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT: cl_device_info = 0x103A;
pub const CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE: cl_device_info = 0x103B;
pub const CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF: cl_device_info = 0x103C;
pub const CL_DEVICE_OPENCL_C_VERSION: cl_device_info = 0x103D;
pub const CL_DEVICE_LINKER_AVAILABLE: cl_device_info = 0x103E;
pub const CL_DEVICE_BUILT_IN_KERNELS: cl_device_info = 0x103F;
pub const CL_DEVICE_IMAGE_MAX_BUFFER_SIZE: cl_device_info = 0x1040;
pub const CL_DEVICE_IMAGE_MAX_ARRAY_SIZE: cl_device_info = 0x1041;
pub const CL_DEVICE_PARENT_DEVICE: cl_device_info = 0x1042;
pub const CL_DEVICE_PARTITION_MAX_SUB_DEVICES: cl_device_info = 0x1043;
pub const CL_DEVICE_PARTITION_PROPERTIES: cl_device_info = 0x1044;
pub const CL_DEVICE_PARTITION_AFFINITY_DOMAIN: cl_device_info = 0x1045;
pub const CL_DEVICE_PARTITION_TYPE: cl_device_info = 0x1046;
pub const CL_DEVICE_REFERENCE_COUNT: cl_device_info = 0x1047;
pub const CL_DEVICE_PREFERRED_INTEROP_USER_SYNC: cl_device_info = 0x1048;
pub const CL_DEVICE_PRINTF_BUFFER_SIZE: cl_device_info = 0x1049;
pub const CL_DEVICE_IMAGE_PITCH_ALIGNMENT: cl_device_info = 0x104A;
pub const CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT: cl_device_info = 0x104B;
pub const CL_DEVICE_MAX_READ_WRITE_IMAGE_ARGS: cl_device_info = 0x104C;
pub const CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE: cl_device_info = 0x104D;
pub const CL_DEVICE_QUEUE_ON_DEVICE_PROPERTIES: cl_device_info = 0x104E;
pub const CL_DEVICE_QUEUE_ON_DEVICE_PREFERRED_SIZE: cl_device_info = 0x104F;
pub const CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE: cl_device_info = 0x1050;
pub const CL_DEVICE_MAX_ON_DEVICE_QUEUES: cl_device_info = 0x1051;
pub const CL_DEVICE_MAX_ON_DEVICE_EVENTS: cl_device_info = 0x1052;
pub const CL_DEVICE_SVM_CAPABILITIES: cl_device_info = 0x1053;
pub const CL_DEVICE_GLOBAL_VARIABLE_PREFERRED_TOTAL_SIZE: cl_device_info = 0x1054;
pub const CL_DEVICE_MAX_PIPE_ARGS: cl_device_info = 0x1055;
pub const CL_DEVICE_PIPE_MAX_ACTIVE_RESERVATIONS: cl_device_info = 0x1056;
pub const CL_DEVICE_PIPE_MAX_PACKET_SIZE: cl_device_info = 0x1057;
pub const CL_DEVICE_PREFERRED_PLATFORM_ATOMIC_ALIGNMENT: cl_device_info = 0x1058;
pub const CL_DEVICE_PREFERRED_GLOBAL_ATOMIC_ALIGNMENT: cl_device_info = 0x1059;
pub const CL_DEVICE_PREFERRED_LOCAL_ATOMIC_ALIGNMENT: cl_device_info = 0x105A;

// cl_device_fp_config - bitfield
pub const CL_FP_DENORM: cl_device_fp_config = (1 << 0);
pub const CL_FP_INF_NAN: cl_device_fp_config = (1 << 1);
pub const CL_FP_ROUND_TO_NEAREST: cl_device_fp_config = (1 << 2);
pub const CL_FP_ROUND_TO_ZERO: cl_device_fp_config = (1 << 3);
pub const CL_FP_ROUND_TO_INF: cl_device_fp_config = (1 << 4);
pub const CL_FP_FMA: cl_device_fp_config = (1 << 5);
pub const CL_FP_SOFT_FLOAT: cl_device_fp_config = (1 << 6);
pub const CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT: cl_device_fp_config = (1 << 7);

// cl_device_mem_cache_type
pub const CL_NONE: cl_device_mem_cache_type = 0x0;
pub const CL_READ_ONLY_CACHE: cl_device_mem_cache_type = 0x1;
pub const CL_READ_WRITE_CACHE: cl_device_mem_cache_type = 0x2;

// cl_device_local_mem_type
pub const CL_LOCAL: cl_device_local_mem_type = 0x1;
pub const CL_GLOBAL: cl_device_local_mem_type = 0x2;

// cl_device_exec_capabilities - bitfield
pub const CL_EXEC_KERNEL: cl_device_exec_capabilities = (1 << 0);
pub const CL_EXEC_NATIVE_KERNEL: cl_device_exec_capabilities = (1 << 1);

// cl_command_queue_properties - bitfield
pub const CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE: cl_command_queue_properties = (1 << 0);
pub const CL_QUEUE_PROFILING_ENABLE: cl_command_queue_properties = (1 << 1);
pub const CL_QUEUE_ON_DEVICE: cl_command_queue_properties = (1 << 2);
pub const CL_QUEUE_ON_DEVICE_DEFAULT: cl_command_queue_properties = (1 << 3);

// cl_context_info
pub const CL_CONTEXT_REFERENCE_COUNT: cl_context_info = 0x1080;
pub const CL_CONTEXT_DEVICES: cl_context_info = 0x1081;
pub const CL_CONTEXT_PROPERTIES: cl_context_info = 0x1082;
pub const CL_CONTEXT_NUM_DEVICES: cl_context_info = 0x1083;

// cl_context_properties
pub const CL_CONTEXT_PLATFORM: cl_context_properties = 0x1084;
pub const CL_CONTEXT_INTEROP_USER_SYNC: cl_context_properties = 0x1085;

// cl_device_partition_property
pub const CL_DEVICE_PARTITION_EQUALLY: cl_device_partition_property = 0x1086;
pub const CL_DEVICE_PARTITION_BY_COUNTS: cl_device_partition_property = 0x1087;
pub const CL_DEVICE_PARTITION_BY_COUNTS_LIST_END: cl_device_partition_property = 0x0;
pub const CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN: cl_device_partition_property = 0x1088;

// cl_device_affinity_domain
pub const CL_DEVICE_AFFINITY_DOMAIN_NUMA: cl_device_affinity_domain = (1 << 0);
pub const CL_DEVICE_AFFINITY_DOMAIN_L4_CACHE: cl_device_affinity_domain = (1 << 1);
pub const CL_DEVICE_AFFINITY_DOMAIN_L3_CACHE: cl_device_affinity_domain = (1 << 2);
pub const CL_DEVICE_AFFINITY_DOMAIN_L2_CACHE: cl_device_affinity_domain = (1 << 3);
pub const CL_DEVICE_AFFINITY_DOMAIN_L1_CACHE: cl_device_affinity_domain = (1 << 4);
pub const CL_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE: cl_device_affinity_domain = (1 << 5);

// cl_device_svm_capabilities
pub const CL_DEVICE_SVM_COARSE_GRAIN_BUFFER: cl_device_svm_capabilities = (1 << 0);
pub const CL_DEVICE_SVM_FINE_GRAIN_BUFFER: cl_device_svm_capabilities = (1 << 1);
pub const CL_DEVICE_SVM_FINE_GRAIN_SYSTEM: cl_device_svm_capabilities = (1 << 2);
pub const CL_DEVICE_SVM_ATOMICS: cl_device_svm_capabilities = (1 << 3);

// cl_command_queue_info
pub const CL_QUEUE_CONTEXT: cl_command_queue_info = 0x1090;
pub const CL_QUEUE_DEVICE: cl_command_queue_info = 0x1091;
pub const CL_QUEUE_REFERENCE_COUNT: cl_command_queue_info = 0x1092;
pub const CL_QUEUE_PROPERTIES: cl_command_queue_info = 0x1093;
pub const CL_QUEUE_SIZE: cl_command_queue_info = 0x1094;

// cl_mem_flags and cl_svm_mem_flags - bitfield
pub const CL_MEM_READ_WRITE: cl_mem_flags = (1 << 0);
pub const CL_MEM_WRITE_ONLY: cl_mem_flags = (1 << 1);
pub const CL_MEM_READ_ONLY: cl_mem_flags = (1 << 2);
pub const CL_MEM_USE_HOST_PTR: cl_mem_flags = (1 << 3);
pub const CL_MEM_ALLOC_HOST_PTR: cl_mem_flags = (1 << 4);
pub const CL_MEM_COPY_HOST_PTR: cl_mem_flags = (1 << 5);
// reserved                                         (1 << 6)
pub const CL_MEM_HOST_WRITE_ONLY: cl_mem_flags = (1 << 7);
pub const CL_MEM_HOST_READ_ONLY: cl_mem_flags = (1 << 8);
pub const CL_MEM_HOST_NO_ACCESS: cl_mem_flags = (1 << 9);
pub const CL_MEM_SVM_FINE_GRAIN_BUFFER: cl_svm_mem_flags = (1 << 10);   /* used by cl_svm_mem_flags only */
pub const CL_MEM_SVM_ATOMICS: cl_svm_mem_flags = (1 << 11);  /* used by cl_svm_mem_flags only */
pub const CL_MEM_KERNEL_READ_AND_WRITE: cl_mem_flags = (1 << 12);

// cl_mem_migration_flags - bitfield
pub const CL_MIGRATE_MEM_OBJECT_HOST: cl_mem_migration_flags = (1 << 0);
pub const CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED: cl_mem_migration_flags = (1 << 1);

// cl_channel_order
pub const CL_R: cl_channel_order = 0x10B0;
pub const CL_A: cl_channel_order = 0x10B1;
pub const CL_RG: cl_channel_order = 0x10B2;
pub const CL_RA: cl_channel_order = 0x10B3;
pub const CL_RGB: cl_channel_order = 0x10B4;
pub const CL_RGBA: cl_channel_order = 0x10B5;
pub const CL_BGRA: cl_channel_order = 0x10B6;
pub const CL_ARGB: cl_channel_order = 0x10B7;
pub const CL_INTENSITY: cl_channel_order = 0x10B8;
pub const CL_LUMINANCE: cl_channel_order = 0x10B9;
pub const CL_Rx: cl_channel_order = 0x10BA;
pub const CL_RGx: cl_channel_order = 0x10BB;
pub const CL_RGBx: cl_channel_order = 0x10BC;
pub const CL_DEPTH: cl_channel_order = 0x10BD;
pub const CL_DEPTH_STENCIL: cl_channel_order = 0x10BE;
pub const CL_sRGB: cl_channel_order = 0x10BF;
pub const CL_sRGBx: cl_channel_order = 0x10C0;
pub const CL_sRGBA: cl_channel_order = 0x10C1;
pub const CL_sBGRA: cl_channel_order = 0x10C2;
pub const CL_ABGR: cl_channel_order = 0x10C3;


// cl_channel_type
pub const CL_SNORM_INT8: cl_channel_type = 0x10D0;
pub const CL_SNORM_INT16: cl_channel_type = 0x10D1;
pub const CL_UNORM_INT8: cl_channel_type = 0x10D2;
pub const CL_UNORM_INT16: cl_channel_type = 0x10D3;
pub const CL_UNORM_SHORT_565: cl_channel_type = 0x10D4;
pub const CL_UNORM_SHORT_555: cl_channel_type = 0x10D5;
pub const CL_UNORM_INT_101010: cl_channel_type = 0x10D6;
pub const CL_SIGNED_INT8: cl_channel_type = 0x10D7;
pub const CL_SIGNED_INT16: cl_channel_type = 0x10D8;
pub const CL_SIGNED_INT32: cl_channel_type = 0x10D9;
pub const CL_UNSIGNED_INT8: cl_channel_type = 0x10DA;
pub const CL_UNSIGNED_INT16: cl_channel_type = 0x10DB;
pub const CL_UNSIGNED_INT32: cl_channel_type = 0x10DC;
pub const CL_HALF_FLOAT: cl_channel_type = 0x10DD;
pub const CL_FLOAT: cl_channel_type = 0x10DE;
pub const CL_UNORM_INT24: cl_channel_type = 0x10DF;

// cl_mem_object_type
pub const CL_MEM_OBJECT_BUFFER: cl_mem_object_type = 0x10F0;
pub const CL_MEM_OBJECT_IMAGE2D: cl_mem_object_type = 0x10F1;
pub const CL_MEM_OBJECT_IMAGE3D: cl_mem_object_type = 0x10F2;
pub const CL_MEM_OBJECT_IMAGE2D_ARRAY: cl_mem_object_type = 0x10F3;
pub const CL_MEM_OBJECT_IMAGE1D: cl_mem_object_type = 0x10F4;
pub const CL_MEM_OBJECT_IMAGE1D_ARRAY: cl_mem_object_type = 0x10F5;
pub const CL_MEM_OBJECT_IMAGE1D_BUFFER: cl_mem_object_type = 0x10F6;
pub const CL_MEM_OBJECT_PIPE: cl_mem_object_type = 0x10F7;

// cl_mem_info
pub const CL_MEM_TYPE: cl_mem_info = 0x1100;
pub const CL_MEM_FLAGS: cl_mem_info = 0x1101;
pub const CL_MEM_SIZE: cl_mem_info = 0x1102;
pub const CL_MEM_HOST_PTR: cl_mem_info = 0x1103;
pub const CL_MEM_MAP_COUNT: cl_mem_info = 0x1104;
pub const CL_MEM_REFERENCE_COUNT: cl_mem_info = 0x1105;
pub const CL_MEM_CONTEXT: cl_mem_info = 0x1106;
pub const CL_MEM_ASSOCIATED_MEMOBJECT: cl_mem_info = 0x1107;
pub const CL_MEM_OFFSET: cl_mem_info = 0x1108;
pub const CL_MEM_USES_SVM_POINTER: cl_mem_info = 0x1109;

// cl_image_info
pub const CL_IMAGE_FORMAT: cl_image_info = 0x1110;
pub const CL_IMAGE_ELEMENT_SIZE: cl_image_info = 0x1111;
pub const CL_IMAGE_ROW_PITCH: cl_image_info = 0x1112;
pub const CL_IMAGE_SLICE_PITCH: cl_image_info = 0x1113;
pub const CL_IMAGE_WIDTH: cl_image_info = 0x1114;
pub const CL_IMAGE_HEIGHT: cl_image_info = 0x1115;
pub const CL_IMAGE_DEPTH: cl_image_info = 0x1116;
pub const CL_IMAGE_ARRAY_SIZE: cl_image_info = 0x1117;
pub const CL_IMAGE_BUFFER: cl_image_info = 0x1118;
pub const CL_IMAGE_NUM_MIP_LEVELS: cl_image_info = 0x1119;
pub const CL_IMAGE_NUM_SAMPLES: cl_image_info = 0x111A;

// cl_pipe_info
pub const CL_PIPE_PACKET_SIZE: cl_pipe_info = 0x1120;
pub const CL_PIPE_MAX_PACKETS: cl_pipe_info = 0x1121;

// cl_addressing_mode
pub const CL_ADDRESS_NONE: cl_addressing_mode = 0x1130;
pub const CL_ADDRESS_CLAMP_TO_EDGE: cl_addressing_mode = 0x1131;
pub const CL_ADDRESS_CLAMP: cl_addressing_mode = 0x1132;
pub const CL_ADDRESS_REPEAT: cl_addressing_mode = 0x1133;
pub const CL_ADDRESS_MIRRORED_REPEAT: cl_addressing_mode = 0x1134;

// cl_filter_mode
pub const CL_FILTER_NEAREST: cl_filter_mode = 0x1140;
pub const CL_FILTER_LINEAR: cl_filter_mode = 0x1141;

// cl_sampler_info
pub const CL_SAMPLER_REFERENCE_COUNT: cl_sampler_info = 0x1150;
pub const CL_SAMPLER_CONTEXT: cl_sampler_info = 0x1151;
pub const CL_SAMPLER_NORMALIZED_COORDS: cl_sampler_info = 0x1152;
pub const CL_SAMPLER_ADDRESSING_MODE: cl_sampler_info = 0x1153;
pub const CL_SAMPLER_FILTER_MODE: cl_sampler_info = 0x1154;
pub const CL_SAMPLER_MIP_FILTER_MODE: cl_sampler_info = 0x1155;
pub const CL_SAMPLER_LOD_MIN: cl_sampler_info = 0x1156;
pub const CL_SAMPLER_LOD_MAX: cl_sampler_info = 0x1157;

// cl_map_flags - bitfield
pub const CL_MAP_READ: cl_map_flags = (1 << 0);
pub const CL_MAP_WRITE: cl_map_flags = (1 << 1);
pub const CL_MAP_WRITE_INVALIDATE_REGION: cl_map_flags = (1 << 2);

// cl_program_info
pub const CL_PROGRAM_REFERENCE_COUNT: cl_program_info = 0x1160;
pub const CL_PROGRAM_CONTEXT: cl_program_info = 0x1161;
pub const CL_PROGRAM_NUM_DEVICES: cl_program_info = 0x1162;
pub const CL_PROGRAM_DEVICES: cl_program_info = 0x1163;
pub const CL_PROGRAM_SOURCE: cl_program_info = 0x1164;
pub const CL_PROGRAM_BINARY_SIZES: cl_program_info = 0x1165;
pub const CL_PROGRAM_BINARIES: cl_program_info = 0x1166;
pub const CL_PROGRAM_NUM_KERNELS: cl_program_info = 0x1167;
pub const CL_PROGRAM_KERNEL_NAMES: cl_program_info = 0x1168;

// cl_program_build_info
pub const CL_PROGRAM_BUILD_STATUS: cl_program_build_info = 0x1181;
pub const CL_PROGRAM_BUILD_OPTIONS: cl_program_build_info = 0x1182;
pub const CL_PROGRAM_BUILD_LOG: cl_program_build_info = 0x1183;
pub const CL_PROGRAM_BINARY_TYPE: cl_program_build_info = 0x1184;
pub const CL_PROGRAM_BUILD_GLOBAL_VARIABLE_TOTAL_SIZE: cl_program_build_info = 0x1185;

// cl_program_binary_type
pub const CL_PROGRAM_BINARY_TYPE_NONE: cl_program_binary_type = 0x0;
pub const CL_PROGRAM_BINARY_TYPE_COMPILED_OBJECT: cl_program_binary_type = 0x1;
pub const CL_PROGRAM_BINARY_TYPE_LIBRARY: cl_program_binary_type = 0x2;
pub const CL_PROGRAM_BINARY_TYPE_EXECUTABLE: cl_program_binary_type = 0x4;

// cl_build_status
pub const CL_BUILD_SUCCESS: cl_build_status = 0;
pub const CL_BUILD_NONE: cl_build_status = -1;
pub const CL_BUILD_ERROR: cl_build_status = -2;
pub const CL_BUILD_IN_PROGRESS: cl_build_status = -3;

// cl_kernel_info
pub const CL_KERNEL_FUNCTION_NAME: cl_kernel_info = 0x1190;
pub const CL_KERNEL_NUM_ARGS: cl_kernel_info = 0x1191;
pub const CL_KERNEL_REFERENCE_COUNT: cl_kernel_info = 0x1192;
pub const CL_KERNEL_CONTEXT: cl_kernel_info = 0x1193;
pub const CL_KERNEL_PROGRAM: cl_kernel_info = 0x1194;
pub const CL_KERNEL_ATTRIBUTES: cl_kernel_info = 0x1195;

// cl_kernel_arg_info
pub const CL_KERNEL_ARG_ADDRESS_QUALIFIER: cl_kernel_arg_info = 0x1196;
pub const CL_KERNEL_ARG_ACCESS_QUALIFIER: cl_kernel_arg_info = 0x1197;
pub const CL_KERNEL_ARG_TYPE_NAME: cl_kernel_arg_info = 0x1198;
pub const CL_KERNEL_ARG_TYPE_QUALIFIER: cl_kernel_arg_info = 0x1199;
pub const CL_KERNEL_ARG_NAME: cl_kernel_arg_info = 0x119A;

// cl_kernel_arg_address_qualifier
pub const CL_KERNEL_ARG_ADDRESS_GLOBAL: cl_kernel_arg_address_qualifier = 0x119B;
pub const CL_KERNEL_ARG_ADDRESS_LOCAL: cl_kernel_arg_address_qualifier = 0x119C;
pub const CL_KERNEL_ARG_ADDRESS_CONSTANT: cl_kernel_arg_address_qualifier = 0x119D;
pub const CL_KERNEL_ARG_ADDRESS_PRIVATE: cl_kernel_arg_address_qualifier = 0x119E;

// cl_kernel_arg_access_qualifier
pub const CL_KERNEL_ARG_ACCESS_READ_ONLY: cl_kernel_arg_access_qualifier = 0x11A0;
pub const CL_KERNEL_ARG_ACCESS_WRITE_ONLY: cl_kernel_arg_access_qualifier = 0x11A1;
pub const CL_KERNEL_ARG_ACCESS_READ_WRITE: cl_kernel_arg_access_qualifier = 0x11A2;
pub const CL_KERNEL_ARG_ACCESS_NONE: cl_kernel_arg_access_qualifier = 0x11A3;

// cl_kernel_arg_type_qualifier
pub const CL_KERNEL_ARG_TYPE_NONE: cl_kernel_arg_type_qualifier = 0;
pub const CL_KERNEL_ARG_TYPE_CONST: cl_kernel_arg_type_qualifier = (1 << 0);
pub const CL_KERNEL_ARG_TYPE_RESTRICT: cl_kernel_arg_type_qualifier = (1 << 1);
pub const CL_KERNEL_ARG_TYPE_VOLATILE: cl_kernel_arg_type_qualifier = (1 << 2);
pub const CL_KERNEL_ARG_TYPE_PIPE: cl_kernel_arg_type_qualifier = (1 << 3);

// cl_kernel_work_group_info
pub const CL_KERNEL_WORK_GROUP_SIZE: cl_kernel_work_group_info = 0x11B0;
pub const CL_KERNEL_COMPILE_WORK_GROUP_SIZE: cl_kernel_work_group_info = 0x11B1;
pub const CL_KERNEL_LOCAL_MEM_SIZE: cl_kernel_work_group_info = 0x11B2;
pub const CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE: cl_kernel_work_group_info = 0x11B3;
pub const CL_KERNEL_PRIVATE_MEM_SIZE: cl_kernel_work_group_info = 0x11B4;
pub const CL_KERNEL_GLOBAL_WORK_SIZE: cl_kernel_work_group_info = 0x11B5;

// cl_kernel_exec_info
pub const CL_KERNEL_EXEC_INFO_SVM_PTRS: cl_kernel_exec_info = 0x11B6;
pub const CL_KERNEL_EXEC_INFO_SVM_FINE_GRAIN_SYSTEM: cl_kernel_exec_info = 0x11B7;

// cl_event_info
pub const CL_EVENT_COMMAND_QUEUE: cl_event_info = 0x11D0;
pub const CL_EVENT_COMMAND_TYPE: cl_event_info = 0x11D1;
pub const CL_EVENT_REFERENCE_COUNT: cl_event_info = 0x11D2;
pub const CL_EVENT_COMMAND_EXECUTION_STATUS: cl_event_info = 0x11D3;
pub const CL_EVENT_CONTEXT: cl_event_info = 0x11D4;

// cl_command_type
pub const CL_COMMAND_NDRANGE_KERNEL: cl_command_type = 0x11F0;
pub const CL_COMMAND_TASK: cl_command_type = 0x11F1;
pub const CL_COMMAND_NATIVE_KERNEL: cl_command_type = 0x11F2;
pub const CL_COMMAND_READ_BUFFER: cl_command_type = 0x11F3;
pub const CL_COMMAND_WRITE_BUFFER: cl_command_type = 0x11F4;
pub const CL_COMMAND_COPY_BUFFER: cl_command_type = 0x11F5;
pub const CL_COMMAND_READ_IMAGE: cl_command_type = 0x11F6;
pub const CL_COMMAND_WRITE_IMAGE: cl_command_type = 0x11F7;
pub const CL_COMMAND_COPY_IMAGE: cl_command_type = 0x11F8;
pub const CL_COMMAND_COPY_IMAGE_TO_BUFFER: cl_command_type = 0x11F9;
pub const CL_COMMAND_COPY_BUFFER_TO_IMAGE: cl_command_type = 0x11FA;
pub const CL_COMMAND_MAP_BUFFER: cl_command_type = 0x11FB;
pub const CL_COMMAND_MAP_IMAGE: cl_command_type = 0x11FC;
pub const CL_COMMAND_UNMAP_MEM_OBJECT: cl_command_type = 0x11FD;
pub const CL_COMMAND_MARKER: cl_command_type = 0x11FE;
pub const CL_COMMAND_ACQUIRE_GL_OBJECTS: cl_command_type = 0x11FF;
pub const CL_COMMAND_RELEASE_GL_OBJECTS: cl_command_type = 0x1200;
pub const CL_COMMAND_READ_BUFFER_RECT: cl_command_type = 0x1201;
pub const CL_COMMAND_WRITE_BUFFER_RECT: cl_command_type = 0x1202;
pub const CL_COMMAND_COPY_BUFFER_RECT: cl_command_type = 0x1203;
pub const CL_COMMAND_USER: cl_command_type = 0x1204;
pub const CL_COMMAND_BARRIER: cl_command_type = 0x1205;
pub const CL_COMMAND_MIGRATE_MEM_OBJECTS: cl_command_type = 0x1206;
pub const CL_COMMAND_FILL_BUFFER: cl_command_type = 0x1207;
pub const CL_COMMAND_FILL_IMAGE: cl_command_type = 0x1208;
pub const CL_COMMAND_SVM_FREE: cl_command_type = 0x1209;
pub const CL_COMMAND_SVM_MEMCPY: cl_command_type = 0x120A;
pub const CL_COMMAND_SVM_MEMFILL: cl_command_type = 0x120B;
pub const CL_COMMAND_SVM_MAP: cl_command_type = 0x120C;
pub const CL_COMMAND_SVM_UNMAP: cl_command_type = 0x120D;

// command execution status
pub const CL_COMPLETE: cl_uint = 0x0;
pub const CL_RUNNING: cl_uint = 0x1;
pub const CL_SUBMITTED: cl_uint = 0x2;
pub const CL_QUEUED: cl_uint = 0x3;

// cl_buffer_create_type
pub const CL_BUFFER_CREATE_TYPE_REGION: cl_buffer_create_type = 0x1220;

// cl_profiling_info
pub const CL_PROFILING_COMMAND_QUEUED: cl_profiling_info = 0x1280;
pub const CL_PROFILING_COMMAND_SUBMIT: cl_profiling_info = 0x1281;
pub const CL_PROFILING_COMMAND_START: cl_profiling_info = 0x1282;
pub const CL_PROFILING_COMMAND_END: cl_profiling_info = 0x1283;
pub const CL_PROFILING_COMMAND_COMPLETE: cl_profiling_info = 0x1284;

extern "C" {
    pub fn clGetPlatformIDs(arg1: cl_uint,
                            arg2: *mut cl_platform_id,
                            arg3: *mut cl_uint)
                            -> cl_int;
    pub fn clGetPlatformInfo(arg1: cl_platform_id,
                             arg2: cl_platform_info,
                             arg3: size_t,
                             arg4: *mut c_void,
                             arg5: *mut size_t)
                             -> cl_int;
    pub fn clGetDeviceIDs(arg1: cl_platform_id,
                          arg2: cl_device_type,
                          arg3: cl_uint,
                          arg4: *mut cl_device_id,
                          arg5: *mut cl_uint)
                          -> cl_int;
    pub fn clGetDeviceInfo(arg1: cl_device_id,
                           arg2: cl_device_info,
                           arg3: size_t,
                           arg4: *mut c_void,
                           arg5: *mut size_t)
                           -> cl_int;
    pub fn clCreateSubDevices(arg1: cl_device_id,
                              arg2: *const cl_device_partition_property,
                              arg3: cl_uint,
                              arg4: *mut cl_device_id,
                              arg5: *mut cl_uint)
                              -> cl_int;
    pub fn clRetainDevice(arg1: cl_device_id) -> cl_int;
    pub fn clReleaseDevice(arg1: cl_device_id) -> cl_int;
    pub fn clCreateContext(arg1: *const cl_context_properties,
                           arg2: cl_uint,
                           arg3: *const cl_device_id,
                           arg4: fn_notify,
                           arg5: *mut c_void,
                           arg6: *mut cl_int)
                           -> cl_context;
    pub fn clCreateContextFromType(arg1: *const cl_context_properties,
                                   arg2: cl_device_type,
                                   arg3: fn_notify,
                                   arg4: *mut c_void,
                                   arg5: *mut cl_int)
                                   -> cl_context;
    pub fn clRetainContext(arg1: cl_context) -> cl_int;
    pub fn clReleaseContext(arg1: cl_context) -> cl_int;
    pub fn clGetContextInfo(arg1: cl_context,
                            arg2: cl_context_info,
                            arg3: size_t,
                            arg4: *mut c_void,
                            arg5: *mut size_t)
                            -> cl_int;
    pub fn clCreateCommandQueueWithProperties(arg1: cl_context,
                                              arg2: cl_device_id,
                                              arg3: *const cl_queue_properties,
                                              arg4: *mut cl_int)
                                              -> cl_command_queue;
    pub fn clRetainCommandQueue(arg1: cl_command_queue) -> cl_int;
    pub fn clReleaseCommandQueue(arg1: cl_command_queue) -> cl_int;
    pub fn clGetCommandQueueInfo(arg1: cl_command_queue,
                                 arg2: cl_command_queue_info,
                                 arg3: size_t,
                                 arg4: *mut c_void,
                                 arg5: *mut size_t)
                                 -> cl_int;
    pub fn clCreateBuffer(arg1: cl_context,
                          arg2: cl_mem_flags,
                          arg3: size_t,
                          arg4: *mut c_void,
                          arg5: *mut cl_int)
                          -> cl_mem;
    pub fn clCreateSubBuffer(arg1: cl_mem,
                             arg2: cl_mem_flags,
                             arg3: cl_buffer_create_type,
                             arg4: *const c_void,
                             arg5: *mut cl_int)
                             -> cl_mem;
    pub fn clCreateImage(arg1: cl_context,
                         arg2: cl_mem_flags,
                         arg3: *const cl_image_format,
                         arg4: *const cl_image_desc,
                         arg5: *mut c_void,
                         arg6: *mut cl_int)
                         -> cl_mem;
    pub fn clCreatePipe(arg1: cl_context,
                        arg2: cl_mem_flags,
                        arg3: cl_uint,
                        arg4: cl_uint,
                        arg5: *const cl_pipe_properties,
                        arg6: *mut cl_int)
                        -> cl_mem;
    pub fn clRetainMemObject(arg1: cl_mem) -> cl_int;
    pub fn clReleaseMemObject(arg1: cl_mem) -> cl_int;
    pub fn clGetSupportedImageFormats(arg1: cl_context,
                                      arg2: cl_mem_flags,
                                      arg3: cl_mem_object_type,
                                      arg4: cl_uint,
                                      arg5: *mut cl_image_format,
                                      arg6: *mut cl_uint)
                                      -> cl_int;
    pub fn clGetMemObjectInfo(arg1: cl_mem,
                              arg2: cl_mem_info,
                              arg3: size_t,
                              arg4: *mut c_void,
                              arg5: *mut size_t)
                              -> cl_int;
    pub fn clGetImageInfo(arg1: cl_mem,
                          arg2: cl_image_info,
                          arg3: size_t,
                          arg4: *mut c_void,
                          arg5: *mut size_t)
                          -> cl_int;
    pub fn clGetPipeInfo(arg1: cl_mem,
                         arg2: cl_pipe_info,
                         arg3: size_t,
                         arg4: *mut c_void,
                         arg5: *mut size_t)
                         -> cl_int;
    pub fn clSetMemObjectDestructorCallback(arg1: cl_mem,
                                            arg2:
                                                ::std::option::Option<unsafe extern "C" fn(arg1:
                                                cl_mem,
                                            arg2:
                                                *mut c_void)>,
                                            arg3: *mut c_void)
     -> cl_int;
    pub fn clSVMAlloc(arg1: cl_context,
                      arg2: cl_svm_mem_flags,
                      arg3: size_t,
                      arg4: cl_uint)
                      -> *mut c_void;
    pub fn clSVMFree(arg1: cl_context, arg2: *mut c_void);
    pub fn clCreateSamplerWithProperties(arg1: cl_context,
                                         arg2: *const cl_sampler_properties,
                                         arg3: *mut cl_int)
                                         -> cl_sampler;
    pub fn clRetainSampler(arg1: cl_sampler) -> cl_int;
    pub fn clReleaseSampler(arg1: cl_sampler) -> cl_int;
    pub fn clGetSamplerInfo(arg1: cl_sampler,
                            arg2: cl_sampler_info,
                            arg3: size_t,
                            arg4: *mut c_void,
                            arg5: *mut size_t)
                            -> cl_int;
    pub fn clCreateProgramWithSource(arg1: cl_context,
                                     arg2: cl_uint,
                                     arg3: *mut *const ::std::os::raw::c_char,
                                     arg4: *const size_t,
                                     arg5: *mut cl_int)
                                     -> cl_program;
    pub fn clCreateProgramWithBinary(arg1: cl_context,
                                     arg2: cl_uint,
                                     arg3: *const cl_device_id,
                                     arg4: *const size_t,
                                     arg5: *mut *const ::std::os::raw::c_uchar,
                                     arg6: *mut cl_int,
                                     arg7: *mut cl_int)
                                     -> cl_program;
    pub fn clCreateProgramWithBuiltInKernels(arg1: cl_context,
                                             arg2: cl_uint,
                                             arg3: *const cl_device_id,
                                             arg4: *const ::std::os::raw::c_char,
                                             arg5: *mut cl_int)
                                             -> cl_program;
    pub fn clRetainProgram(arg1: cl_program) -> cl_int;
    pub fn clReleaseProgram(arg1: cl_program) -> cl_int;
    pub fn clBuildProgram(arg1: cl_program,
                          arg2: cl_uint,
                          arg3: *const cl_device_id,
                          arg4: *const ::std::os::raw::c_char,
                          arg5: ::std::option::Option<unsafe extern "C" fn(arg1: cl_program,
                                                                             arg2: *mut c_void)
                                                                            >,
                          arg6: *mut c_void)
                          -> cl_int;
    pub fn clCompileProgram(arg1: cl_program,
                            arg2: cl_uint,
                            arg3: *const cl_device_id,
                            arg4: *const ::std::os::raw::c_char,
                            arg5: cl_uint,
                            arg6: *const cl_program,
                            arg7: *mut *const ::std::os::raw::c_char,
                            arg8: ::std::option::Option<unsafe extern "C" fn(arg1: cl_program,
                                                                               arg2: *mut c_void)
                                                                              >,
                            arg9: *mut c_void)
                            -> cl_int;
    pub fn clLinkProgram(arg1: cl_context,
                         arg2: cl_uint,
                         arg3: *const cl_device_id,
                         arg4: *const ::std::os::raw::c_char,
                         arg5: cl_uint,
                         arg6: *const cl_program,
                         arg7: ::std::option::Option<unsafe extern "C" fn(arg1: cl_program,
                                                                            arg2: *mut c_void)
                                                                           >,
                         arg8: *mut c_void,
                         arg9: *mut cl_int)
                         -> cl_program;
    pub fn clUnloadPlatformCompiler(arg1: cl_platform_id) -> cl_int;
    pub fn clGetProgramInfo(arg1: cl_program,
                            arg2: cl_program_info,
                            arg3: size_t,
                            arg4: *mut c_void,
                            arg5: *mut size_t)
                            -> cl_int;
    pub fn clGetProgramBuildInfo(arg1: cl_program,
                                 arg2: cl_device_id,
                                 arg3: cl_program_build_info,
                                 arg4: size_t,
                                 arg5: *mut c_void,
                                 arg6: *mut size_t)
                                 -> cl_int;
    pub fn clCreateKernel(arg1: cl_program,
                          arg2: *const ::std::os::raw::c_char,
                          arg3: *mut cl_int)
                          -> cl_kernel;
    pub fn clCreateKernelsInProgram(arg1: cl_program,
                                    arg2: cl_uint,
                                    arg3: *mut cl_kernel,
                                    arg4: *mut cl_uint)
                                    -> cl_int;
    pub fn clRetainKernel(arg1: cl_kernel) -> cl_int;
    pub fn clReleaseKernel(arg1: cl_kernel) -> cl_int;
    pub fn clSetKernelArg(arg1: cl_kernel,
                          arg2: cl_uint,
                          arg3: size_t,
                          arg4: *const c_void)
                          -> cl_int;
    pub fn clSetKernelArgSVMPointer(arg1: cl_kernel, arg2: cl_uint, arg3: *const c_void) -> cl_int;
    pub fn clSetKernelExecInfo(arg1: cl_kernel,
                               arg2: cl_kernel_exec_info,
                               arg3: size_t,
                               arg4: *const c_void)
                               -> cl_int;
    pub fn clGetKernelInfo(arg1: cl_kernel,
                           arg2: cl_kernel_info,
                           arg3: size_t,
                           arg4: *mut c_void,
                           arg5: *mut size_t)
                           -> cl_int;
    pub fn clGetKernelArgInfo(arg1: cl_kernel,
                              arg2: cl_uint,
                              arg3: cl_kernel_arg_info,
                              arg4: size_t,
                              arg5: *mut c_void,
                              arg6: *mut size_t)
                              -> cl_int;
    pub fn clGetKernelWorkGroupInfo(arg1: cl_kernel,
                                    arg2: cl_device_id,
                                    arg3: cl_kernel_work_group_info,
                                    arg4: size_t,
                                    arg5: *mut c_void,
                                    arg6: *mut size_t)
                                    -> cl_int;
    pub fn clWaitForEvents(arg1: cl_uint, arg2: *const cl_event) -> cl_int;
    pub fn clGetEventInfo(arg1: cl_event,
                          arg2: cl_event_info,
                          arg3: size_t,
                          arg4: *mut c_void,
                          arg5: *mut size_t)
                          -> cl_int;
    pub fn clCreateUserEvent(arg1: cl_context, arg2: *mut cl_int) -> cl_event;
    pub fn clRetainEvent(arg1: cl_event) -> cl_int;
    pub fn clReleaseEvent(arg1: cl_event) -> cl_int;
    pub fn clSetUserEventStatus(arg1: cl_event, arg2: cl_int) -> cl_int;
    pub fn clSetEventCallback(arg1: cl_event, arg2: cl_int,
                              arg3:
                                  ::std::option::Option<unsafe extern "C" fn(arg1:
                        cl_event,
                    arg2:
                        cl_int,
                    arg3:
                        *mut c_void)>,
                              arg4: *mut c_void) -> cl_int;
    pub fn clGetEventProfilingInfo(arg1: cl_event,
                                   arg2: cl_profiling_info,
                                   arg3: size_t,
                                   arg4: *mut c_void,
                                   arg5: *mut size_t)
                                   -> cl_int;
    pub fn clFlush(arg1: cl_command_queue) -> cl_int;
    pub fn clFinish(arg1: cl_command_queue) -> cl_int;
    pub fn clEnqueueReadBuffer(arg1: cl_command_queue,
                               arg2: cl_mem,
                               arg3: cl_bool,
                               arg4: size_t,
                               arg5: size_t,
                               arg6: *mut c_void,
                               arg7: cl_uint,
                               arg8: *const cl_event,
                               arg9: *mut cl_event)
                               -> cl_int;
    pub fn clEnqueueReadBufferRect(arg1: cl_command_queue,
                                   arg2: cl_mem,
                                   arg3: cl_bool,
                                   arg4: *const size_t,
                                   arg5: *const size_t,
                                   arg6: *const size_t,
                                   arg7: size_t,
                                   arg8: size_t,
                                   arg9: size_t,
                                   arg10: size_t,
                                   arg11: *mut c_void,
                                   arg12: cl_uint,
                                   arg13: *const cl_event,
                                   arg14: *mut cl_event)
                                   -> cl_int;
    pub fn clEnqueueWriteBuffer(arg1: cl_command_queue,
                                arg2: cl_mem,
                                arg3: cl_bool,
                                arg4: size_t,
                                arg5: size_t,
                                arg6: *const c_void,
                                arg7: cl_uint,
                                arg8: *const cl_event,
                                arg9: *mut cl_event)
                                -> cl_int;
    pub fn clEnqueueWriteBufferRect(arg1: cl_command_queue,
                                    arg2: cl_mem,
                                    arg3: cl_bool,
                                    arg4: *const size_t,
                                    arg5: *const size_t,
                                    arg6: *const size_t,
                                    arg7: size_t,
                                    arg8: size_t,
                                    arg9: size_t,
                                    arg10: size_t,
                                    arg11: *const c_void,
                                    arg12: cl_uint,
                                    arg13: *const cl_event,
                                    arg14: *mut cl_event)
                                    -> cl_int;
    pub fn clEnqueueFillBuffer(arg1: cl_command_queue,
                               arg2: cl_mem,
                               arg3: *const c_void,
                               arg4: size_t,
                               arg5: size_t,
                               arg6: size_t,
                               arg7: cl_uint,
                               arg8: *const cl_event,
                               arg9: *mut cl_event)
                               -> cl_int;
    pub fn clEnqueueCopyBuffer(arg1: cl_command_queue,
                               arg2: cl_mem,
                               arg3: cl_mem,
                               arg4: size_t,
                               arg5: size_t,
                               arg6: size_t,
                               arg7: cl_uint,
                               arg8: *const cl_event,
                               arg9: *mut cl_event)
                               -> cl_int;
    pub fn clEnqueueCopyBufferRect(arg1: cl_command_queue,
                                   arg2: cl_mem,
                                   arg3: cl_mem,
                                   arg4: *const size_t,
                                   arg5: *const size_t,
                                   arg6: *const size_t,
                                   arg7: size_t,
                                   arg8: size_t,
                                   arg9: size_t,
                                   arg10: size_t,
                                   arg11: cl_uint,
                                   arg12: *const cl_event,
                                   arg13: *mut cl_event)
                                   -> cl_int;
    pub fn clEnqueueReadImage(arg1: cl_command_queue,
                              arg2: cl_mem,
                              arg3: cl_bool,
                              arg4: *const size_t,
                              arg5: *const size_t,
                              arg6: size_t,
                              arg7: size_t,
                              arg8: *mut c_void,
                              arg9: cl_uint,
                              arg10: *const cl_event,
                              arg11: *mut cl_event)
                              -> cl_int;
    pub fn clEnqueueWriteImage(arg1: cl_command_queue,
                               arg2: cl_mem,
                               arg3: cl_bool,
                               arg4: *const size_t,
                               arg5: *const size_t,
                               arg6: size_t,
                               arg7: size_t,
                               arg8: *const c_void,
                               arg9: cl_uint,
                               arg10: *const cl_event,
                               arg11: *mut cl_event)
                               -> cl_int;
    pub fn clEnqueueFillImage(arg1: cl_command_queue,
                              arg2: cl_mem,
                              arg3: *const c_void,
                              arg4: *const size_t,
                              arg5: *const size_t,
                              arg6: cl_uint,
                              arg7: *const cl_event,
                              arg8: *mut cl_event)
                              -> cl_int;
    pub fn clEnqueueCopyImage(arg1: cl_command_queue,
                              arg2: cl_mem,
                              arg3: cl_mem,
                              arg4: *const size_t,
                              arg5: *const size_t,
                              arg6: *const size_t,
                              arg7: cl_uint,
                              arg8: *const cl_event,
                              arg9: *mut cl_event)
                              -> cl_int;
    pub fn clEnqueueCopyImageToBuffer(arg1: cl_command_queue,
                                      arg2: cl_mem,
                                      arg3: cl_mem,
                                      arg4: *const size_t,
                                      arg5: *const size_t,
                                      arg6: size_t,
                                      arg7: cl_uint,
                                      arg8: *const cl_event,
                                      arg9: *mut cl_event)
                                      -> cl_int;
    pub fn clEnqueueCopyBufferToImage(arg1: cl_command_queue,
                                      arg2: cl_mem,
                                      arg3: cl_mem,
                                      arg4: size_t,
                                      arg5: *const size_t,
                                      arg6: *const size_t,
                                      arg7: cl_uint,
                                      arg8: *const cl_event,
                                      arg9: *mut cl_event)
                                      -> cl_int;
    pub fn clEnqueueMapBuffer(arg1: cl_command_queue,
                              arg2: cl_mem,
                              arg3: cl_bool,
                              arg4: cl_map_flags,
                              arg5: size_t,
                              arg6: size_t,
                              arg7: cl_uint,
                              arg8: *const cl_event,
                              arg9: *mut cl_event,
                              arg10: *mut cl_int)
                              -> *mut c_void;
    pub fn clEnqueueMapImage(arg1: cl_command_queue,
                             arg2: cl_mem,
                             arg3: cl_bool,
                             arg4: cl_map_flags,
                             arg5: *const size_t,
                             arg6: *const size_t,
                             arg7: *mut size_t,
                             arg8: *mut size_t,
                             arg9: cl_uint,
                             arg10: *const cl_event,
                             arg11: *mut cl_event,
                             arg12: *mut cl_int)
                             -> *mut c_void;
    pub fn clEnqueueUnmapMemObject(arg1: cl_command_queue,
                                   arg2: cl_mem,
                                   arg3: *mut c_void,
                                   arg4: cl_uint,
                                   arg5: *const cl_event,
                                   arg6: *mut cl_event)
                                   -> cl_int;
    pub fn clEnqueueMigrateMemObjects(arg1: cl_command_queue,
                                      arg2: cl_uint,
                                      arg3: *const cl_mem,
                                      arg4: cl_mem_migration_flags,
                                      arg5: cl_uint,
                                      arg6: *const cl_event,
                                      arg7: *mut cl_event)
                                      -> cl_int;
    pub fn clEnqueueNDRangeKernel(arg1: cl_command_queue,
                                  arg2: cl_kernel,
                                  arg3: cl_uint,
                                  arg4: *const size_t,
                                  arg5: *const size_t,
                                  arg6: *const size_t,
                                  arg7: cl_uint,
                                  arg8: *const cl_event,
                                  arg9: *mut cl_event)
                                  -> cl_int;
    pub fn clEnqueueNativeKernel(arg1: cl_command_queue,
                                 arg2:
                                     ::std::option::Option<unsafe extern "C" fn(arg1:
                                        *mut c_void)>,
                                 arg3: *mut c_void,
                                 arg4: size_t, arg5: cl_uint,
                                 arg6: *const cl_mem,
                                 arg7: *mut *const c_void,
                                 arg8: cl_uint, arg9: *const cl_event,
                                 arg10: *mut cl_event) -> cl_int;
    pub fn clEnqueueMarkerWithWaitList(arg1: cl_command_queue,
                                       arg2: cl_uint,
                                       arg3: *const cl_event,
                                       arg4: *mut cl_event)
                                       -> cl_int;
    pub fn clEnqueueBarrierWithWaitList(arg1: cl_command_queue,
                                        arg2: cl_uint,
                                        arg3: *const cl_event,
                                        arg4: *mut cl_event)
                                        -> cl_int;
    pub fn clEnqueueSVMFree(arg1: cl_command_queue, arg2: cl_uint,
                            arg3: *mut *mut c_void,
                            arg4:
                                ::std::option::Option<unsafe extern "C" fn(arg1:
                              cl_command_queue,
                          arg2:
                              cl_uint,
                          arg3:
                              *mut *mut c_void,
                          arg4:
                              *mut c_void)>,
                            arg5: *mut c_void, arg6: cl_uint,
                            arg7: *const cl_event, arg8: *mut cl_event)
     -> cl_int;
    pub fn clEnqueueSVMMemcpy(arg1: cl_command_queue,
                              arg2: cl_bool,
                              arg3: *mut c_void,
                              arg4: *const c_void,
                              arg5: size_t,
                              arg6: cl_uint,
                              arg7: *const cl_event,
                              arg8: *mut cl_event)
                              -> cl_int;
    pub fn clEnqueueSVMMemFill(arg1: cl_command_queue,
                               arg2: *mut c_void,
                               arg3: *const c_void,
                               arg4: size_t,
                               arg5: size_t,
                               arg6: cl_uint,
                               arg7: *const cl_event,
                               arg8: *mut cl_event)
                               -> cl_int;
    pub fn clEnqueueSVMMap(arg1: cl_command_queue,
                           arg2: cl_bool,
                           arg3: cl_map_flags,
                           arg4: *mut c_void,
                           arg5: size_t,
                           arg6: cl_uint,
                           arg7: *const cl_event,
                           arg8: *mut cl_event)
                           -> cl_int;
    pub fn clEnqueueSVMUnmap(arg1: cl_command_queue,
                             arg2: *mut c_void,
                             arg3: cl_uint,
                             arg4: *const cl_event,
                             arg5: *mut cl_event)
                             -> cl_int;
    pub fn clGetExtensionFunctionAddressForPlatform(arg1: cl_platform_id,
                                                    arg2: *const ::std::os::raw::c_char)
                                                    -> *mut c_void;
    pub fn clCreateImage2D(arg1: cl_context,
                           arg2: cl_mem_flags,
                           arg3: *const cl_image_format,
                           arg4: size_t,
                           arg5: size_t,
                           arg6: size_t,
                           arg7: *mut c_void,
                           arg8: *mut cl_int)
                           -> cl_mem;
    pub fn clCreateImage3D(arg1: cl_context,
                           arg2: cl_mem_flags,
                           arg3: *const cl_image_format,
                           arg4: size_t,
                           arg5: size_t,
                           arg6: size_t,
                           arg7: size_t,
                           arg8: size_t,
                           arg9: *mut c_void,
                           arg10: *mut cl_int)
                           -> cl_mem;
    pub fn clEnqueueMarker(arg1: cl_command_queue, arg2: *mut cl_event) -> cl_int;
    pub fn clEnqueueWaitForEvents(arg1: cl_command_queue,
                                  arg2: cl_uint,
                                  arg3: *const cl_event)
                                  -> cl_int;
    pub fn clEnqueueBarrier(arg1: cl_command_queue) -> cl_int;
    pub fn clUnloadCompiler() -> cl_int;
    pub fn clGetExtensionFunctionAddress(arg1: *const ::std::os::raw::c_char) -> *mut c_void;
    pub fn clCreateCommandQueue(arg1: cl_context,
                                arg2: cl_device_id,
                                arg3: cl_command_queue_properties,
                                arg4: *mut cl_int)
                                -> cl_command_queue;
    pub fn clCreateSampler(arg1: cl_context,
                           arg2: cl_bool,
                           arg3: cl_addressing_mode,
                           arg4: cl_filter_mode,
                           arg5: *mut cl_int)
                           -> cl_sampler;
    pub fn clEnqueueTask(arg1: cl_command_queue,
                         arg2: cl_kernel,
                         arg3: cl_uint,
                         arg4: *const cl_event,
                         arg5: *mut cl_event)
                         -> cl_int;
}
