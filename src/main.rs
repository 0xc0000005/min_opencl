extern crate libc;
use libc::{c_void, size_t};

use std::ptr;
use std::mem;
use std::ffi::CString;

mod opencl;
use opencl::*;

// OpenCL kernel to perform an element-wise addition
const KERNEL_SRC: &'static str =
"__kernel void vecadd(__global int *A, __global int *B, __global int *C) {\n
    // Get the work-item’s unique ID\n
    int idx=get_global_id(0);\n
    // Add the corresponding locations of\n
    // ‘A’ and ‘B’, and store the result in ‘C’.\n
    C[idx] = A[idx] + B[idx];\n
}\n";

fn main() {
    #![allow(non_snake_case)]

    // get opencl platform
    let mut num_of_platforms = 0 as cl_uint;
    let error = unsafe { clGetPlatformIDs(0, ptr::null_mut(), &mut num_of_platforms) };
    assert!(error >= 0, "clGetPlatformIDs()");
    let mut platforms: Vec<cl_platform_id> = vec![0 as cl_platform_id; num_of_platforms as usize];
    let error = unsafe {
        clGetPlatformIDs(num_of_platforms, platforms.as_mut_ptr(), ptr::null_mut())
    };
    assert!(error >= 0, "clGetPlatformIDs()");

    // create context for all GPU devices (use first found platform)
    let context = unsafe {
        let platform: cl_context_properties = mem::transmute(platforms[0]);
        let mut props: Vec<_> = vec![CL_CONTEXT_PLATFORM as cl_context_properties, platform, 0];
        let mut error: cl_int = 0;
        let context = clCreateContextFromType(props.as_mut_ptr(),
                                              CL_DEVICE_TYPE_GPU,
                                              None,
                                              ptr::null_mut(),
                                              &mut error as *mut _);
        assert!(error >= 0, "clCreateContextFromType()");
        context
    };

    // get devices from context
    let mut size_in_bytes = 0 as usize;
    let error = unsafe {
        clGetContextInfo(context,
                         CL_CONTEXT_DEVICES,
                         0,
                         ptr::null_mut(),
                         &mut size_in_bytes)
    };
    assert!(error >= 0, "clGetContextInfo()");
    let num_of_devices = size_in_bytes / mem::size_of::<cl_device_id>();
    let mut devices: Vec<cl_device_id> = vec![0 as cl_device_id; num_of_devices];
    let error = unsafe {
        clGetContextInfo(context,
                         CL_CONTEXT_DEVICES,
                         size_in_bytes,
                         devices.as_mut_ptr() as *mut c_void,
                         ptr::null_mut())
    };
    assert!(error >= 0, "clGetContextInfo()");

    // create command queue for the first found device
    let mut error: cl_int = 0;
    let cmd_queue = unsafe { clCreateCommandQueue(context, devices[0], 0, &mut error as *mut _) };
    assert!(error >= 0, "clCreateCommandQueue()");

    let elements = 1048 * 1048 * 50;
    let A: Vec<cl_int> = (0..elements).collect();
    let B: Vec<cl_int> = (0..elements).collect();
    let C: Vec<cl_int> = vec![0; elements as usize];

    let data_size = mem::size_of::<cl_int>() * elements as usize;
    let mut error: cl_int = 0;
    let bufA = unsafe {
        clCreateBuffer(context,
                       CL_MEM_READ_ONLY,
                       data_size,
                       ptr::null_mut(),
                       &mut error as *mut _)
    };
    assert!(error >= 0, "clCreateBuffer() for A");
    let bufB = unsafe {
        clCreateBuffer(context,
                       CL_MEM_READ_ONLY,
                       data_size,
                       ptr::null_mut(),
                       &mut error as *mut _)
    };
    assert!(error >= 0, "clCreateBuffer() for B");
    let bufC = unsafe {
        clCreateBuffer(context,
                       CL_MEM_WRITE_ONLY,
                       data_size,
                       ptr::null_mut(),
                       &mut error as *mut _)
    };
    assert!(error >= 0, "clCreateBuffer() for C");

    // Write input array A to the device buffer bufA
    let error = unsafe {
        clEnqueueWriteBuffer(cmd_queue,
                             bufA,
                             CL_FALSE,
                             0,
                             data_size,
                             A.as_ptr() as *const c_void,
                             0,
                             ptr::null_mut(),
                             ptr::null_mut())
    };
    assert!(error >= 0, "clEnqueueWriteBuffer() for bufA");
    // Write input array B to the device buffer bufB
    let error = unsafe {
        clEnqueueWriteBuffer(cmd_queue,
                             bufB,
                             CL_FALSE,
                             0,
                             data_size,
                             B.as_ptr() as *const c_void,
                             0,
                             ptr::null_mut(),
                             ptr::null_mut())
    };
    assert!(error >= 0, "clEnqueueWriteBuffer() for bufB");

    // Create a program with source code
    let mut error: cl_int = 0;
    let src_ptrs = vec![KERNEL_SRC.as_ptr()];
    let src_lens = vec![KERNEL_SRC.len() as size_t];
    let program = unsafe {
        clCreateProgramWithSource(context,
                                  src_lens.len() as cl_uint,
                                  src_ptrs.as_ptr() as *mut *const _,
                                  src_lens.as_ptr(),
                                  &mut error as *mut _)
    };
    assert!(error >= 0, "clCreateProgramWithSource()");

    // Build the programm
    let error = unsafe {
        clBuildProgram(program,
                       1,
                       devices.as_ptr() as *const _,
                       ptr::null_mut(),
                       None,
                       ptr::null_mut())
    };
    assert!(error >= 0, "clBuildProgram()");

    // Create the kernel
    let mut error: cl_int = 0;
    let kernel = unsafe {
        clCreateKernel(program,
                       CString::new("vecadd").unwrap().as_ptr() as *const i8,
                       &mut error as *mut _)
    };
    assert!(error >= 0, "clCreateKernel()");

    // Set kernel params
    let error = unsafe {
        clSetKernelArg(kernel,
                       0,
                       mem::size_of::<cl_mem>(),
                       &bufA as *const _ as *const c_void)
    };
    assert!(error >= 0, "clSetKernelArg() for bufA");
    let error = unsafe {
        clSetKernelArg(kernel,
                       1,
                       mem::size_of::<cl_mem>(),
                       &bufB as *const _ as *const c_void)
    };
    assert!(error >= 0, "clSetKernelArg() for bufB");
    let error = unsafe {
        clSetKernelArg(kernel,
                       2,
                       mem::size_of::<cl_mem>(),
                       &bufC as *const _ as *const c_void)
    };
    assert!(error >= 0, "clSetKernelArg() for bufC");

    let global_worksize = vec![elements as usize];
    let error = unsafe {
        clEnqueueNDRangeKernel(cmd_queue,
                               kernel,
                               1,
                               ptr::null_mut(),
                               global_worksize.as_ptr(),
                               ptr::null_mut(),
                               0,
                               ptr::null_mut(),
                               ptr::null_mut())
    };
    assert!(error >= 0, "clEnqueueNDRangeKernel()");

    // Get results
    let error = unsafe {
        clEnqueueReadBuffer(cmd_queue,
                            bufC,
                            CL_TRUE,
                            0,
                            data_size,
                            C.as_ptr() as cl_mem,
                            0,
                            ptr::null_mut(),
                            ptr::null_mut())
    };
    assert!(error >= 0, "clEnqueueReadBuffer()");

    // Free OpenCL resources
    unsafe {
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(cmd_queue);
        clReleaseMemObject(bufA);
        clReleaseMemObject(bufB);
        clReleaseMemObject(bufC);
        clReleaseContext(context);
    }

    // Verify the result
    for i in 0..elements {
        let i = i as usize;
        if C[i] != A[i] + B[i] {
            println!("Error at pos {}", i);
            break;
        }
    }
}
