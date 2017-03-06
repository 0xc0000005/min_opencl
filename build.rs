use std::env;

#[cfg_attr(rustfmt, rustfmt_skip)]
fn main() {
    let target = env::var("TARGET").unwrap();
    let x86_64 = target.starts_with("x86_64");
    let amd_sdk = env::var_os("AMDAPPSDKROOT")
                      .expect("Environment variable AMDAPPSDKROOT is not set");

    println!("cargo:rustc-link-lib=dylib=OpenCL");
    println!("cargo:rustc-link-search={}/lib/{}",
             amd_sdk.into_string().unwrap(),
             if x86_64 {"x86_64"} else {"x86"});
    // for value in env::vars() {
    //     println!("{} -> {}", value.0, value.1);
    // }
    //panic!("to debug build script");
}
