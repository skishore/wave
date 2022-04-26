use std::cell::UnsafeCell;

extern { fn show_vec_u32(data: *const u32, size: usize); }

thread_local! {
  static VALUES: UnsafeCell<Vec<u32>> = vec![].into();
}

#[no_mangle]
pub extern "C" fn add(x: i32, y: i32) -> i32 {
  return x + y;
}

#[no_mangle]
pub extern "C" fn range(n: usize) -> () {
  VALUES.with(|values| {
    let result = unsafe { &mut *values.get() };
    result.clear();
    result.reserve(n);
    for i in 0..n {
      result.push((n - i - 1) as u32);
    }
    unsafe { show_vec_u32(result.as_ptr(), result.len()); }
  });
}
