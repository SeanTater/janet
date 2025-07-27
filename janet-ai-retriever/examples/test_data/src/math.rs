//! Mathematical utility functions

/// Add two numbers together
///
/// # Examples
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

/// Subtract the second number from the first
pub fn subtract(a: i32, b: i32) -> i32 {
    a - b
}

/// Multiply two numbers
pub fn multiply(a: i32, b: i32) -> i32 {
    a * b
}

/// Divide the first number by the second
///
/// # Panics
///
/// Panics if the divisor is zero.
pub fn divide(a: i32, b: i32) -> i32 {
    if b == 0 {
        panic!("Cannot divide by zero");
    }
    a / b
}

/// Calculate the factorial of a number
pub fn factorial(n: u32) -> u64 {
    match n {
        0 | 1 => 1,
        _ => n as u64 * factorial(n - 1),
    }
}

/// Find the greatest common divisor of two numbers using Euclidean algorithm
pub fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
    }
    a
}
