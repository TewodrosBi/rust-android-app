pub mod crossfader;
pub mod isolator;

/// Flush subnormal floating-point values to zero to prevent CPU microcode
/// traps on recursive IIR filter states when audio decays into silence.
#[inline(always)]
pub fn sanitize_denormal(val: f32) -> f32 {
    if val.abs() < 1.0e-20_f32 {
        0.0_f32
    } else {
        val
    }
}