//! Generic RSA implementation

use alloc::borrow::Cow;
use alloc::vec::Vec;
use num_bigint::{BigInt, BigUint, IntoBigInt, IntoBigUint, ModInverse, RandBigInt, ToBigInt};
use num_integer::{sqrt, Integer};
use num_traits::{FromPrimitive, One, Pow, Signed, Zero};
use rand_core::CryptoRngCore;
use zeroize::{Zeroize, Zeroizing};

#[cfg(all(target_os = "zkvm", target_vendor = "succinct"))]
use bytemuck::cast_ref;
#[cfg(all(target_os = "zkvm", target_vendor = "succinct"))]
use core::convert::TryInto;
#[cfg(all(target_os = "zkvm", target_vendor = "succinct"))]
use crypto_bigint::{Encoding, Integer as CryptoInteger, NonZero, U2048, U256, U3072, U4096, U6144, U8192};
#[cfg(all(target_os = "zkvm", target_vendor = "succinct"))]
use sp1_lib::io::hint_slice;

use crate::errors::{Error, Result};
use crate::traits::{PrivateKeyParts, PublicKeyParts};

/// ⚠️ Raw RSA encryption of m with the public key. No padding is performed.
///
/// # ☢️️ WARNING: HAZARDOUS API ☢️
///
/// Use this function with great care! Raw RSA should never be used without an appropriate padding
/// or signature scheme. See the [module-level documentation][crate::hazmat] for more information.
#[inline]
pub fn rsa_encrypt<K: PublicKeyParts>(key: &K, m: &BigUint) -> Result<BigUint> {
    #[cfg(all(target_os = "zkvm", target_vendor = "succinct"))]
    {
        match key.size() {
            256 => {
                use zkvm::*;
                let m_chunks = zkvm::from_biguint_to_chunks::<8>(m);
                let e_chunks = zkvm::from_biguint_to_chunks::<8>(key.e()); 
                let n_chunks = zkvm::from_biguint_to_chunks::<8>(key.n());
                let result = zkvm::custom_modpow_2048(&m_chunks, &e_chunks, &n_chunks);
                return Ok(result);
            },
            384 => {
                use zkvm::*;
                let m_chunks = zkvm::from_biguint_to_chunks::<12>(m);
                let e_chunks = zkvm::from_biguint_to_chunks::<12>(key.e()); 
                let n_chunks = zkvm::from_biguint_to_chunks::<12>(key.n());
                let result = zkvm::custom_modpow_3072(&m_chunks, &e_chunks, &n_chunks);
                return Ok(result);
            },
            512 => {
                use zkvm::*;
                let m_chunks = zkvm::from_biguint_to_chunks::<16>(m);
                let e_chunks = zkvm::from_biguint_to_chunks::<16>(key.e()); 
                let n_chunks = zkvm::from_biguint_to_chunks::<16>(key.n());
                let result = zkvm::custom_modpow_4096(&m_chunks, &e_chunks, &n_chunks);
                return Ok(result);
            },
            _ => {
                // Fall through to standard modpow for unsupported sizes
            }
        }
    }
    Ok(m.modpow(key.e(), key.n()))
}

/// # ☢️️ WARNING: HAZARDOUS API ☢️
///
/// All inputs are ASSUMED to be on the range of [0, 2^4096).]
///
/// Attempting to use this function with values outside of this range will result in truncation,
/// and may have unintended side effects!
#[cfg(all(target_os = "zkvm", target_vendor = "succinct"))]
mod zkvm {
    use super::*;

    // Macro to generate mul_mod functions for different bit sizes
    macro_rules! impl_mul_mod {
        ($name:ident, $chunks:expr, $bytes:expr, $fd_type:expr) => {
            fn $name(
                a_chunks: &[[u32; 8]; $chunks], 
                b_chunks: &[[u32; 8]; $chunks], 
                modulus_chunks: &[[u32; 8]; $chunks]
            ) -> [[u32; 8]; $chunks] {
                let prod_chunks = mul_generic_chunks::<$chunks, {$chunks * 2}>(a_chunks, b_chunks);
                
                // Convert to bytes for SP1 I/O using direct transmute
                let prod_bytes: [u8; $bytes * 2] = unsafe {
                    std::mem::transmute::<[[u32; 8]; $chunks * 2], [u8; $bytes * 2]>(prod_chunks)
                };
                let modulus_bytes: [u8; $bytes] = unsafe {
                    std::mem::transmute::<[[u32; 8]; $chunks], [u8; $bytes]>(*modulus_chunks)
                };
                
                // Call the hook to perform the modmul operation in the executor
                sp1_lib::io::write(
                    $fd_type,
                    &prod_bytes.into_iter().chain(modulus_bytes.into_iter()).collect::<Vec<_>>(),
                );

                let result_bytes: [u8; $bytes] = sp1_lib::io::read_vec().try_into().unwrap();
                let quotient_bytes: [u8; $bytes] = sp1_lib::io::read_vec().try_into().unwrap();

                // Convert back to chunks
                let result_chunks: [[u32; 8]; $chunks] = unsafe {
                    std::mem::transmute::<[u8; $bytes], [[u32; 8]; $chunks]>(result_bytes)
                };
                let quotient_chunks: [[u32; 8]; $chunks] = unsafe {
                    std::mem::transmute::<[u8; $bytes], [[u32; 8]; $chunks]>(quotient_bytes)
                };
                
                // Verify: prod == quotient * modulus + result and 0 <= result < modulus.
                let quotient_mul_chunks = mul_generic_chunks::<$chunks, {$chunks * 2}>(&quotient_chunks, modulus_chunks);
                
                let mut verification_prod = quotient_mul_chunks;
                add_generic_chunks(&mut verification_prod, &result_chunks);
                
                for i in 0..($chunks * 2) {
                    for j in 0..8 {
                        assert_eq!(prod_chunks[i][j], verification_prod[i][j], "equality check failed");
                    }
                }

                assert_less_than::<$chunks>(&result_chunks, modulus_chunks);
                
                result_chunks
            }
        };
    }

    // Generate the three mul_mod functions
    impl_mul_mod!(mul_mod_2048, 8, 256, sp1_lib::io::FD_RSA_MUL_MOD);
    impl_mul_mod!(mul_mod_3072, 12, 384, sp1_lib::io::FD_RSA_MUL_MOD);
    impl_mul_mod!(mul_mod_4096, 16, 512, sp1_lib::io::FD_RSA_MUL_MOD);

    // Macro to generate modpow functions
    macro_rules! impl_modpow {
        ($name:ident, $chunks:expr, $bigint_type:ty, $mul_mod_fn:ident) => {
            pub(super) fn $name(
                base_chunks: &[[u32; 8]; $chunks], 
                exp_chunks: &[[u32; 8]; $chunks], 
                modulus_chunks: &[[u32; 8]; $chunks]
            ) -> BigUint {
                // Convert chunks to crypto_bigint type for easier manipulation
                let exp_bytes = chunks_to_bytes::<$chunks>(exp_chunks);
                let exp_bigint = <$bigint_type>::from_le_slice(&exp_bytes);
                
                assert!(!chunks_is_zero::<$chunks>(modulus_chunks), "modulo cannot be zero");
                
                let result_chunks = if exp_bigint == <$bigint_type>::from_u64(65537u64) {
                    // Optimized path for e = 65537
                    let one_chunks = chunks_one::<$chunks>();
                    let mut result_chunks = $mul_mod_fn(base_chunks, &one_chunks, modulus_chunks);
                    let base_reduced = result_chunks;
                    
                    // Square 16 times
                    for _ in 0..16 {
                        result_chunks = $mul_mod_fn(&result_chunks, &result_chunks, modulus_chunks);
                    }
                    
                    $mul_mod_fn(&result_chunks, &base_reduced, modulus_chunks)
                } else {
                    // Square-and-multiply
                    let one_chunks = chunks_one::<$chunks>();
                    let mut result_chunks = one_chunks;
                    let mut base_chunks = $mul_mod_fn(base_chunks, &one_chunks, modulus_chunks);
                    let mut exp = exp_bigint;
                    
                    while exp > <$bigint_type>::ZERO {
                        if exp.is_odd().into() {
                            result_chunks = $mul_mod_fn(&result_chunks, &base_chunks, modulus_chunks);
                        }
                        exp = exp.shr(1);
                        base_chunks = $mul_mod_fn(&base_chunks, &base_chunks, modulus_chunks);
                    }
                    
                    result_chunks
                };
                
                BigUint::from_bytes_le(&chunks_to_bytes::<$chunks>(&result_chunks))
            }
        };
    }

    // Generate the three modpow functions
    impl_modpow!(custom_modpow_2048, 8, U2048, mul_mod_2048);
    impl_modpow!(custom_modpow_3072, 12, U3072, mul_mod_3072);
    impl_modpow!(custom_modpow_4096, 16, U4096, mul_mod_4096);
    
    /// Generic multiplication using schoolbook algorithm with 256-bit chunks
    /// Returns a vector of 256-bit chunks representing the full product
    fn mul_generic_chunks<const N: usize, const N2: usize>(a_chunks: &[[u32; 8]; N], b_chunks: &[[u32; 8]; N]) -> [[u32; 8]; N2] {
        let mut out = [[0u32; 8]; N2];
        
        let mut lo = [0u32; 8];
        let mut hi = [0u32; 8];
        let mut tmp_hi = [0u32; 8];
        let zero_carry = [0u32; 8];
        
        for i in 0..N {
            for j in 0..N {
                let k = i + j;
                
                unsafe {
                    sp1_lib::syscall_uint256_mul_with_carry(
                        a_chunks[i].as_ptr() as *const [u32; 8],
                        b_chunks[j].as_ptr() as *const [u32; 8],
                        zero_carry.as_ptr() as *const [u32; 8],
                        lo.as_mut_ptr() as *mut [u32; 8],
                        hi.as_mut_ptr() as *mut [u32; 8],
                    );
                }
                
                unsafe {
                    sp1_lib::syscall_uint256_add_with_carry(
                        out[k].as_ptr() as *const [u32; 8],
                        lo.as_ptr() as *const [u32; 8],
                        zero_carry.as_ptr() as *const [u32; 8],
                        out[k].as_mut_ptr() as *mut [u32; 8],
                        tmp_hi.as_mut_ptr() as *mut [u32; 8],
                    );
                }

                unsafe {
                    sp1_lib::syscall_uint256_add_with_carry(
                        out[k + 1].as_ptr() as *const [u32; 8],
                        hi.as_ptr() as *const [u32; 8],
                        tmp_hi.as_ptr() as *const [u32; 8],
                        out[k + 1].as_mut_ptr() as *mut [u32; 8],
                        tmp_hi.as_mut_ptr() as *mut [u32; 8],
                    );
                }
                
                let mut idx = k + 2;
                while tmp_hi[0] != 0 && idx < N2 {
                    unsafe {
                        sp1_lib::syscall_uint256_add_with_carry(
                            out[idx].as_ptr() as *const [u32; 8],
                            zero_carry.as_ptr() as *const [u32; 8],
                            tmp_hi.as_ptr() as *const [u32; 8],
                            out[idx].as_mut_ptr() as *mut [u32; 8],
                            tmp_hi.as_mut_ptr() as *mut [u32; 8],
                        );
                    }
                    idx += 1;
                }
            }
        }
        
        out
    }

    /// Generic addition of two chunk arrays with different sizes
    /// Adds smaller array to the lower part of larger array, handling carries
    fn add_generic_chunks(larger: &mut [[u32; 8]], smaller: &[[u32; 8]]) {
        let mut carry = [0u32; 8];
        let zero_carry = [0u32; 8];
        
        for i in 0..smaller.len() {
            unsafe {
                sp1_lib::syscall_uint256_add_with_carry(
                    larger[i].as_ptr() as *const [u32; 8],
                    smaller[i].as_ptr() as *const [u32; 8],
                    carry.as_ptr() as *const [u32; 8],
                    larger[i].as_mut_ptr() as *mut [u32; 8],
                    carry.as_mut_ptr() as *mut [u32; 8],
                );
            }
        }
        
        let mut idx = smaller.len();
        while idx < larger.len() && carry[0] != 0 {
            unsafe {
                sp1_lib::syscall_uint256_add_with_carry(
                    larger[idx].as_ptr() as *const [u32; 8],
                    zero_carry.as_ptr() as *const [u32; 8],
                    carry.as_ptr() as *const [u32; 8],
                    larger[idx].as_mut_ptr() as *mut [u32; 8],
                    carry.as_mut_ptr() as *mut [u32; 8],
                );
            }
            idx += 1;
        }
    }

    /// Assert that the result is less than the modulus
    fn assert_less_than<const N: usize>(result_chunk: &[[u32; 8]; N], modulus_chunk: &[[u32; 8]; N]) {
        for i in (0..N).rev() {
            for j in (0..8).rev() {
                if result_chunk[i][j] < modulus_chunk[i][j] {
                    return;
                }
                assert!(result_chunk[i][j] == modulus_chunk[i][j], "result < modulus check failed");
            }
        }
        assert!(false, "result < modulus check failed");
    }
    
    /// Generic helper to convert bytes to chunks
    fn bytes_to_chunks<const N: usize>(bytes: &[u8]) -> [[u32; 8]; N] {
        let mut chunks = [[0u32; 8]; N];
        assert!(bytes.len() == 32 * N, "incorrect length");
        for (i, chunk) in chunks.iter_mut().enumerate() {
            for (j, word) in chunk.iter_mut().enumerate() {
                let byte_idx = (i * 8 + j) * 4;
                *word = u32::from_le_bytes([
                    bytes[byte_idx],
                    bytes[byte_idx + 1], 
                    bytes[byte_idx + 2],
                    bytes[byte_idx + 3],
                ]);
            }
        }
        chunks
    }
    
    /// Generic helper to convert chunks to bytes
    fn chunks_to_bytes<const N: usize>(chunks: &[[u32; 8]; N]) -> Vec<u8> {
        chunks.iter()
            .flat_map(|chunk| chunk.iter().flat_map(|&word| word.to_le_bytes()))
            .collect()
    }
    
    /// Check if chunk array is zero
    fn chunks_is_zero<const N: usize>(chunks: &[[u32; 8]; N]) -> bool {
        for i in 0..N {
            for j in 0..8 {
                if chunks[i][j] != 0 {
                    return false;
                }
            }
        }
        true
    }
    
    /// Get chunk array representing one
    fn chunks_one<const N: usize>() -> [[u32; 8]; N] {
        let mut chunks = [[0u32; 8]; N];
        chunks[0][0] = 1;
        chunks
    }
    
    /// Convert BigUint to chunks for arbitrary key sizes
    pub(super) fn from_biguint_to_chunks<const N: usize>(value: &BigUint) -> [[u32; 8]; N] {
        let mut padded_bytes = vec![0u8; N * 32];
        let value_bytes = value.to_bytes_le();
        for (i, &byte) in value_bytes.iter().enumerate() {
            if i >= padded_bytes.len() {
                panic!("value larger than allowed size");
            }
            padded_bytes[i] = byte;
        }
        bytes_to_chunks(&padded_bytes)
    }

}

/// ⚠️ Performs raw RSA decryption with no padding or error checking.
///
/// Returns a plaintext `BigUint`. Performs RSA blinding if an `Rng` is passed.
///
/// # ☢️️ WARNING: HAZARDOUS API ☢️
///
/// Use this function with great care! Raw RSA should never be used without an appropriate padding
/// or signature scheme. See the [module-level documentation][crate::hazmat] for more information.
#[inline]
pub fn rsa_decrypt<R: CryptoRngCore + ?Sized>(
    mut rng: Option<&mut R>,
    priv_key: &impl PrivateKeyParts,
    c: &BigUint,
) -> Result<BigUint> {
    if c >= priv_key.n() {
        return Err(Error::Decryption);
    }

    if priv_key.n().is_zero() {
        return Err(Error::Decryption);
    }

    let mut ir = None;

    let c = if let Some(ref mut rng) = rng {
        let (blinded, unblinder) = blind(rng, priv_key, c);
        ir = Some(unblinder);
        Cow::Owned(blinded)
    } else {
        Cow::Borrowed(c)
    };

    let dp = priv_key.dp();
    let dq = priv_key.dq();
    let qinv = priv_key.qinv();
    let crt_values = priv_key.crt_values();

    let m = match (dp, dq, qinv, crt_values) {
        (Some(dp), Some(dq), Some(qinv), Some(crt_values)) => {
            // We have the precalculated values needed for the CRT.

            let p = &priv_key.primes()[0];
            let q = &priv_key.primes()[1];

            let mut m = c.modpow(dp, p).into_bigint().unwrap();
            let mut m2 = c.modpow(dq, q).into_bigint().unwrap();

            m -= &m2;

            let mut primes: Vec<_> = priv_key
                .primes()
                .iter()
                .map(ToBigInt::to_bigint)
                .map(Option::unwrap)
                .collect();

            while m.is_negative() {
                m += &primes[0];
            }
            m *= qinv;
            m %= &primes[0];
            m *= &primes[1];
            m += &m2;

            let mut c = c.into_owned().into_bigint().unwrap();
            for (i, value) in crt_values.iter().enumerate() {
                let prime = &primes[2 + i];
                m2 = c.modpow(&value.exp, prime);
                m2 -= &m;
                m2 *= &value.coeff;
                m2 %= prime;
                while m2.is_negative() {
                    m2 += prime;
                }
                m2 *= &value.r;
                m += &m2;
            }

            // clear tmp values
            for prime in primes.iter_mut() {
                prime.zeroize();
            }
            primes.clear();
            c.zeroize();
            m2.zeroize();

            m.into_biguint().expect("failed to decrypt")
        }
        _ => c.modpow(priv_key.d(), priv_key.n()),
    };

    match ir {
        Some(ref ir) => {
            // unblind
            Ok(unblind(priv_key, &m, ir))
        }
        None => Ok(m),
    }
}

/// ⚠️ Performs raw RSA decryption with no padding.
///
/// Returns a plaintext `BigUint`. Performs RSA blinding if an `Rng` is passed.  This will also
/// check for errors in the CRT computation.
///
/// # ☢️️ WARNING: HAZARDOUS API ☢️
///
/// Use this function with great care! Raw RSA should never be used without an appropriate padding
/// or signature scheme. See the [module-level documentation][crate::hazmat] for more information.
#[inline]
pub fn rsa_decrypt_and_check<R: CryptoRngCore + ?Sized>(
    priv_key: &impl PrivateKeyParts,
    rng: Option<&mut R>,
    c: &BigUint,
) -> Result<BigUint> {
    let m = rsa_decrypt(rng, priv_key, c)?;

    // In order to defend against errors in the CRT computation, m^e is
    // calculated, which should match the original ciphertext.
    let check = rsa_encrypt(priv_key, &m)?;

    if c != &check {
        return Err(Error::Internal);
    }

    Ok(m)
}

/// Returns the blinded c, along with the unblinding factor.
fn blind<R: CryptoRngCore, K: PublicKeyParts>(
    rng: &mut R,
    key: &K,
    c: &BigUint,
) -> (BigUint, BigUint) {
    // Blinding involves multiplying c by r^e.
    // Then the decryption operation performs (m^e * r^e)^d mod n
    // which equals mr mod n. The factor of r can then be removed
    // by multiplying by the multiplicative inverse of r.

    let mut r: BigUint;
    let mut ir: Option<BigInt>;
    let unblinder;
    loop {
        r = rng.gen_biguint_below(key.n());
        if r.is_zero() {
            r = BigUint::one();
        }
        ir = r.clone().mod_inverse(key.n());
        if let Some(ir) = ir {
            if let Some(ub) = ir.into_biguint() {
                unblinder = ub;
                break;
            }
        }
    }

    let c = {
        let mut rpowe = r.modpow(key.e(), key.n()); // N != 0
        let mut c = c * &rpowe;
        c %= key.n();

        rpowe.zeroize();

        c
    };

    (c, unblinder)
}

/// Given an m and and unblinding factor, unblind the m.
fn unblind(key: &impl PublicKeyParts, m: &BigUint, unblinder: &BigUint) -> BigUint {
    (m * unblinder) % key.n()
}

/// The following (deterministic) algorithm also recovers the prime factors `p` and `q` of a modulus `n`, given the
/// public exponent `e` and private exponent `d` using the method described in
/// [NIST 800-56B Appendix C.2](https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-56Br2.pdf).
pub fn recover_primes(n: &BigUint, e: &BigUint, d: &BigUint) -> Result<(BigUint, BigUint)> {
    // Check precondition
    let two = BigUint::from_u8(2).unwrap();
    if e <= &two.pow(16u32) || e >= &two.pow(256u32) {
        return Err(Error::InvalidArguments);
    }

    // 1. Let a = (de – 1) × GCD(n – 1, de – 1).
    let one = BigUint::one();
    let a = Zeroizing::new((d * e - &one) * (n - &one).gcd(&(d * e - &one)));

    // 2. Let m = floor(a /n) and r = a – m n, so that a = m n + r and 0 ≤ r < n.
    let m = Zeroizing::new(&*a / n);
    let r = Zeroizing::new(&*a - &*m * n);

    // 3. Let b = ( (n – r)/(m + 1) ) + 1; if b is not an integer or b^2 ≤ 4n, then output an error indicator,
    //    and exit without further processing.
    let modulus_check = Zeroizing::new((n - &*r) % (&*m + &one));
    if !modulus_check.is_zero() {
        return Err(Error::InvalidArguments);
    }
    let b = Zeroizing::new((n - &*r) / (&*m + &one) + one);

    let four = BigUint::from_u8(4).unwrap();
    let four_n = Zeroizing::new(n * four);
    let b_squared = Zeroizing::new(b.pow(2u32));
    if *b_squared <= *four_n {
        return Err(Error::InvalidArguments);
    }
    let b_squared_minus_four_n = Zeroizing::new(&*b_squared - &*four_n);

    // 4. Let ϒ be the positive square root of b^2 – 4n; if ϒ is not an integer,
    //    then output an error indicator, and exit without further processing.
    let y = Zeroizing::new(sqrt((*b_squared_minus_four_n).clone()));

    let y_squared = Zeroizing::new(y.pow(2u32));
    let sqrt_is_whole_number = y_squared == b_squared_minus_four_n;
    if !sqrt_is_whole_number {
        return Err(Error::InvalidArguments);
    }
    let p = (&*b + &*y) / &two;
    let q = (&*b - &*y) / two;

    Ok((p, q))
}

/// Compute the modulus of a key from its primes.
pub(crate) fn compute_modulus(primes: &[BigUint]) -> BigUint {
    primes.iter().product()
}

/// Compute the private exponent from its primes (p and q) and public exponent
/// This uses Euler's totient function
#[inline]
pub(crate) fn compute_private_exponent_euler_totient(
    primes: &[BigUint],
    exp: &BigUint,
) -> Result<BigUint> {
    if primes.len() < 2 {
        return Err(Error::InvalidPrime);
    }

    let mut totient = BigUint::one();

    for prime in primes {
        totient *= prime - BigUint::one();
    }

    // NOTE: `mod_inverse` checks if `exp` evenly divides `totient` and returns `None` if so.
    // This ensures that `exp` is not a factor of any `(prime - 1)`.
    if let Some(d) = exp.mod_inverse(totient) {
        Ok(d.to_biguint().unwrap())
    } else {
        // `exp` evenly divides `totient`
        Err(Error::InvalidPrime)
    }
}

/// Compute the private exponent from its primes (p and q) and public exponent
///
/// This is using the method defined by
/// [NIST 800-56B Section 6.2.1](https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-56Br2.pdf#page=47).
/// (Carmichael function)
///
/// FIPS 186-4 **requires** the private exponent to be less than λ(n), which would
/// make Euler's totiem unreliable.
#[inline]
pub(crate) fn compute_private_exponent_carmicheal(
    p: &BigUint,
    q: &BigUint,
    exp: &BigUint,
) -> Result<BigUint> {
    let p1 = p - BigUint::one();
    let q1 = q - BigUint::one();

    let lcm = p1.lcm(&q1);
    if let Some(d) = exp.mod_inverse(lcm) {
        Ok(d.to_biguint().unwrap())
    } else {
        // `exp` evenly divides `lcm`
        Err(Error::InvalidPrime)
    }
}

#[cfg(test)]
mod tests {
    use num_traits::FromPrimitive;

    use super::*;

    #[test]
    fn recover_primes_works() {
        let n = BigUint::parse_bytes(b"00d397b84d98a4c26138ed1b695a8106ead91d553bf06041b62d3fdc50a041e222b8f4529689c1b82c5e71554f5dd69fa2f4b6158cf0dbeb57811a0fc327e1f28e74fe74d3bc166c1eabdc1b8b57b934ca8be5b00b4f29975bcc99acaf415b59bb28a6782bb41a2c3c2976b3c18dbadef62f00c6bb226640095096c0cc60d22fe7ef987d75c6a81b10d96bf292028af110dc7cc1bbc43d22adab379a0cd5d8078cc780ff5cd6209dea34c922cf784f7717e428d75b5aec8ff30e5f0141510766e2e0ab8d473c84e8710b2b98227c3db095337ad3452f19e2b9bfbccdd8148abf6776fa552775e6e75956e45229ae5a9c46949bab1e622f0e48f56524a84ed3483b", 16).unwrap();
        let e = BigUint::from_u64(65537).unwrap();
        let d = BigUint::parse_bytes(b"00c4e70c689162c94c660828191b52b4d8392115df486a9adbe831e458d73958320dc1b755456e93701e9702d76fb0b92f90e01d1fe248153281fe79aa9763a92fae69d8d7ecd144de29fa135bd14f9573e349e45031e3b76982f583003826c552e89a397c1a06bd2163488630d92e8c2bb643d7abef700da95d685c941489a46f54b5316f62b5d2c3a7f1bbd134cb37353a44683fdc9d95d36458de22f6c44057fe74a0a436c4308f73f4da42f35c47ac16a7138d483afc91e41dc3a1127382e0c0f5119b0221b4fc639d6b9c38177a6de9b526ebd88c38d7982c07f98a0efd877d508aae275b946915c02e2e1106d175d74ec6777f5e80d12c053d9c7be1e341", 16).unwrap();
        let p = BigUint::parse_bytes(b"00f827bbf3a41877c7cc59aebf42ed4b29c32defcb8ed96863d5b090a05a8930dd624a21c9dcf9838568fdfa0df65b8462a5f2ac913d6c56f975532bd8e78fb07bd405ca99a484bcf59f019bbddcb3933f2bce706300b4f7b110120c5df9018159067c35da3061a56c8635a52b54273b31271b4311f0795df6021e6355e1a42e61",16).unwrap();
        let q = BigUint::parse_bytes(b"00da4817ce0089dd36f2ade6a3ff410c73ec34bf1b4f6bda38431bfede11cef1f7f6efa70e5f8063a3b1f6e17296ffb15feefa0912a0325b8d1fd65a559e717b5b961ec345072e0ec5203d03441d29af4d64054a04507410cf1da78e7b6119d909ec66e6ad625bf995b279a4b3c5be7d895cd7c5b9c4c497fde730916fcdb4e41b", 16).unwrap();

        let (mut p1, mut q1) = recover_primes(&n, &e, &d).unwrap();

        if p1 < q1 {
            std::mem::swap(&mut p1, &mut q1);
        }
        assert_eq!(p, p1);
        assert_eq!(q, q1);
    }
}
