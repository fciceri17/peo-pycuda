typedef struct {

  unsigned long long int lo;

  unsigned long long int hi;

} my_uint128;

__host__ __device__ inline bool operator==(const my_uint128& lhs, const my_uint128& rhs){ return lhs.lo == rhs.lo && lhs.hi == rhs.hi; }
__host__ __device__ inline bool operator>(const my_uint128& lhs, const my_uint128& rhs){ return (lhs.lo > rhs.lo && lhs.hi == rhs.hi) || lhs.hi > rhs.hi; }

__host__ __device__ my_uint128 int_to_my_uint128(int a)
{
    my_uint128 res;
    res.lo = a;
    res.hi = 0;
    return res;
}

__host__ __device__ my_uint128 llint_to_uint128(unsigned long long int a, unsigned long long int b)
{
    my_uint128 res;
    res.hi = a;
    res.lo = b;
    return res;
}


__host__ __device__ my_uint128 add_my_uint128 (my_uint128 a, my_uint128 b)

{

  my_uint128 res;

  res.lo = a.lo + b.lo;

  res.hi = a.hi + b.hi + (res.lo < a.lo);

  return res;

}



__host__ __device__ my_uint128 sub_my_uint128 (my_uint128 a, my_uint128 b)

{

  my_uint128 res;

  res.lo = a.lo - b.lo;

  res.hi = a.hi - b.hi - (res.lo > a.lo);

  return res;

}



__host__ __device__ my_uint128 shl_my_uint128 (my_uint128 a, int s)

{

  if (s) {

    a.hi = (a.hi << s) | (a.lo >> (64 - s));

    a.lo =  a.lo << s;

  }

  return a;

}



__host__ __device__ my_uint128 mul10_my_uint128 (my_uint128 a)

{

  my_uint128 s, t;

  s = shl_my_uint128 (a, 3);

  t = shl_my_uint128 (a, 1);

  return add_my_uint128 (s, t);

}



static const my_uint128 pwrten [] =

{

  {0x0000000000000001, 0x0000000000000000}, /* 10**0  */

  {0x000000000000000a, 0x0000000000000000}, /* 10**1  */

  {0x0000000000000064, 0x0000000000000000}, /* 10**2  */

  {0x00000000000003e8, 0x0000000000000000}, /* 10**3  */

  {0x0000000000002710, 0x0000000000000000}, /* 10**4  */

  {0x00000000000186a0, 0x0000000000000000}, /* 10**5  */

  {0x00000000000f4240, 0x0000000000000000}, /* 10**6  */

  {0x0000000000989680, 0x0000000000000000}, /* 10**7  */

  {0x0000000005f5e100, 0x0000000000000000}, /* 10**8  */

  {0x000000003b9aca00, 0x0000000000000000}, /* 10**9  */

  {0x00000002540be400, 0x0000000000000000}, /* 10**10 */

  {0x000000174876e800, 0x0000000000000000}, /* 10**11 */

  {0x000000e8d4a51000, 0x0000000000000000}, /* 10**12 */

  {0x000009184e72a000, 0x0000000000000000}, /* 10**13 */

  {0x00005af3107a4000, 0x0000000000000000}, /* 10**14 */

  {0x00038d7ea4c68000, 0x0000000000000000}, /* 10**15 */

  {0x002386f26fc10000, 0x0000000000000000}, /* 10**16 */

  {0x016345785d8a0000, 0x0000000000000000}, /* 10**17 */

  {0x0de0b6b3a7640000, 0x0000000000000000}, /* 10**18 */

  {0x8ac7230489e80000, 0x0000000000000000}, /* 10**19 */

  {0x6bc75e2d63100000, 0x0000000000000005}, /* 10**20 */

  {0x35c9adc5dea00000, 0x0000000000000036}, /* 10**21 */

  {0x19e0c9bab2400000, 0x000000000000021e}, /* 10**22 */

  {0x02c7e14af6800000, 0x000000000000152d}, /* 10**23 */

  {0x1bcecceda1000000, 0x000000000000d3c2}, /* 10**24 */

  {0x161401484a000000, 0x0000000000084595}, /* 10**25 */

  {0xdcc80cd2e4000000, 0x000000000052b7d2}, /* 10**26 */

  {0x9fd0803ce8000000, 0x00000000033b2e3c}, /* 10**27 */

  {0x3e25026110000000, 0x00000000204fce5e}, /* 10**28 */

  {0x6d7217caa0000000, 0x00000001431e0fae}, /* 10**29 */

  {0x4674edea40000000, 0x0000000c9f2c9cd0}, /* 10**30 */

  {0xc0914b2680000000, 0x0000007e37be2022}, /* 10**31 */

  {0x85acef8100000000, 0x000004ee2d6d415b}, /* 10**32 */

  {0x38c15b0a00000000, 0x0000314dc6448d93}, /* 10**33 */

  {0x378d8e6400000000, 0x0001ed09bead87c0}, /* 10**34 */

  {0x2b878fe800000000, 0x0013426172c74d82}, /* 10**35 */

  {0xb34b9f1000000000, 0x00c097ce7bc90715}, /* 10**36 */

  {0x00f436a000000000, 0x0785ee10d5da46d9}, /* 10**37 */

};

#define MAX_PWR ((int)(sizeof(pwrten)/sizeof(pwrten[0]))-1)

#define DIGITS  (MAX_PWR+1)



void cvt_my_uint128_to_str (my_uint128 a, char *cp) 

{

  my_uint128 t;

  int pwr, bit, non_zero, digit, remainder_neg;

  non_zero = 0;

  for (pwr = MAX_PWR; pwr >= 0; pwr--) {

    digit = 0;

    for (bit = 3; bit >= 0; bit--) {

      t = shl_my_uint128 (pwrten[pwr], bit);

      a = sub_my_uint128 (a, t);

      remainder_neg = ((long long int)a.hi) < 0;

      digit = (digit << 1) | !remainder_neg;

      if (remainder_neg) {

        a = add_my_uint128 (a, t);

      }

    }

    non_zero |= digit;

    if (non_zero || pwr == 0) {

      *cp++ = '0' + digit;

    }

    *cp = 0;

  }

}



my_uint128 cvt_str_to_my_uint128 (char *cp)

{

  my_uint128 a = {0, 0};

  my_uint128 t = {0, 0};

  while (*cp) {

    a = mul10_my_uint128 (a);

    t.lo = *cp++ - '0';

    a = add_my_uint128 (a, t);

  }

  return a;

}
