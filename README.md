# range Bits Back Coding

This repo implements Bits back coding based on range Asymmetric Numeral System (rANS) as a final project for EE274 Data compression, called range Bits Back Coding (rBBC). Bits back coding is introduced in [Compressing Multisets with Large Alphabets](
https://arxiv.org/abs/2107.09202), which is based on ANS. We extend the implementation to rANS. 

In short, we can compress multiset to the information limit. In comparison, directly apply rANS will record the order information, which is redundant. The numerical experiment illustrates this point, as rBBC has better compression performance than rANS on distributions which are close to a uniform distribution.

## How to use?

It is the same as rANS! For detailed usage, you may check the test functions in BitsBackCoding.py.

## Contact
Yifei Wang wangyf18 at stanford dot edu.