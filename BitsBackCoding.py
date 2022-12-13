# This implements Bits-back-coding for compressing multi-set based on rANS, although 
# the original paper is based on ANS.
# Compressing Multisets with Large Alphabets
# https://arxiv.org/abs/2107.09202

from multiset import Multiset
import compressors.rANS as rANS
from dataclasses import dataclass
import numpy as np
from typing import Tuple, Any, List
from core.data_encoder_decoder import DataDecoder, DataEncoder
from utils.bitarray_utils import BitArray, get_bit_width, uint_to_bitarray, bitarray_to_uint
from core.data_block import DataBlock
from core.prob_dist import Frequencies, get_avg_neg_log_prob
from utils.test_utils import get_random_data_block, try_lossless_compression
from utils.misc_utils import cache

# add an option for debugging

@dataclass
class rBBCParams:
    """base parameters for the rANS encoder/decoder.
    More details in the overview
    """

    ## define global params
    freqs: Frequencies

    debug: False

    # num bits used to represent the data_block size
    DATA_BLOCK_SIZE_BITS: int = 32

    # the encoder can output NUM_BITS_OUT at a time when it performs the state shrinking operation
    NUM_BITS_OUT: int = 1  # number of bits

    # rANS state is limited to the range [RANGE_FACTOR*total_freq, (2**NUM_BITS_OUT)*RANGE_FACTOR*total_freq - 1)]
    # RANGE_FACTOR is a base parameter controlling this range
    RANGE_FACTOR: int = 1 << 16

    def __post_init__(self):
        ## define derived params
        # M -> sum of frequencies
        self.M = self.freqs.total_freq

        # the state always lies in the range [L,H]
        self.L = self.RANGE_FACTOR * self.M
        self.H = self.L * (1 << self.NUM_BITS_OUT) - 1

        # define min max range for shrunk_state (useful during encoding)
        self.min_shrunk_state = {}
        self.max_shrunk_state = {}
        for s in self.freqs.alphabet:
            f = self.freqs.frequency(s)
            self.min_shrunk_state[s] = self.RANGE_FACTOR * f
            self.max_shrunk_state[s] = self.RANGE_FACTOR * f * (1 << self.NUM_BITS_OUT) - 1

        ## define initial state, state bits etc.
        # NOTE: the choice of  this state is somewhat arbitrary, the only condition being, it should lie in the acceptable range [L, H]
        self.INITIAL_STATE = self.L

        # define num bits used to represent the final state
        self.NUM_STATE_BITS = get_bit_width(self.H)
        self.BITS_OUT_MASK = 1 << self.NUM_BITS_OUT

class rBBCMultiSetEncoder(rANS.rANSEncoder):
    
    def expand_state_local(self, state: int) -> int:
        # remap the state into the acceptable range
        while state < self.params.L:
            state = (state << self.params.NUM_BITS_OUT) 
        return state

    def encode_symbol(self, s, state: int, M_in: Multiset, in_bits: BitArray) -> Tuple[int, BitArray, Multiset]:
        """Encodes the next symbol, returns some bits and  the updated state

        Args:
            s (Any): next symbol to be encoded
            state (int): the rANS state
            M_in (Multiset): the Multiset

        Returns:
            state (int), symbol_bitarray (BitArray), M_out (Multiset):
        """
        # output bits to the stream so that the state is in the acceptable range
        # [L, H] *after*the `rans_base_encode_step`
        
        # begin additional steps in bits back coding
        
        freqs_local = Frequencies(M_in)
        rParams_local = rANS.rANSParams(freqs_local)
        decoder_local = rANS.rANSDecoder(rParams_local)
        state_start = state
        s, state = decoder_local.rans_base_decode_step(state)
        if self.params.debug:
            print('decode {}'.format(state))
#         state = self.expand_state(state,in_bits)
        state = self.expand_state_local(state)
        if self.params.debug:
            print('expand {}'.format(state))
        # end additional steps in bits back coding
        state, out_bits = self.shrink_state(state, s)
        M_out = M_in - Multiset(s)
        if self.params.debug:
            print('shrink {}'.format(state))
        # NOTE: we are prepending bits for pedagogy. In practice, it might be faster to assign a larger memory chunk and then fill it from the back
        # see: https://github.com/rygorous/ryg_rans/blob/c9d162d996fd600315af9ae8eb89d832576cb32d/main.cpp#L176 for example
        symbol_bitarray = out_bits 

        # core encoding step
        state = self.rans_base_encode_step(s, state)
        if self.params.debug:
            print('encode {}'.format(state))
        return state, symbol_bitarray, M_out

    def encode_block(self, data_block: DataBlock):
        # initialize the output
        encoded_bitarray = BitArray("")
        
        M = Multiset(data_block.data_list)

        # initialize the state 
        state = self.params.INITIAL_STATE
        
        symbol_bitarray = uint_to_bitarray(0,self.params.NUM_BITS_OUT)

        # update the state
        for e,s in enumerate(data_block.data_list):
            state, symbol_bitarray, M = self.encode_symbol(s, state, M,symbol_bitarray)
            if self.params.debug:
                print('iter: {} state: {}'.format(e,state))
            encoded_bitarray = symbol_bitarray + encoded_bitarray
#             print(len(encoded_bitarray))

        # Finally, pre-pend binary representation of the final state
        encoded_bitarray = uint_to_bitarray(state, self.params.NUM_STATE_BITS) + encoded_bitarray

        # add the data_block size at the beginning
        # NOTE: rANS decoding needs a way to indicate where to stop the decoding
        # One way is to add a character at the end which signals EOF. This requires us to
        # change the probabilities of the other symbols. Another way is to just signal the size of the
        # block. These two approaches add a bit of overhead.. the approach we use is much more transparent
        
        encoded_bitarray = (
            uint_to_bitarray(data_block.size, self.params.DATA_BLOCK_SIZE_BITS) + encoded_bitarray
        )
        # encoded_bitarray = bit(data_block.size) || bit(state) || encoded_bitarray
        return encoded_bitarray
    
class rBBCMultiSetDecoder(rANS.rANSDecoder):
    def shrink_state_local(self, state: int) -> Tuple[int]:
        """stream out the lower bits of the state, until the state is below params.max_shrunk_state[next_symbol]"""

        # output bits to the stream to bring the state in the range for the next encoding
        while state > self.params.H:
            state = (state -state%2*self.params.H)>> self.params.NUM_BITS_OUT

        return state
    
    def expand_state(self, state: int, encoded_bitarray: BitArray) -> Tuple[int, int]:
        # remap the state into the acceptable range
        num_bits = 0
        while state < self.params.L:
#             print('num: {:d} state: {:d}'.format(num_bits,state))
            state_remainder = bitarray_to_uint(
                encoded_bitarray[num_bits : num_bits + self.params.NUM_BITS_OUT]
            )
            num_bits += self.params.NUM_BITS_OUT
            state = (state << self.params.NUM_BITS_OUT) + state_remainder
        return state, num_bits
    
    def decode_symbol(self, state: int, encoded_bitarray: BitArray, M_out: Multiset):
        # base rANS decoding step
        s, state = self.rans_base_decode_step(state)
        if self.params.debug:
            print('decode {}'.format(state))
        # remap the state into the acceptable range
        state, num_bits_used_by_expand_state = self.expand_state(state, encoded_bitarray)
        if self.params.debug:
            print('remap {}'.format(state))
        # additional steps in bits back coding
        M_in = M_out + Multiset(s)
        freqs_local = Frequencies(M_in)
        rParams_local = rANS.rANSParams(freqs_local)
        encoder_local = rANS.rANSEncoder(rParams_local)
        state = self.shrink_state_local(state) #questionable
        if self.params.debug:
            print('shrink {}'.format(state))
        state = encoder_local.rans_base_encode_step(s, state)
        if self.params.debug:
            print('encode {}'.format(state))
#         state, _ = self.shrink_state(state,s)
        return s, state, M_in, num_bits_used_by_expand_state

    def decode_block(self, encoded_bitarray: BitArray):
        # get data block size
        data_block_size_bitarray = encoded_bitarray[: self.params.DATA_BLOCK_SIZE_BITS]
        input_data_block_size = bitarray_to_uint(data_block_size_bitarray)
        num_bits_consumed = self.params.DATA_BLOCK_SIZE_BITS
        # print(input_data_block_size)
        
        # get the final state
        state = bitarray_to_uint(
            encoded_bitarray[num_bits_consumed : num_bits_consumed + self.params.NUM_STATE_BITS]
        )
        num_bits_consumed += self.params.NUM_STATE_BITS

        M = Multiset()

        # perform the decoding
        decoded_data_list = []
        for i in range(input_data_block_size):
            if self.params.debug:
                print('iter: {} state: {}'.format(input_data_block_size-1-i,state))
            s, state, M, num_symbol_bits = self.decode_symbol(
                state, encoded_bitarray[num_bits_consumed:], M
            )

            # rANS decoder decodes symbols in the reverse direction,
            # so we add newly decoded symbol at the beginning
            decoded_data_list = [s] + decoded_data_list
            num_bits_consumed += num_symbol_bits

        # Finally, as a sanity check, ensure that the end state should be equal to the initial state
#         assert state == self.params.INITIAL_STATE

        return DataBlock(decoded_data_list), num_bits_consumed

def RelativeEntropy(prob_dist1,prob_dist2):
    alphabet1 = prob_dist1.alphabet
    alphabet1.sort()
    alphabet2 = prob_dist2.alphabet
    alphabet2.sort()
    assert alphabet1==alphabet2
    RelEntropy = 0
    for s in alphabet1:
        p1 = prob_dist1.prob_dict[s]
        p2 = prob_dist1.prob_dict[s]
        RelEntropy+=p1*(np.log(p1)-np.log(p2))
    return RelEntropy


######################################## TESTS ##########################################

def test_BBC_coding():
    debug = False
    freqs_list = [
        Frequencies({"A": 1, "B": 1, "C": 2}),
        Frequencies({"A": 1, "B": 1, "C": 2}),
        Frequencies({"A": 1, "B": 1, "C": 2}),
        Frequencies({"A": 1, "B": 1, "C": 2}),
        Frequencies({"A": 1, "B": 1, "C": 2}),
        Frequencies({"A": 1, "B": 1, "C": 1, "D": 1, "E": 1, "F": 1}),
        Frequencies({"A": 1, "B": 1, "C": 1, "D": 1, "E": 1, "F": 10}),
    ]
    params_list = [
        rBBCParams(freqs_list[0],debug=debug), 
        rBBCParams(freqs_list[1],debug=debug, NUM_BITS_OUT=8),
        rBBCParams(freqs_list[2],debug=debug, RANGE_FACTOR=1 << 8),
        rBBCParams(freqs_list[3],debug=debug, NUM_BITS_OUT=4, RANGE_FACTOR=1 << 12),
        rBBCParams(freqs_list[4],debug=debug, NUM_BITS_OUT=8, RANGE_FACTOR=1 << 8),
        rBBCParams(freqs_list[5],debug=debug, NUM_BITS_OUT=8, RANGE_FACTOR=1 << 8),
        rBBCParams(freqs_list[6],debug=debug, NUM_BITS_OUT=8, RANGE_FACTOR=1 << 8),
    ]
    # generate random data and test if coding is lossless
    DATA_SIZE = 1000
    SEED = 0
    e = 0
    for freq, rans_params in zip(freqs_list, params_list):
        print('Test case {}'.format(e))
        e+=1
        prob_dist = freq.get_prob_dist()
        data_block = get_random_data_block(prob_dist, DATA_SIZE, seed=SEED)
        avg_log_prob = get_avg_neg_log_prob(prob_dist, data_block)
        encoder = rBBCMultiSetEncoder(rans_params)
        decoder = rBBCMultiSetDecoder(rans_params)

        # test encode
        encoded_bitarray = encoder.encode_block(data_block)

        # test decode
        decoded_block, num_bits_consumed = decoder.decode_block(encoded_bitarray)
        prob_dist_decode = decoded_block.get_empirical_distribution()
        prob_dist_raw = data_block.get_empirical_distribution()

        KL1 = RelativeEntropy(prob_dist_raw,prob_dist_decode)
        KL2 = RelativeEntropy(prob_dist_decode,prob_dist_raw)
        print('The KL divergence between raw and decoded is {:.3e}'.format(KL1))
        print('The KL divergence between decoded and raw is {:.3e}'.format(KL2))

    # compare with rANS
    e = 0
    print('Comparison with rANS')
    for freq, rans_params in zip(freqs_list, params_list):
        print('Test case {}'.format(e))
        e+=1
        prob_dist = freq.get_prob_dist()
        data_block = get_random_data_block(prob_dist, DATA_SIZE, seed=SEED)
        avg_log_prob = get_avg_neg_log_prob(prob_dist, data_block)
        encoder1 = rBBCMultiSetEncoder(rans_params)
        encoder2 = rANS.rANSEncoder(rans_params)

        # test encode
        encoded_bitarray1 = encoder1.encode_block(data_block)
        encoded_bitarray2 = encoder2.encode_block(data_block)

        print('rBBC: {}'.format(len(encoded_bitarray1)))
        print('rANS: {}'.format(len(encoded_bitarray2)))

if __name__ == '__main__':
    test_BBC_coding()
