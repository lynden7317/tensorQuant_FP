from Quantize import FixedPoint

class Quantizer_if():
    """Interface for quantizer classes"""
    def quantize(self,tensor):
        raise NotImplementedError

class FixedPointQuantizer_zero(Quantizer_if):
    """Fixed point quantization with fixed_size bits and fixed_prec fractional bits.
       Uses c-kernel for quantization. 
    """
    def __init__(self, fixed_size, fixed_prec):
        self.fixed_size=fixed_size
        self.fixed_prec=fixed_prec
    def quantize(self,tensor):
        return FixedPoint.round_zero(tensor,self.fixed_size,self.fixed_prec)

class FixedPointQuantizer_down(Quantizer_if):
    """Fixed point quantization with fixed_size bits and fixed_prec fractional bits.
       Uses c-kernel for quantization. 
    """
    def __init__(self, fixed_size, fixed_prec):
        self.fixed_size=fixed_size
        self.fixed_prec=fixed_prec
    def quantize(self,tensor):
        return FixedPoint.round_down(tensor,self.fixed_size,self.fixed_prec)

class FixedPointQuantizer_nearest(Quantizer_if):
    """Fixed point quantization with fixed_size bits and fixed_prec fractional bits.
       Uses c-kernel for quantization. 
    """
    def __init__(self, fixed_size, fixed_prec):
        self.fixed_size=fixed_size
        self.fixed_prec=fixed_prec
    def quantize(self,tensor):
        return FixedPoint.round_nearest(tensor,self.fixed_size,self.fixed_prec)

class FixedPointQuantizer_stochastic(Quantizer_if):
    """Fixed point quantization with fixed_size bits and fixed_prec fractional bits.
       Uses c-kernel for quantization. 
    """
    def __init__(self, fixed_size, fixed_prec):
        self.fixed_size=fixed_size
        self.fixed_prec=fixed_prec
    def quantize(self,tensor):
        return FixedPoint.round_stochastic(tensor,self.fixed_size,self.fixed_prec)

class FixedPointQuantizer_floating(Quantizer_if):
    """Floating point quantization with exponent bits and mantissa bits.
       Uses c-kernel for quantization.
    """
    def __init__(self, exp_bit, mant_bit):
        self.exp_bit=exp_bit
        self.mant_bit=mant_bit
    def quantize(self,tensor):
        return FixedPoint.round_floating(tensor,self.exp_bit,self.mant_bit)

class NoQuantizer(Quantizer_if):
    """Applies no quantization to the tensor"""
    def quantize(self,tensor):
        return tensor
