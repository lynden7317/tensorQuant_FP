#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include<cmath>

#include<iostream>
#include<mutex>
using namespace std;
std::mutex mu;

using namespace tensorflow;

REGISTER_OP("RoundFloat")
    .Attr("exp_bit: int")
    .Attr("mant_bit: int")
    .Attr("T: {float}")
    .Input("to_reshape: T")
    .Output("reshaped: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });


union FP754_32 {
    struct {
        unsigned int mantissa : 23;
		unsigned int exponent : 8;
		unsigned int sign : 1;
    } raw;
	float f;
} fp32;		

template<typename T>
class RoundOp : public OpKernel {
 public:
  explicit RoundOp(OpKernelConstruction* context) : OpKernel(context) {
    // Get Attributes
    OP_REQUIRES_OK(context,
                   context->GetAttr("exp_bit", &exp_bit));
    OP_REQUIRES_OK(context,
                   context->GetAttr("mant_bit", &mant_bit));
    // Check Attributes
    OP_REQUIRES(context, exp_bit > 0,
                errors::InvalidArgument("exp_bit needs to be bigger than 0, got ",
                                        exp_bit));
    OP_REQUIRES(context, mant_bit > 0,
                errors::InvalidArgument("mant_bit needs to be bigger than 0, got ",
                                        mant_bit));

    }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<T>();

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output = output_tensor->flat<T>();

    // truncate every element of the tensor to fixed point
    const int N = input.size();
    // const T fixed_max_signed = ((T)(1UL<<(fixed_size-1))-1)/(1UL<<fixed_prec);
    // const T fixed_min_signed = -(1L<<(fixed_size-fixed_prec-1));

    for (int i = 0; i < N; i++) {

		// rounds with user defined floating point
		mu.lock();
		
		fp32.f = input(i);
		int newExp = fp32.raw.exponent - 127;
		unsigned int newMant = fp32.raw.mantissa >> (23-mant_bit);
	  
		//cout << " input(i) = " << input(i) << " fp32.f = " << fp32.f << endl;
		//cout << "fp32.raw.sign = " << fp32.raw.sign << " exponent = " << fp32.raw.exponent << endl;
		//cout << "mantissa = " << fp32.raw.mantissa << endl;
        //cout << "newExp = " << newExp << endl;

		float qOut = 1.0;
		if(fp32.f == 0.0f)
		{
			qOut = 0.0;
		}
		else if(fp32.f == -0.0f)
		{
			qOut = 0.0;
		}
		else
		{
			int based = (1<<(exp_bit-1))-1;
			//cout << "newExp = " << newExp << " based = " << (1UL<<(exp_bit-1))-1 << endl;
			//cout << "test = " << newExp+(1UL<<(exp_bit-1))-1 << endl;
			// Maximum of representation
			if(newExp > based)
			{
				//cout << "Over" << endl;
				newExp = (1<<(exp_bit-1))-1;
				for(unsigned int rs=1; rs <= mant_bit; rs++)
					qOut += 1.0/(1<<rs);
			
			}
			// Minimum of representation
			else if(newExp+(mant_bit-1) < 0-based)//(newExp+based < 0-based)
			{
				qOut = 0.0;
			}
			// Normal type include Denormalization
			else
			{
				// Denormalization
				if(newExp+based <= 0)
				{
					newMant = newMant+(1<<mant_bit);
					qOut = newMant >> (1-based-newExp);
					newExp = 1-based-mant_bit;
				}
				else
				{
					for(unsigned int rs=1; rs <= mant_bit; rs++)
					{
						if(((newMant>>(mant_bit-rs)) & 1UL))
							qOut += 1.0/(1UL<<rs);
					}
				//cout << "qOut = " << qOut << endl;
				}
			}
		
			if(fp32.raw.sign == 1)
			{
				if(newExp > 0)
					qOut = qOut*-1*(1<<newExp);
				else
					qOut = qOut*-1*(1.0/(1<<(newExp*-1)));
			}
			else
			{
				if(newExp > 0)
					qOut = qOut*(1<<newExp);
				else
					qOut = qOut*(1.0/(1<<(newExp*-1)));
			}
		}	
		// T fixed_number = round(input(i)*(1UL<<fixed_prec)) / (1UL<<fixed_prec);
		// fixed_number = std::max(std::min(fixed_number,fixed_max_signed), fixed_min_signed);
        //cout << " qOut = " << qOut << endl;
		/*
		if(input(i)-qOut > 0.00000001)
		{
			//mu.lock();
			cout << "fp32.raw.sign = " << fp32.raw.sign << endl;
			cout << "input(i) = " << input(i) << " qOut = " << qOut << endl;
			//mu.unlock();
		}
		*/
		//cout << "input=" << input(i) << endl;
		output(i) = qOut;
		//cout << "output= " << output(i) << endl;
		//cout << "===========================" << endl;
        mu.unlock();
    }
  }

 private:
    int exp_bit;
    int mant_bit;
	/*
    union FP754_32 {
	struct {
        unsigned int mantissa : 23;
	    unsigned int exponent : 8;
	    unsigned int sign : 1;
	} raw;
	T f;
    } fp32;
	*/
};

REGISTER_KERNEL_BUILDER(
    Name("RoundFloat")
    .Device(DEVICE_CPU)
    .TypeConstraint<float>("T"),
    RoundOp<float>);
