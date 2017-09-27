import sys, os, chainer, cupy
import numpy as np
from chainer import Variable, cuda
import chainer.functions as F
sys.path.append(os.path.join("..", ".."))
from functions import cupy_ctc, cuda_ctc

gpu_device = 2
cuda.get_device(gpu_device).use()

def test_forward():
	np.set_printoptions(linewidth=200, precision=2)
	xp = cupy
	# np.random.seed(0)
	label_unigram = xp.asarray([
		[1, 2, 2, 3, 5],
		[2, 4, 3, 0, 0],
	], dtype=xp.int32)
	blank_symbol = 0

	length_unigram = xp.asarray([5, 3], dtype=xp.int32)
	path_length = length_unigram * 2 + 1

	vocab_size = 6
	seq_length = 20
	batchsize = 2
	x = xp.random.normal(0, 1, size=batchsize*vocab_size*seq_length).reshape((batchsize, vocab_size, seq_length)).astype(xp.float32)

	in_data = Variable(x)
	in_data = F.swapaxes(in_data, 1, 2)
	in_data = F.reshape(in_data, (batchsize, -1))
	in_data = F.split_axis(in_data, seq_length, axis=1)

	x_length = Variable(xp.asarray([seq_length, seq_length // 2], dtype=xp.int32))

	loss_ctc = cuda_ctc.connectionist_temporal_classification(in_data, label_unigram, blank_symbol, x_length, Variable(length_unigram), reduce="mean")

	print(loss_ctc)

if __name__ == "__main__":
	test_forward()