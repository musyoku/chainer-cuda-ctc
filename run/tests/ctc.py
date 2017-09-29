import math, sys, os
import unittest
import numpy as np
import chainer, cupy
import chainer.functions as F
from chainer import cuda, Variable, gradient_check, testing
from chainer.testing import attr
from chainer.testing import condition
sys.path.append(os.path.join("..", ".."))
from functions import cuda_ctc, cupy_ctc
from functions.cuda_ctc import connectionist_temporal_classification, CTCFunction

class CTCTestBase(object):

	def setUp(self):
		self.x = np.random.uniform(-1, 1, (4, 2, 3)).astype(np.float32)
		self.t = np.array([[0, 1], [1, 0]]).astype(np.int32)
		self.l = np.array([[2, 0, 2, 1, 2],
							  [2, 1, 2, 0, 2]]).astype(np.int32)
		self.blank_symbol = 2
		self.x_length = np.full((len(self.x[0]),), len(self.x), dtype='i')
		self.l_length = np.full((len(self.t),), len(self.t[0]), dtype='i')
		self.use_length = True
		if self.reduce == 'mean':
			self.gy = np.random.uniform(-1, 1, ()).astype(np.float32)
		else:
			self.gy = np.random.uniform(-1, 1, (2,)).astype(np.float32)

	# recursive forward computation.
	def alpha(self, x, l, t, u):
		if u < 0:
			return 0.0
		if t == 0:
			if u == 0:
				return x[0][self.blank_symbol]
			elif u == 1:
				return x[0][l[1]]
			else:
				return 0.0
		elif l[u] == self.blank_symbol or l[u] == l[u - 2]:
			return (x[t][l[u]] *
					(self.alpha(x, l, t - 1, u - 1) +
					 self.alpha(x, l, t - 1, u)))
		else:
			return (x[t][l[u]] *
					(self.alpha(x, l, t - 1, u - 2) +
					 self.alpha(x, l, t - 1, u - 1) +
					 self.alpha(x, l, t - 1, u)))

	def check_forward(self, t_data, xs_data, l_length, x_length):
		x = tuple(chainer.Variable(x_data) for x_data in xs_data)
		t = chainer.Variable(t_data)

		args = (x, t, self.blank_symbol)
		if self.use_length:
			args += (chainer.Variable(x_length), chainer.Variable(l_length))
		loss = connectionist_temporal_classification(
			*args, reduce=self.reduce).data

		# compute expected value by recursive computation.
		xp = cuda.get_array_module(self.x)
		xt = xp.swapaxes(self.x, 0, 1)
		for b in range(xt.shape[0]):
			for t in range(xt.shape[1]):
				xt[b][t] = np.exp(xt[b][t]) / np.sum(np.exp(xt[b][t]))
		batchsize = xt.shape[0]
		path_length = 2 * l_length + 1
		loss_expect = xp.zeros((batchsize,), dtype=xp.float32)
		for i in range(batchsize):
			xtb, lb, xlb, plb = xt[i], self.l[i], x_length[i], path_length[i]
			loss_expect[i] = -math.log(
				self.alpha(xtb, lb, int(xlb - 1), int(plb - 1)) +
				self.alpha(xtb, lb, int(xlb - 1), int(plb - 2)))
		if self.reduce == 'mean':
			loss_expect = xp.mean(loss_expect)
		testing.assert_allclose(loss_expect, loss)

	def test_forward_cpu(self):
		self.check_forward(self.t, tuple(self.x),
						   self.l_length, self.x_length)

	@attr.gpu
	def test_forward_gpu(self):
		self.check_forward(cuda.to_gpu(self.t),
						   tuple(cuda.to_gpu(x_data) for x_data in self.x),
						   cuda.to_gpu(self.l_length),
						   cuda.to_gpu(self.x_length))

	# expected value(via numerical differentiation) from t_data
	def check_backward(self, t_data, xs_data, l_length, x_length, gy_data):
		gradient_check.check_backward(
			CTCFunction(
				self.blank_symbol, self.reduce),
			(x_length, l_length, t_data) + xs_data, gy_data,
			eps=1e-2, atol=1e-4)

	@condition.retry(3)
	def test_backward_cpu(self):
		self.check_backward(self.t, tuple(self.x),
							self.l_length, self.x_length,
							self.gy)

	@condition.retry(3)
	@attr.gpu
	def test_backward_gpu(self):
		self.check_backward(cuda.to_gpu(self.t),
							tuple(cuda.to_gpu(x_data) for x_data in self.x),
							cuda.to_gpu(self.l_length),
							cuda.to_gpu(self.x_length),
							cuda.to_gpu(self.gy))


@testing.parameterize(
	{'reduce': 'mean'},
	{'reduce': 'no'}
)
class TestCTC(unittest.TestCase, CTCTestBase):

	def setUp(self):
		CTCTestBase.setUp(self)


@testing.parameterize(
	{'reduce': 'mean'},
	{'reduce': 'no'}
)
class TestCTCWithoutLength(unittest.TestCase, CTCTestBase):

	def setUp(self):
		CTCTestBase.setUp(self)
		self.use_length = False


@testing.parameterize(
	{'reduce': 'mean'},
	{'reduce': 'no'}
)
class TestCTCWithLabelPadding(unittest.TestCase, CTCTestBase):

	def setUp(self):
		CTCTestBase.setUp(self)
		self.l_length[0] = 1


@testing.parameterize(
	{'reduce': 'mean'},
	{'reduce': 'no'}
)
class TestCTCWithInputPadding(unittest.TestCase, CTCTestBase):

	def setUp(self):
		CTCTestBase.setUp(self)
		self.x_length[0] = 3


@testing.parameterize(
	{'reduce': 'mean'},
	{'reduce': 'no'}
)
class TestCTCWithAllPadding(unittest.TestCase, CTCTestBase):

	def setUp(self):
		CTCTestBase.setUp(self)
		self.x_length[...] = 3
		self.l_length[...] = 1


@testing.parameterize(
	{'reduce': 'mean'},
	{'reduce': 'no'}
)
class TestCTCWithRepeatedLabel(unittest.TestCase, CTCTestBase):

	def setUp(self):
		CTCTestBase.setUp(self)
		self.t = np.array([[0, 1, 1], [0, 1, 0]]).astype(np.int32)
		self.l = np.array([[2, 0, 2, 1, 2, 1, 2],
							  [2, 0, 2, 1, 2, 0, 2]]).astype(np.int32)
		self.l_length = np.full((len(self.t),), len(self.t[0]), dtype='i')


@testing.parameterize(
	{'reduce': 'mean'},
	{'reduce': 'no'}
)
class TestCTCBlankSymbol(unittest.TestCase, CTCTestBase):

	def setUp(self):
		CTCTestBase.setUp(self)
		self.x = np.random.uniform(-1, 1, (4, 2, 4)).astype(np.float32)
		self.l = np.array([[3, 0, 3, 1, 3],
							  [3, 1, 3, 0, 3]]).astype(np.int32)
		self.blank_symbol = 3


class TestCTCUseNoBackpropMode(unittest.TestCase):

	def test_no_backprop_mode(self):
		xs_data = np.random.uniform(-1, 1, (4, 2, 3)).astype(np.float32)
		t_data = np.array([[0, 1], [1, 0]]).astype(np.int32)
		with chainer.no_backprop_mode():
			x = [chainer.Variable(x_data) for x_data in xs_data]
			t = chainer.Variable(t_data)
			connectionist_temporal_classification(x, t, 2)


class TestCTCError(unittest.TestCase):

	def test_not_iterable(self):
		x = chainer.Variable(np.zeros((4, 2, 3), np.float32))
		t = chainer.Variable(np.zeros((2, 2), np.int32))
		with self.assertRaises(TypeError):
			connectionist_temporal_classification(x, t, 0)


class TestCTCInvalidReductionOption(unittest.TestCase):

	def test_not_iterable(self):
		x = chainer.Variable(np.zeros((4, 2, 3), np.float32))
		t = chainer.Variable(np.zeros((2, 2), np.int32))
		with self.assertRaises(ValueError):
			connectionist_temporal_classification(
				tuple(x), t, 0, reduce='invalid_option')

def test_forward(batchsize, label_length, seq_length, vocab_size, total_labels_to_fill, repeat=3):
	xp = cupy
	label_unigram = xp.random.randint(1, total_labels_to_fill, size=(batchsize, label_length)).astype(xp.int32)

	num_transitions_to_same_label = xp.count_nonzero(label_unigram == xp.roll(label_unigram, 1, axis=1))
	assert seq_length >= label_length + num_transitions_to_same_label + 1

	length_unigram = xp.full((batchsize,), label_length, dtype=np.int32)
	blank_symbol = 0

	x_data = xp.random.normal(0, 1, size=batchsize*vocab_size*seq_length).reshape((batchsize, vocab_size, seq_length)).astype(xp.float32)

	x = Variable(x_data)
	out_data = F.swapaxes(x, 1, 2)
	out_data = F.reshape(out_data, (batchsize, -1))
	out_data = F.split_axis(out_data, seq_length, axis=1)

	x_length = Variable(xp.full((batchsize,), seq_length, dtype=np.int32))

	loss_cuda = cuda_ctc.connectionist_temporal_classification(out_data, label_unigram, blank_symbol, x_length, Variable(length_unigram), reduce="mean")
	loss_cupy = cupy_ctc.connectionist_temporal_classification(out_data, label_unigram, blank_symbol, x_length, Variable(length_unigram), reduce="mean")

	error_forward = abs(float(loss_cupy.data) - float(loss_cuda.data))

	assert error_forward < 5e-4, "error={}, batchsize={}, label_length={}, seq_length={}, vocab={}, labels={}, loss_cupy={}, loss_cuda={}".format(error_forward, 
		batchsize, label_length, seq_length, vocab_size, total_labels_to_fill, loss_cupy.data, loss_cuda.data)

	x.cleargrad()
	loss_cuda.backward()
	grad_cuda = x.grad.copy()
	loss_cupy.backward()
	grad_cupy = x.grad.copy()

	error_backward = float(xp.mean(abs(grad_cupy - grad_cuda)))

	assert error_backward < 5e-3, "error={}, batchsize={}, label_length={}, seq_length={}, vocab={}, labels={}, loss_cupy={}, loss_cuda={}".format(error_backward, 
		batchsize, label_length, seq_length, vocab_size, total_labels_to_fill, loss_cupy.data, loss_cuda.data)

	return error_forward, error_backward

def test_recurrence_relation(batchsize, label_length, total_labels_to_fill):
	xp = cupy
	label = xp.random.randint(1, total_labels_to_fill, size=(batchsize, label_length)).astype(xp.int32)
	flags_true = (label != xp.take(label, xp.arange(-1, label_length - 1) % label_length + xp.arange(0, batchsize * label_length, label_length)[:, None]))
	flags_roll = (label != xp.roll(label, 1, axis=1))

	if xp.sum(flags_true.astype(xp.int32) - flags_roll.astype(xp.int32)) != 0:
		print(label)
		print(flags_true)
		print(flags_roll)
	assert xp.sum(flags_true.astype(xp.int32) - flags_roll.astype(xp.int32)) == 0

def main():
	np.set_printoptions(linewidth=200)
	np.random.seed(0)
	cupy.random.seed(0)
	batchsize_list = [16, 32]
	label_length_list = [10, 30, 50]
	vocab_size_list = [100, 500, 1000]

	error = test_forward(16, 10, 30, 40, 40)

	for batchsize in batchsize_list:
		for label_length in label_length_list:
			for vocab_size in vocab_size_list:
				total_labels_list = [vocab_size // div for div in [2, 5, 10, 50]]
				for total_labels in total_labels_list:
					seq_length_list = [label_length * mul for mul in [3, 4, 5]]
					for seq_length in seq_length_list:
						total_labels = vocab_size
						test_recurrence_relation(batchsize, label_length, total_labels)
						error = test_forward(batchsize, label_length, seq_length, vocab_size, total_labels)
						print("batchsize={}, label_length={}, seq_length={}, vocab={}, labels={}".format(batchsize, label_length, seq_length, vocab_size, total_labels), "OK", error)

	testing.run_module(__name__, __file__)

if __name__ == "__main__":
	main()