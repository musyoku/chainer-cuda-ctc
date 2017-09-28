# coding: utf-8
from __future__ import division
import collections, cupy, six, chainer, math, time
import numpy as np
from chainer import cuda
from chainer import function
from chainer import utils
from chainer.utils import type_check
from chainer import variable

CUDA_CTC_KERNEL = """
extern "C" 
{
	__global__ 
	void forward_probability(
			const float* __restrict__ y_ptr, 				// 各ラベルの確率
			const int* __restrict__ unit_index_ptr, 		// パスの各ノードが示すラベルID
			const float* __restrict__ prev_forward_prob_ptr, 	// 1つ前の時刻での各ノードへ到達する前向き確率
			const float* __restrict__ recurrence_relation_ptr, 	// 接続関係
			float* __restrict__ unit_prob_ptr, 				// 各時刻の各ノードへ到達する前向き確率
			float* __restrict__ next_forward_prob_ptr,
			const int batchsize, 
			const int* __restrict__ path_length_ptr, 
			const int max_path_length)
	{
		int column = blockIdx.x * blockDim.x + threadIdx.x;	// 0 <= column < batchsize * max_path_length
		int total_columns = batchsize * max_path_length;
		if(column >= total_columns) return;
		int batch_index = column / max_path_length;
		int path_pos = column % max_path_length;			// パスのノードの位置
		int path_length = *(path_length_ptr + batch_index);

		float* node_ptr = next_forward_prob_ptr + column;
		*node_ptr = -10000000000;
		for(int s = 0;s < path_length;s++){
			const float connection = *(recurrence_relation_ptr + batch_index * max_path_length * max_path_length + path_pos * max_path_length + s);
			const float prev_forward_prob = *(prev_forward_prob_ptr + batch_index * max_path_length + s);
			const float prob = connection + prev_forward_prob;
			// logsumexp
			float max_value = prob;
			float min_value = *node_ptr;
			if (min_value > max_value) {
				max_value = *node_ptr;
				min_value = prob;
			}
			*node_ptr = max_value + logf(1 + expf(min_value - max_value));
		}
		int unit_index = *(unit_index_ptr + column);
		*node_ptr += y_ptr[unit_index];
	}

	__global__ 
	void backward_probability(
			float* __restrict__ prev_backward_prob_ptr, 		// 1つ前の時刻での各ノードへ到達する前向き確率
			const float* __restrict__ recurrence_relation_ptr, 	// 接続関係
			float* __restrict__ next_backward_prob_ptr,
			const int* __restrict__ prob_index_ptr,
			float* __restrict__ prob_ptr,
			const float* __restrict__ y_inv_ptr,
			const int* __restrict__ r_index_ptr,
			const int batchsize, 
			const int* __restrict__ path_length_ptr, 
			const int max_path_length,
			const int t)
	{
		int column = blockIdx.x * blockDim.x + threadIdx.x;	// 0 <= column < batchsize * max_path_length
		int total_columns = batchsize * max_path_length;
		if(column >= total_columns) return;
		int batch_index = column / max_path_length;
		int path_pos = column % max_path_length;			// パスのノードの位置
		int path_length = *(path_length_ptr + batch_index);

		float* node_ptr = next_backward_prob_ptr + column;
		*node_ptr = -10000000000;
		for(int s = 0;s < path_length;s++){
			const float connection = *(recurrence_relation_ptr + batch_index * max_path_length * max_path_length + path_pos * max_path_length + s);
			const float prev_forward_prob = *(prev_backward_prob_ptr + batch_index * max_path_length + s);
			const float prob = connection + prev_forward_prob;
			// logsumexp
			float max_value = prob;
			float min_value = *node_ptr;
			if (min_value > max_value) {
				max_value = *node_ptr;
				min_value = prob;
			}
			*node_ptr = max_value + logf(1 + expf(min_value - max_value));
		}
		float* prob_t_ptr = prob_ptr + t * batchsize * max_path_length + column;
		const int prob_index = *(prob_index_ptr + column);

		// xp.take(backward_prob[:, ::-1], backward_prob_index)
		*prob_t_ptr += *(next_backward_prob_ptr + (batch_index + 1) * max_path_length - prob_index % max_path_length - 1);

		int r_index = *(r_index_ptr + column);
		*(prev_backward_prob_ptr + column) = *(next_backward_prob_ptr + column) + y_inv_ptr[r_index];

	}
}
"""

def _logsumexp(a, xp, axis=None):
	vmax = xp.amax(a, axis=axis, keepdims=True)
	vmax += xp.log(xp.sum(xp.exp(a - vmax), axis=axis, keepdims=True, dtype=a.dtype))
	return xp.squeeze(vmax, axis=axis)


def _softmax(x, xp):
	val = xp.exp(x - xp.amax(x, axis=2, keepdims=True))
	val /= xp.sum(val, axis=2, keepdims=True)
	return val


def _label_to_path(labels, blank_symbol, xp):
	path = xp.full((len(labels), labels.shape[1] * 2 + 1),
				   blank_symbol, dtype=np.int32)
	path[:, 1::2] = labels
	return path


def _log_dot(prob, rr, xp):
	return _logsumexp(prob + rr, xp, axis=2)


def _move_label_to_back(path, path_length, xp):
	s1 = path.shape[1]  # TODO(okuta): Change name
	index = (xp.arange(0, path.size, s1, dtype=np.int32)[:, None] + (xp.arange(s1) + path_length[:, None])[:, ::-1] % s1)
	return xp.take(path, index)


def _move_inputs(prob, input_length, xp):
	seq, batch, ch = prob.shape
	rotate = (xp.arange(seq)[:, None] + input_length) % seq
	index = rotate * batch + xp.arange(batch)
	return xp.take(prob.reshape(seq * batch, ch), index, axis=0)


options = ["-ftz=true"]
nvcc = None
cupy_version = 0
if hasattr(cupy.cuda.compiler, "compile_using_nvrtc"):	# CuPy v2
	nvcc = cupy.cuda.compiler.compile_using_nvrtc
	cupy_version = 2
elif hasattr(cupy.cuda.compiler, "nvcc"):				# CuPy v1
	nvcc = cupy.cuda.compiler.nvcc
	cupy_version = 1
else:
	raise NotImplementedError()

CUDA_CTC_PTX = nvcc(CUDA_CTC_KERNEL, options, None)

def _as_contiguous(args):
	if isinstance(args, (list, tuple)):
		ret = []
		for arg in args:
			if arg is None:
				ret.append(None)
				continue
			if arg.flags.c_contiguous is False:
				xp = cuda.get_array_module(arg)
				arg = xp.ascontiguousarray(arg)
			ret.append(arg)
		return ret

	if args.flags.c_contiguous is False:
		xp = cuda.get_array_module(args)
		args = xp.ascontiguousarray(args)

	return args

class CTCFunction(function.Function):
	_cuda_module = None

	def _cuda_elementwise(self, name, args, block, grid):
		func = self._cuda_get_function(name)
		func(args=args, block=block, grid=grid)

	def _cuda_get_function(self, name):
		module = self._cuda_get_module()
		return module.get_function(name)

	def _cuda_get_module(self):
		if CTCFunction._cuda_module is not None:
			return CTCFunction._cuda_module

		CTCFunction._cuda_module = cupy.cuda.function.Module()
		
		if cupy_version == 1:
			CTCFunction._cuda_module.load(CUDA_CTC_PTX)
			return CTCFunction._cuda_module

		if cupy_version == 2:
			ls = function.LinkState()
			ls.add_ptr_data(CUDA_CTC_PTX, u"cupy.ptx")
			CTCFunction._cuda_module.load(ls.complete())
			return CTCFunction._cuda_module

		raise NotImplementedError()

	def __init__(self, blank_symbol, reduce='mean'):
		self.blank_symbol = blank_symbol
		self.zero_padding = -100.0

		if reduce not in ('mean', 'no'):
			raise ValueError(
				"only 'mean' and 'no' are valid "
				"for 'reduce', but '%s' is given" % reduce)
		self.reduce = reduce

	def check_type_forward(self, in_types):
		type_check.expect(in_types.size() > 3)  # TODO(okuta): > 3?
		l_type = in_types[2]
		type_check.expect(l_type.dtype == np.int32)

		x_basetype = in_types[3]  # TODO(oktua): Check x_basetype size

		for i in six.moves.range(3, len(in_types)):
			x_type = in_types[i]
			type_check.expect(
				x_type.dtype == np.float32,
				x_type.shape == x_basetype.shape,
			)

	def log_matrix(self, x, xp):
		if xp == np:
			res = np.ma.log(x).filled(fill_value=self.zero_padding)
		else:
			create_recurrence_relation = cuda.cupy.ElementwiseKernel(
				'T x, T e', 'T y',
				'y = x == 0 ? e : log(x)',
				'create_recurrence_relation')
			res = create_recurrence_relation(x, self.zero_padding)
		return res.astype(np.float32)

	def recurrence_relation(self, label, path_length, max_length, dtype, xp):
		batch, lab = label.shape
		repeat_mask = xp.ones((batch, lab * 2 + 1))
		repeat_mask[:, 1::2] = (label !=
								xp.take(label, xp.arange(-1, lab - 1)
										% lab + xp.arange(0, batch * lab,
														  lab)[:, None]))
		repeat_mask[:, 1] = 1
		rr = (xp.eye(max_length, dtype=dtype)[None, :] +
			  xp.eye(max_length, k=1, dtype=dtype)[None, :] +
			  (xp.eye(max_length, k=2, dtype=dtype) *
			   (xp.arange(max_length, dtype=dtype) % dtype(2))[None, :]
			   * repeat_mask[:, None]))
		return self.log_matrix(rr * (path_length[:, None] > xp.arange(max_length))[..., None], xp).swapaxes(1, 2)

	# path probablity to label probability
	def label_probability(self, label_size, path, path_length, multiply_seq, xp):
		labels_prob = self.log_matrix(xp.zeros((len(path), label_size),
											   dtype=multiply_seq.dtype), xp)
		ret = xp.empty(
			(len(multiply_seq),) + labels_prob.shape, dtype=labels_prob.dtype)
		ret[...] = labels_prob
		if xp == np:
			for b in six.moves.range(len(path)):
				target_path = path[b][0:path_length[b]]
				chars = {c for c in target_path}
				for c in chars:
					ret[:, b, c] = _logsumexp(
						multiply_seq[:, b, 0:path_length[b]]
						[:, target_path == c], np, axis=1)
		else:
			for i, multiply in enumerate(multiply_seq):
				# TODO(okuta): remove loop
				cuda.cupy.ElementwiseKernel(
					'raw T x, raw I y, raw I l, I b_max, I c_max',
					'T z',
					'''
					T value = z;
					I c = i % b_max, b = i / b_max;
					int ind[2] = {b, -1};
					for (int index = 0; index < c_max; ++index) {
						ind[1] = index;
						if (ind[1] < l[ind[0]] && y[ind] == c) {
							T xvalue = x[ind];
							T at = xvalue, bt = value;
							if (value > xvalue) {
								at = value;
								bt = xvalue;
							}
							value = at + log(1 + exp(bt - at));
						}
					}
					z = value;
					''',
					'reduce_probability')(multiply, path, path_length,
										  labels_prob.shape[1],
										  path.shape[1], ret[i])
		return ret

	def calc_trans(self, yseq, input_length, label, label_length, path, path_length, xp):
		max_path_length = path.shape[1]
		batchsize = yseq.shape[1]

		forward_prob = self.log_matrix(xp.eye(path.shape[1], dtype='f')[0], xp)[None, :]
		forward_prob = xp.repeat(forward_prob, batchsize, axis=0)
		backward_prob = forward_prob
		offset = xp.arange(0, yseq[0].size, yseq[0].shape[1], dtype=path.dtype)[:, None]

		# prob[i] := forward[i] + backward[-i-1]
		unit_index = offset + path
		frr = self.recurrence_relation(label, path_length, path.shape[1], np.float32, xp)
		prob = xp.empty((len(yseq),) + unit_index.shape, dtype=forward_prob.dtype)

		total_columns = max_path_length * batchsize
		thread_per_block = min(512, total_columns)
		num_block = math.ceil(total_columns / thread_per_block)
		assert thread_per_block * num_block >= total_columns

		# forward computation.
		frr = _as_contiguous(frr)
		yseq = _as_contiguous(yseq)
		forward_prob = _as_contiguous(forward_prob)
		next_forward_prob = xp.zeros_like(forward_prob)

		if xp is np:
			for i, y in enumerate(yseq):
				# calc forward probability in log scale
				forward_prob = xp.take(y, unit_index) + _log_dot(forward_prob[:, None, :], frr, xp)
				prob[i] = forward_prob
		else:
			cuda_func_forward_probability = self._cuda_get_function("forward_probability")
			for i, y in enumerate(yseq):
				# calc forward probability in log scale
				y = _as_contiguous(y)
				# print("y")
				# print(y)
				# print("take y")
				# print(xp.take(y, unit_index))
				# start_time = time.time()
				# _forward_prob = xp.take(y, unit_index) + _log_dot(forward_prob[:, None, :], frr, xp)
				# print(time.time() - start_time)

				# take = xp.take(y, unit_index)
				# _take = xp.zeros_like(take)

				# print("unit_index")
				# print(unit_index)
				# print("frr")
				# print(frr)
				# print("_forward_prob")
				# print(_forward_prob)

				# start_time = time.time()
				cuda_func_forward_probability(
					args=[
						y.data.ptr,
						unit_index.data.ptr,
						forward_prob.data.ptr,
						frr.data.ptr,
						prob.data.ptr,
						next_forward_prob.data.ptr,
						batchsize,
						path_length.data.ptr,
						max_path_length,
					], 
					block=(thread_per_block, 1, 1), 
					grid=(num_block, 1, 1))
				# print(time.time() - start_time)
				# print("next_forward_prob")
				# print(next_forward_prob)

				forward_prob = next_forward_prob.copy()
				prob[i] = forward_prob

		r_index = offset + _move_label_to_back(path, path_length, xp)

		# rotate yseq with path_length
		yseq_inv = _move_inputs(yseq, input_length, xp)[::-1]
		brr = self.recurrence_relation(_move_label_to_back(label, label_length, xp), path_length, path.shape[1], np.float32, xp)
		# move to back.
		prob = _as_contiguous(_move_inputs(prob, input_length, xp))
		# print("prob")
		# print(prob)

		# backward computation.
		ps1 = path.shape[1]
		backward_prob_index = (xp.arange(0, path.size, ps1, dtype=np.int32)[:, None] + (xp.arange(ps1) - path_length[:, None]) % ps1).astype(np.int32)
		backward_prob_index = _as_contiguous(backward_prob_index)
		# print("unit_index")
		# print(unit_index)
		# print("backward_prob_index")
		# print(backward_prob_index)

		brr = _as_contiguous(brr)
		# print("brr")
		# print(brr)
		yseq = _as_contiguous(yseq)
		yseq_inv = _as_contiguous(yseq_inv)
		next_backward_prob = xp.zeros_like(backward_prob)
		# print(yseq_inv.shape)
		if xp is np:
			for i, y_inv in enumerate(yseq_inv):
				# calc backward probability
				backward_prob = _log_dot(backward_prob[:, None, :], brr, xp)
				prob[-i - 1] += xp.take(backward_prob[:, ::-1], backward_prob_index)
				backward_prob = xp.take(y_inv, r_index) + backward_prob
		else:
			cuda_func_backward_probability = self._cuda_get_function("backward_probability")
			for i, y_inv in enumerate(yseq_inv):
				t = yseq.shape[0] - 1 - i
				y_inv = _as_contiguous(y_inv)
				# calc backward probability
				# print("backward_prob")
				# print(backward_prob)
				# _backward_prob = _log_dot(backward_prob[:, None, :], brr, xp)
				# print("_log_dot")
				# print(_backward_prob)
				# print("take_bp")
				# print(xp.take(backward_prob[:, ::-1], backward_prob_index))
				# print("inv")
				# print(backward_prob[:, ::-1])
				# prob[-i - 1] += xp.take(backward_prob[:, ::-1], backward_prob_index)
				# print("prob")
				# print(prob[-i - 1] + xp.take(backward_prob[:, ::-1], backward_prob_index))
				# print("take_y")
				# print(xp.take(y_inv, r_index))
				# backward_prob = xp.take(y_inv, r_index) + backward_prob
				# print("backward_prob")
				# print(xp.take(y_inv, r_index) + backward_prob)
				# print("y_inv")
				# print(y_inv)
				# print("r_index")
				# print(r_index)

				# _backward_prob = _log_dot(backward_prob[:, None, :], brr, xp)
				# print(xp.take(_backward_prob[:, ::-1], backward_prob_index))
				# print("log_dot")
				# print(_backward_prob)
				# print("prob[-i-1")
				# print(prob[-i - 1] + xp.take(_backward_prob[:, ::-1], backward_prob_index))


				# _backward_prob = _log_dot(backward_prob[:, None, :], brr, xp)
				# _backward_prob = xp.take(y_inv, r_index) + _backward_prob
				# print(_backward_prob)
				
				cuda_func_backward_probability(
					args=[
						backward_prob.data.ptr,
						brr.data.ptr,
						next_backward_prob.data.ptr,
						backward_prob_index.data.ptr,
						prob.data.ptr,
						y_inv.data.ptr,
						r_index.data.ptr,
						batchsize,
						path_length.data.ptr,
						max_path_length,
						t,
					], 
					block=(thread_per_block, 1, 1), 
					grid=(num_block, 1, 1))

				# _backward_prob = xp.take(y_inv, r_index) + _backward_prob

				# print("next_backward_prob")
				# print(next_backward_prob)
				# print(prob[t])

				# raise Exception()

				# print("next_backward_prob")
				# print(next_backward_prob)
				# print("prob")
				# print(prob[t])
				# print("backward_prob")
				# print(backward_prob)
				# print("done")
				# print(backward_prob)
				# raise Exception()
				# raise Exception()

		# move to front.
		return _move_inputs(prob, -self.input_length, xp)

	def forward(self, inputs):
		xp = cuda.get_array_module(inputs[0])
		self.input_length = inputs[0]
		label_length = inputs[1]
		t = inputs[2]
		xs = inputs[3:]

		if chainer.is_debug():
			# Batch size check.
			assert len(xs[0]) == len(t)
			assert len(xs[0]) == len(self.input_length)
			assert len(xs[0]) == len(label_length)

			# Length check.
			assert len(xs) >= xp.max(self.input_length)
			assert len(t[0]) >= xp.max(label_length)

		self.path_length = 2 * label_length + 1

		yseq_shape = (len(xs),) + xs[0].shape
		stack = xp.vstack(xs).reshape(yseq_shape)
		self.yseq = _softmax(stack, xp)
		log_yseq = self.log_matrix(self.yseq, xp)
		self.path = _label_to_path(t, self.blank_symbol, xp)
		self.prob_trans = self.calc_trans(log_yseq, self.input_length, t, label_length, self.path, self.path_length, xp)

		loss = -_logsumexp(self.prob_trans[0], xp, axis=1)
		if self.reduce == 'mean':
			loss = utils.force_array(xp.mean(loss))
		return loss,

	def backward(self, inputs, grad_output):
		xp = cuda.get_array_module(inputs[0])
		batch_size = len(inputs[2])

		total_probability = _logsumexp(self.prob_trans[0], xp, axis=1)
		label_prob = self.label_probability(self.yseq.shape[2], self.path, self.path_length, self.prob_trans, xp)
		self.yseq -= xp.exp(label_prob - total_probability[:, None])
		if self.reduce == 'mean':
			self.yseq *= grad_output[0] / batch_size
		else:
			self.yseq *= grad_output[0][..., None]
		# mask
		self.yseq *= (
			xp.arange(len(self.yseq))[:, None] < self.input_length)[..., None]
		return (None, None, None) + tuple([y for y in self.yseq])


def connectionist_temporal_classification(x, t, blank_symbol, input_length=None, label_length=None, reduce='mean'):
	if not isinstance(x, collections.Sequence):
		raise TypeError('x must be a list of Variables')
	if not isinstance(blank_symbol, int):
		raise TypeError('blank_symbol must be non-negative integer.')
	assert blank_symbol >= 0
	assert blank_symbol < x[0].shape[1]
	# This implementation only supports 1-dimensional data.
	# TODO(jnishi): Support d(>1)-dimentinal inputs.
	assert(len(x[0].shape) == 2)

	if input_length is None:
		xp = cuda.get_array_module(x[0].data)
		input_length = variable.Variable(xp.full((len(x[0].data),), len(x), dtype=np.int32))
		label_length = variable.Variable(xp.full((len(t.data),), len(t.data[0]), dtype=np.int32))

	return CTCFunction(blank_symbol, reduce)(input_length, label_length, t, *x)