import sys, os, chainer, cupy, time
import seaborn as sns
import numpy as np
import pandas as pd
from chainer import Variable, cuda
import chainer.functions as F
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
sys.path.append(os.path.join("..", ".."))
from functions import cupy_ctc, cuda_ctc

gpu_device = 0
cuda.get_device(gpu_device).use()
xp = cupy

def benchmark_cupy_ctc(batchsize, label_length, seq_length, vocab_size, repeat=50):
	label_unigram = xp.random.randint(1, vocab_size, size=(batchsize, label_length)).astype(xp.int32)
	length_unigram = xp.full((batchsize,), label_length, dtype=np.int32)
	blank_symbol = 0

	x = xp.random.normal(0, 1, size=batchsize*vocab_size*seq_length).reshape((batchsize, vocab_size, seq_length)).astype(xp.float32)

	in_data = Variable(x)
	in_data = F.swapaxes(in_data, 1, 2)
	in_data = F.reshape(in_data, (batchsize, -1))
	in_data = F.split_axis(in_data, seq_length, axis=1)

	x_length = Variable(xp.full((batchsize,), seq_length, dtype=np.int32))

	start_time = time.time()
	for i in range(repeat):
		loss_ctc = cupy_ctc.connectionist_temporal_classification(in_data, label_unigram, blank_symbol, x_length, Variable(length_unigram), reduce="mean")
	forward_time_mean = (time.time() - start_time) / repeat

	start_time = time.time()
	for i in range(repeat):
		loss_ctc = cupy_ctc.connectionist_temporal_classification(in_data, label_unigram, blank_symbol, x_length, Variable(length_unigram), reduce="mean")
		loss_ctc.backward()
	backward_time_mean = (time.time() - start_time) / repeat

	return forward_time_mean, backward_time_mean

def benchmark_cuda_ctc(batchsize, label_length, seq_length, vocab_size, repeat=50):
	label_unigram = xp.random.randint(1, vocab_size, size=(batchsize, label_length)).astype(xp.int32)
	length_unigram = xp.full((batchsize,), label_length, dtype=np.int32)
	blank_symbol = 0

	x = xp.random.normal(0, 1, size=batchsize*vocab_size*seq_length).reshape((batchsize, vocab_size, seq_length)).astype(xp.float32)

	in_data = Variable(x)
	in_data = F.swapaxes(in_data, 1, 2)
	in_data = F.reshape(in_data, (batchsize, -1))
	in_data = F.split_axis(in_data, seq_length, axis=1)

	x_length = Variable(xp.full((batchsize,), seq_length, dtype=np.int32))

	start_time = time.time()
	for i in range(repeat):
		loss_ctc = cuda_ctc.connectionist_temporal_classification(in_data, label_unigram, blank_symbol, x_length, Variable(length_unigram), reduce="mean")
	forward_time_mean = (time.time() - start_time) / repeat

	start_time = time.time()
	for i in range(repeat):
		loss_ctc = cuda_ctc.connectionist_temporal_classification(in_data, label_unigram, blank_symbol, x_length, Variable(length_unigram), reduce="mean")
		loss_ctc.backward()
	backward_time_mean = (time.time() - start_time) / repeat

	return forward_time_mean, backward_time_mean

def generate_cmap(colors):
	values = range(len(colors))
	vmax = np.ceil(np.max(values))
	color_list = []
	for v, c in zip(values, colors):
		color_list.append( ( v/ vmax, c) )
	return LinearSegmentedColormap.from_list('custom_cmap', color_list)

def plot(df, title):
	sns.set(font_scale=1.5)
	sns.set_style("whitegrid", {"grid.linestyle": "--"})
	df.index = ["forward","backward"]
	df = df.T
	plt.clf()
	ax = df.plot.barh(stacked=True, cmap=generate_cmap(["#597DBE", "#A0C7F1"]), width=0.2, figsize=(8, 4))
	ax.set_title(title)
	ax.set(xlabel="[ms]")
	plt.tight_layout()
	plt.savefig("{}.png".format(title))
	
def main():
	batchsize_list = [16, 32]
	label_length_list = [10, 30, 50]
	seq_length_list = [50, 100, 200]
	vocab_size_list = [100, 500, 1000]

	# dummy
	result_cupy = benchmark_cupy_ctc(16, 10, 50, 100)

	for batchsize in batchsize_list:
		for label_length, seq_length in zip(label_length_list, seq_length_list):
			for vocab_size in vocab_size_list:
				result_cupy = benchmark_cupy_ctc(batchsize, label_length, seq_length, vocab_size)
				result_cuda = benchmark_cuda_ctc(batchsize, label_length, seq_length, vocab_size)

				forward_cupy, backward_cupy = result_cupy
				forward_cuda, backward_cuda = result_cuda

				df = pd.DataFrame({
					"CUDA CTC": [forward_cuda * 1000, backward_cuda * 1000],
					"CuPy CTC": [forward_cupy * 1000, backward_cupy * 1000],
					})

				title = "l={}, vocab={}, batchsize={}".format(seq_length, vocab_size, batchsize)
				plot(df, title)

if __name__ == "__main__":
	main()