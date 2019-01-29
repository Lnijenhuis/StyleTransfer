import numpy as np
import math
import tensorflow as tf


def encoder(data):
	#[512x228]
	L0 = tf.layers.conv2d(data, filters = 8, kernel_size = [5,5], strides = 1, padding = "same", name = "enc_L0", activation = tf.nn.elu)
	L1 = tf.layers.conv2d(L0, filters = 12, kernel_size = [5,5], strides = 2, padding = "same", name = "enc_L1", activation= tf.nn.elu)
	#[256x114]
	L2 = tf.layers.conv2d(L1, filters = 16, kernel_size = [5,5], strides = 2, padding = "same", name = "enc_L2", activation= tf.nn.elu)
	#[128x72]
	L3 = tf.layers.conv2d(L2, filters = 20, kernel_size = [5,5], strides = 2, padding = "same", name = "enc_L3", activation= tf.nn.elu)
	#L3_1 = tf.layers.conv2d(L3, filters = 10, kernel_size = [1,1], strides = 1, padding = "same")
	#[64X36]
	L3_2 = tf.layers.conv2d(L3, filters = 30, kernel_size = [5,5], strides = 2, padding = "same", name = "enc_L3_2", activation= tf.nn.elu)
	#[32x18]
	L3_3 = tf.layers.conv2d(L3_2, filters = 32, kernel_size = [5,5], strides = 2, padding = "same", name = "enc_L3_3", activation = tf.nn.elu)
	#[16x9]
	#dimreduce_L3 = tf.layers.conv2d(L3_2, filters = 3, kernel_size= [1,1], strides = 1, padding= "same", name = "dimreduce_L3", activation = tf.nn.elu)
	#[32x18]
	#flatten_L3 = tf.layers.flatten(dimreduce_L3, name = "flatten_L3")
	#[576]
#	L4 = tf.layers.dense(flatten_L3, 512 , activation = tf.nn.elu ,name = "L4_dense")
	#[512]
	return L3_3
	
	
def decoderA(data,training = False):
	with tf.variable_scope("decoderA", reuse=tf.AUTO_REUSE):
		#[512]
		#L4_norm = tf.layers.batch_normalization(data,training=training)
		#L4 = tf.layers.dense(L4_norm, 576 * 3, activation = tf.nn.elu)
		#[2304] [-1,64,36,3]
		#res_L4 = tf.reshape(L4, [-1,18,32,3]);#
		#[16x9]
		L4_1 = tf.layers.conv2d_transpose(data, filters = 20, kernel_size = [5,5], strides = 2, padding = "same", activation = tf.nn.elu)
		#[32x16]
		L4_2 = tf.layers.conv2d_transpose(L4_1, filters = 16, kernel_size = [5,5], strides = 2, padding = "same", activation = tf.nn.elu)
		#[64x36]
		#L5 = tf.layers.conv2d_transpose(L4_ups, filters = 14, kernel_size = [5,5], strides = 2, padding = "same", activation= tf.nn.elu)
		L5 = tf.layers.conv2d_transpose(L4_2 , filters = 10, kernel_size = [5,5], strides = 2, padding = "same", activation= tf.nn.elu)
		#[128x72]
		L6 = tf.layers.conv2d_transpose(L5, filters = 8, kernel_size = [5,5], strides = 2, padding = "same", activation= tf.nn.elu)
		#[256x114]
		L7 = tf.layers.conv2d_transpose(L6, filters = 6, kernel_size = [5,5], strides = 2, padding = "same", activation= tf.nn.elu)
		#[512x228]
		L8 = tf.layers.conv2d(L7, filters = 4, kernel_size = [5,5], strides = 1, padding = "same", activation = tf.nn.elu)
		L9 = tf.layers.conv2d(L8, filters = 3, kernel_size = [1,1], strides = 1, padding = "same", activation = tf.nn.sigmoid)
	return L9
	
def decoderB(data,training = False):
	with tf.variable_scope("decoderB", reuse=tf.AUTO_REUSE):
		#[512]
		#L4_norm = tf.layers.batch_normalization(data,training=training)
		#L4 = tf.layers.dense(L4_norm, 576 * 3, activation = tf.nn.elu)
		#[2304] [-1,64,36,3]
		#res_L4 = tf.reshape(L4, [-1,18,32,3]);#
		#[16x9]
		L4_1 = tf.layers.conv2d_transpose(data, filters = 20, kernel_size = [5,5], strides = 2, padding = "same", activation = tf.nn.elu)
		#[32x16]
		L4_2 = tf.layers.conv2d_transpose(L4_1, filters = 16, kernel_size = [5,5], strides = 2, padding = "same", activation = tf.nn.elu)
		#[64x36]
		#L5 = tf.layers.conv2d_transpose(L4_ups, filters = 14, kernel_size = [5,5], strides = 2, padding = "same", activation= tf.nn.elu)
		L5 = tf.layers.conv2d_transpose(L4_2, filters = 10, kernel_size = [5,5], strides = 2, padding = "same", activation= tf.nn.elu)
		#[128x72]
		L6 = tf.layers.conv2d_transpose(L5, filters = 8, kernel_size = [5,5], strides = 2, padding = "same", activation= tf.nn.elu)
		#[256x114]
		L7 = tf.layers.conv2d_transpose(L6, filters = 6, kernel_size = [5,5], strides = 2, padding = "same", activation= tf.nn.elu)
		#[512x228]
		L8 = tf.layers.conv2d(L7, filters = 4, kernel_size = [5,5], strides = 1, padding = "same", activation = tf.nn.elu)
		L9 = tf.layers.conv2d(L8, filters = 3, kernel_size = [1,1], strides = 1, padding = "same", activation = tf.nn.sigmoid)
	return L9
	
def _parse_function(filename):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_image(image_string,channels=3)
  image_decoded.set_shape([288,512,3])
  #tf.image.resize_images(image_decoded, [288, 512])
  image_resized = tf.cast(image_decoded/255, tf.float32)
  return image_resized



dirA = "C:\A_pic\*.jpg"
dirB = "C:\B_pic\*.jpg"
filenamesA = tf.train.match_filenames_once(dirA)
filenamesB = tf.train.match_filenames_once(dirB)
DS_A = tf.data.Dataset.from_tensor_slices(filenamesA)
DS_B = tf.data.Dataset.from_tensor_slices(filenamesB)
DS_A_eval = tf.data.Dataset.from_tensor_slices(filenamesA).map(_parse_function, num_parallel_calls = 8).repeat(1000).batch(1)
DS_B_eval = tf.data.Dataset.from_tensor_slices(filenamesB).map(_parse_function, num_parallel_calls = 8).repeat(1000).batch(1)
DS_A = DS_A.map(_parse_function,num_parallel_calls = 8).shuffle(1000).repeat(3000).batch(12).prefetch(10)
DS_B = DS_B.map(_parse_function,num_parallel_calls = 8).shuffle(1000).repeat(3000).batch(12).prefetch(10)

encoder_impl = tf.make_template('encoder', encoder)
#decoderA = tf.make_template('decoderA', decoderA(imA))
iteratorA = DS_A.make_initializable_iterator()
iteratorB = DS_B.make_initializable_iterator()
iteratorA_eval = DS_A_eval.make_initializable_iterator()
iteratorB_eval = DS_B_eval.make_initializable_iterator()
imA = iteratorA.get_next()
imB = iteratorB.get_next()


noise_vec = tf.random_normal(shape=tf.shape(imA), mean = 0.0, stddev = (25.5/255.0), dtype=tf.float32) 
imA_noise = imA + noise_vec;
imB_noise = imB + noise_vec;

imA_eval = iteratorA_eval.get_next()
imB_eval = iteratorB_eval.get_next()
#decoderA = tf.make_template('decoderA', decoderA(imA,"memeA"))
#decoderB = tf.make_template('decoderB', decoderA(imA,"memeB"))
TF_training = tf.placeholder(tf.bool)
AutoA = decoderA(encoder_impl(imA_noise),TF_training)
AutoB = decoderB(encoder_impl(imB_noise),TF_training)

AutoA_eval = decoderA(encoder_impl(imB_eval), False)
AutoB_eval = decoderB(encoder_impl(imA_eval), False)
#AutoA_eval = tf.split(AutoA_eval,[288,512,3],axis = 1, num = 1)
#AutoB_eval = tf.split(AutoB_eval,[288,512,3],axis = 1, num = 1)
#AutoA_eval = tf.split(AutoA_eval,1)
#AutoB_eval = tf.split(AutoB_eval,1)
#AutoA_eval.set_shape([288,512,3])
#AutoB_eval.set_shape([288,512,3])
encode_imA = tf.image.encode_jpeg(tf.cast(255*AutoA_eval[0],tf.uint8))
encode_imB = tf.image.encode_jpeg(tf.cast(255*AutoB_eval[0],tf.uint8))

encode_imAA = tf.image.encode_jpeg(tf.cast(255*AutoA[0],tf.uint8))
encode_imBB = tf.image.encode_jpeg(tf.cast(255*AutoB[0],tf.uint8))

name_A = tf.placeholder(tf.string)
name_B = tf.placeholder(tf.string)
name_AA = tf.placeholder(tf.string)
name_BB = tf.placeholder(tf.string)

fwriteA = tf.write_file(name_A, encode_imA)
fwriteB = tf.write_file(name_B, encode_imB)
fwriteAA = tf.write_file(name_AA, encode_imAA)
fwriteBB = tf.write_file(name_BB, encode_imBB)

diffA = tf.reduce_mean(tf.square(imA - AutoA))
diffB = tf.reduce_mean(tf.square(imB - AutoB))

kf = tf.placeholder(tf.float32)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
	optimizeA = tf.train.AdamOptimizer(kf).minimize(diffA)
	optimizeB = tf.train.AdamOptimizer(kf).minimize(diffB)

#Option for GTX 970 on home PC. Design flaw in that card makes last bit of VRAM Slow
gpu_opt = tf.GPUOptions(per_process_gpu_memory_fraction = 0.7)

with tf.Session() as sess:#$config=tf.ConfigProto(gpu_options=gpu_opt)) as sess:
	sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])

	sess.run([iteratorA.initializer, iteratorB.initializer, iteratorA_eval.initializer, iteratorB_eval.initializer])
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)
	j = 0
	while j < (1000):
	#Train
		j = j+1
		i = 0
		k = 0
		lA = []
		lB = []
		while i < (2800/12):
			t = (2800*(j-1)+12*i)%(2800*12*5);
			dr = 1.0/(2800 * 10.0);
			lr = 0.01 * (math.exp(-dr * t)) * math.exp( - 1 * (2800*(j-1)+12*i) // (2800*12*5) )
			filenameAA = "D:/FAST/A_sample/" + str(j) + "/number_" + str(i) + ".jpg"
			filenameBB = "D:/FAST/B_sample/" + str(j) + "/number_" + str(i) + ".jpg"
			_, _, a_loss, b_loss, _, _ = sess.run([ optimizeA, optimizeB, diffA, diffB, fwriteAA, fwriteBB], feed_dict = {TF_training: True, name_AA: filenameAA, name_BB: filenameBB, kf: lr})
			lA.append(a_loss)
			lB.append(b_loss)
			i = i + 1
		meanAloss = (sum(lA)/len(lA))
		meanBloss = (sum(lB)/len(lB))
		print("Epoch: " + str(j) + " | A_loss: " + '%.3E' %meanAloss + " | B_loss: " + '%.3E' %meanBloss + " | LR = " + '%.3E' %lr)
	#Dump transfer images
		while k < (10):
			filenameA = "D:/FAST/CrossAB/" + str(j) + "/number_" + str(k) + ".jpg"
			filenameB = "D:/FAST/CrossBA/" + str(j) + "/number_" + str(k) + ".jpg"
			a, b = sess.run([fwriteA, fwriteB], feed_dict = {name_A: filenameA , name_B: filenameB})
			#print(k)
			k = k + 1

	coord.request_stop()
	coord.join(threads)