from utils import summarize_performance, generate_fake_samples, generate_real_samples
from discriminator import Discriminator
from generator import Generator
from gan import GAN
from dataloader import DataLoader
from datetime import datetime 

def main():
	# init models
	disc = Discriminator((256, 256, 3))
	d_model = disc.model
	gen = Generator()
	g_model = gen.model
	gan = GAN(gen.get_model(), disc.get_model(), (256, 256, 3))
	gan_model = gan.get_model()
	
	# get data
	data = DataLoader()
	dataset = data.preprocess_data()
	#data.plot_sample_images()
	
	# train
	start1 = datetime.now() 
	train(d_model, g_model, gan_model, dataset, n_epochs=10, n_batch=1)
	stop1 = datetime.now()
	
	#Execution time of the model 
	execution_time = stop1-start1 
	print("Execution time is: ", execution_time)


# train pix2pix models
def train(d_model, g_model, gan_model, dataset, n_epochs=100, n_batch=1):
	# determine the output square shape of the discriminator
	n_patch = d_model.output_shape[1]
	# unpack dataset
	trainA, trainB = dataset
	# calculate the number of batches per training epoch
	bat_per_epo = int(len(trainA) / n_batch)
	# calculate the number of training iterations
	n_steps = bat_per_epo * n_epochs
	# manually enumerate epochs
	for i in range(n_steps):
		# select a batch of real samples
		[X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)
		# generate a batch of fake samples
		X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
		# update discriminator for real samples
		d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
		# update discriminator for generated samples
		d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
		# update the generator
		g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
		# summarize performance
		print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))
		# summarize model performance
		if (i+1) % (bat_per_epo * 10) == 0:
			summarize_performance(i, g_model, dataset)
			

if __name__ == "__main__":
	main()