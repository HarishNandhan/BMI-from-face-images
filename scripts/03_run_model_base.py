import logging
#logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO)

import click

@click.command()
@click.option('--mode', default='predict', 
	help='one of {train, predict}', 
	type = click.Choice(['train','predict']))
@click.option('--model_type', default='vgg16', 
	help='model type, one of {vgg16, resnet50}',
	type = click.Choice(['vgg16','resnet50']))
@click.option('--model_tag',
	default='base', 
	help='tag of model')
@click.option('--bs', default=8, help='batch size')
@click.option('--epochs', default=2, help='epochs')
@click.option('--freeze_backbone', default=True, 
	help='True => transfer learning; False => train from scratch')
@click.option('--version', is_flag=True, help='show version')

def main(model_type, model_tag, mode, bs, epochs, freeze_backbone, version):

	if version:
		click.echo('v1.0.0')
		exit(0)

	import glob
	import pandas as pd
	import os
	import json
	from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
	from matplotlib import pyplot as plt
	from pathlib import Path
	from models import FacePrediction

	assert mode in ['train', 'predict'], 'mode is not right'

	model_id = '{:s}_{:s}'.format(model_type, model_tag)
	model_dir = './saved_model/model_{:s}.h5'.format(model_id)

	img_input_dir = './data/face_aligned/'
	train_data_path = './data/train.csv'
	valid_data_path = './data/valid.csv'
	test_data_path = './data/test/test_aligned'

	allimages = os.listdir(img_input_dir)
	train = pd.read_csv(train_data_path)
	valid = pd.read_csv(valid_data_path)

	train = train.loc[train['index'].isin(allimages)]
	valid = valid.loc[valid['index'].isin(allimages)]

	# create metrics, model dirs
	Path('./metrics').mkdir(parents = True, exist_ok = True)
	Path('./saved_model').mkdir(parents = True, exist_ok = True)

	data = pd.concat([train, valid])

	es = EarlyStopping(patience=3)
	ckp = ModelCheckpoint(model_dir, save_best_only=True, save_weights_only=True, verbose=1)
	tb = TensorBoard('./tb/%s'%(model_type))
	callbacks = [es, ckp, tb]

	model = FacePrediction(img_dir = img_input_dir, model_type = model_type)
	model.define_model(freeze_backbone = freeze_backbone)

	if mode == 'train':
		model.train(train, valid, bs = bs, epochs = epochs, callbacks = callbacks)
	else:
		model.load_weights(model_dir)

	metrics = model.evaulate(valid)
	metrics['model'] = model_id
	with open('./metrics/{:s}.json'.format(model_id), 'w') as f:
		json.dump(metrics, f)

	metrics = []
	for i in glob.glob('./metrics/*.json'):
		with open(i, 'r') as f:
			res = json.load(f)
		metrics.append(res)
	metrics = pd.DataFrame(metrics)
	metrics['model'] = metrics['model'].apply(lambda i: '* ' + i if i == model_id else i)
	print(metrics.set_index('model').round(3))

	print(model.predict_df(test_data_path))

if __name__ == '__main__':
	main()