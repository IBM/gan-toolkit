def generate_samples(model, batch_size, generated_num, output_file, conf_data):
    samples = []
    for _ in range(int(generated_num / batch_size)):
        sample = model.sample(batch_size, conf_data['generator']['sequece_length']).cpu().data.numpy().tolist()
        samples.extend(sample)
    with open(output_file, 'w') as fout:
        for sample in samples:
            string = ' '.join([str(s) for s in sample])
            fout.write('%s\n' % string)