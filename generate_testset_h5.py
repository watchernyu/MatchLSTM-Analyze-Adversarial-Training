from lib.dataset.squad_dataset import SquadDataset
from lib.dataset.squad_dataset_testpart import SquadTestDataset
from helpers.generic import print_shape_info, print_data_samples, random_generator, squad_trim, add_char_level_stuff,\
    torch_model_summarize, generator_queue, evaluate

# if squad_testset.1.0.h5 file is not yet here,
# run this file once to generate it.

testdataset = SquadTestDataset(dataset_h5='squad_testset.1.0.h5',
                       data_path='test_data' + "/",
                       ignore_case=True)

add_any_4_testdata, add_one_sent_testdata, add_best_sent_testdata = testdataset.get_data() # each is a data dict

print_shape_info(add_any_4_testdata)
print_shape_info(add_one_sent_testdata)
print_shape_info(add_best_sent_testdata)

if True:
    print('----------------------------------  printing out data shape')
    print_data_samples(testdataset, add_any_4_testdata, 0, 3)
    print_data_samples(testdataset, add_one_sent_testdata, 0, 3)
    print_data_samples(testdataset, add_best_sent_testdata, 0, 3)

#self.data['train'], self.data['valid'], self.data['test']

print "all finished"