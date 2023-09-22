from .val_base_options import ValBaseOptions


class ValOptions(ValBaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = ValBaseOptions.initialize(self, parser)
        parser.add_argument('--dataroot', required=True )
        parser.add_argument('--name', type=str, default='experiment_name',help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--val_dataroot',default='./COVID_data_test/')
        parser.add_argument('--model', type=str, default='asynKL')
        parser.add_argument('--val_dataset_mode', type=str, default='covid_test', help='chooses how datasets are loaded.')

        return parser
