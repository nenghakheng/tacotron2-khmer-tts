# import tensorflow as tf
import logging
from text import symbols

# Set up logging
logging.basicConfig(level=logging.INFO)

print("Hyper Params script")

# Custom HParams class to allow dot notation access
class HParams(dict):
    """Custom class that allows dot notation for dictionary keys."""
    
    def __getattr__(self, name):
        """Override attribute access to allow dot notation."""
        if name in self:
            return self[name]
        else:
            raise AttributeError(f"'HParams' object has no attribute '{name}'")
    
    def __setattr__(self, name, value):
        """Override setting attributes."""
        self[name] = value

    def __delattr__(self, name):
        """Override deleting attributes."""
        del self[name]

# Create Hyperparameters using the custom HParams class
def create_hparams(hparams_string=None, verbose=False):
    """Create model hyperparameters. Parse nondefault from given string."""

    # Initialize hyperparameters using the custom class
    hparams = HParams({
        ################################
        # Experiment Parameters        #
        ################################
        'epochs': 1600,
        'iters_per_checkpoint': 1000,
        'seed': 1234,
        'dynamic_loss_scaling': True,
        'fp16_run': False,
        'distributed_run': False,
        'dist_backend': "nccl",
        'dist_url': "tcp://localhost:54321",
        'cudnn_enabled': True,
        'cudnn_benchmark': False,
        'ignore_layers': ['embedding.weight'],

        ################################
        # Data Parameters             #
        ################################
        'load_mel_from_disk': False,
        'training_files': './datasets/train_datasets/line_index.tsv',
        'validation_files': './datasets/validation_datasets/line_index.tsv',
        'text_cleaners': ['transliteration_cleaners'],

        ################################
        # Audio Parameters             #
        ################################
        'max_wav_value': 32768.0,
        'sampling_rate': 22050,
        'filter_length': 1024,
        'hop_length': 256,
        'win_length': 1024,
        'n_mel_channels': 80,
        'mel_fmin': 0.0,
        'mel_fmax': 8000.0,

        ################################
        # Model Parameters             #
        ################################
        'n_symbols': len(symbols),
        'symbols_embedding_dim': 512,

        # Encoder parameters
        'encoder_kernel_size': 5,
        'encoder_n_convolutions': 3,
        'encoder_embedding_dim': 512,

        # Decoder parameters
        'n_frames_per_step': 1,  # currently only 1 is supported
        'decoder_rnn_dim': 1024,
        'prenet_dim': 256,
        'max_decoder_steps': 1000,
        'gate_threshold': 0.5,
        'p_attention_dropout': 0.1,
        'p_decoder_dropout': 0.1,

        # Attention parameters
        'attention_rnn_dim': 1024,
        'attention_dim': 128,

        # Location Layer parameters
        'attention_location_n_filters': 32,
        'attention_location_kernel_size': 31,

        # Mel-post processing network parameters
        'postnet_embedding_dim': 512,
        'postnet_kernel_size': 5,
        'postnet_n_convolutions': 5,

        ################################
        # Optimization Hyperparameters #
        ################################
        'use_saved_learning_rate': False,
        'learning_rate': 1e-3,
        'weight_decay': 1e-6,
        'grad_clip_thresh': 1.0,
        'batch_size': 32,
        'mask_padding': True  # set model's padded outputs to padded values
    })

    # If a hparams string is provided, parse it
    if hparams_string:
        logging.info('Parsing command line hparams: %s', hparams_string)
        # Assuming hparams_string is in a format where key=value pairs are provided (like 'epochs=1000')
        hparams_list = hparams_string.split(',')
        for param in hparams_list:
            key, value = param.split('=')
            if key in hparams:
                hparams[key] = type(hparams[key])(value)  # Convert to the correct type
            else:
                logging.warning("Unknown parameter: %s", key)

    # If verbose, log the final parsed hyperparameters
    if verbose:
        logging.info('Final parsed hparams: %s', hparams)

    return hparams