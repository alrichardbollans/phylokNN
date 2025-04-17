import os

import keras
import pandas as pd
from keras import callbacks
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def autoencode_pairwise_distances(distance_data: pd.DataFrame, reduction_fraction: float, _output_dir: str = None, plot=False):
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # If  want to use CPU as GPU doesn't have enough memory

    _early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True)

    # Define input dimensions
    input_dim = distance_data.shape[1]

    def create_autoencoder(_input_dim: int, encoded_dim: int):
        """
        Create an autoencoder model.

        :param _input_dim: The dimension of the input data.

        :param encoded_dim: The dimension of the encoded layer.
        :return: The created autoencoder model.
        """
        ##
        # Add a Dense layer with a L1 activity regularizer to constrain the representations to be compact
        autoencoder = keras.Sequential([
            keras.Input(shape=(_input_dim,)),
            keras.layers.Dense(encoded_dim * 2, activation='leaky_relu'),

            keras.layers.Dense(encoded_dim, activation='leaky_relu', name='encoder'),
            keras.layers.Dense(encoded_dim * 2, activation='leaky_relu'),
            keras.layers.Dense(_input_dim, activation='leaky_relu', name='decoder'),
            # relu for final activation as output is in the range [0,∞), but use leaky_relu as was experiencing dead neurons in the output

        ])

        return autoencoder

    # Create the autoencoder model
    encoding_dim = int(len(distance_data.columns) * reduction_fraction)
    autoencoder = create_autoencoder(input_dim, encoded_dim=encoding_dim)

    # Compile the model
    autoencoder.compile(optimizer='adam', loss='mse')

    X_train, X_val = train_test_split(distance_data, test_size=0.2)

    # Train the model
    # the task is to encode the given dataset.
    history = autoencoder.fit(X_train, X_train, callbacks=[_early_stopping], epochs=1000,
                              batch_size=32, shuffle=True,
                              validation_data=(X_val, X_val), verbose=0)

    # predicted_X_val = autoencoder.predict(X_val)

    # Define encoder model
    encoder_model = keras.Model(inputs=autoencoder.layers[0].input, outputs=autoencoder.get_layer('encoder').output)

    encoded = encoder_model.predict(distance_data)

    ### Outputs
    best_train_loss = history.history['loss'][-1]
    best_val_loss = history.history['val_loss'][-1]
    if _output_dir is not None:
        mean_example = distance_data.mean().to_frame().T
        full_mean_df = pd.concat([mean_example] * len(distance_data.index), ignore_index=True)
        mean_loss = mean_squared_error(distance_data, full_mean_df)
        baseline_df = pd.DataFrame([[mean_loss, best_train_loss, best_val_loss, len(history.history['loss']), encoding_dim]],
                                   columns=['Baseline mse', 'train_loss', 'val_loss', 'number_of_epochs', 'latent space size'])
        baseline_df.to_csv(os.path.join(_output_dir, 'autoencoder_metrics.csv'))

        if plot:
            ## Display of the model
            keras.utils.plot_model(autoencoder, to_file=os.path.join(_output_dir, 'phylogeny_autoencoder.png'), show_shapes=True,
                                   show_dtype=True,
                                   show_layer_names=True,
                                   rankdir='TB',
                                   expand_nested=True,
                                   dpi=300
                                   )

            # Plot the learning progression
            plt.plot(history.history['loss'], label='train_loss')
            plt.plot(history.history['val_loss'], label='val_loss')
            plt.title('Autoencoder Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(os.path.join(_output_dir, 'phylogeny_autoencoder_training_loss.png'))

            ## Display of the encoder
            keras.utils.plot_model(encoder_model, to_file=os.path.join(_output_dir, 'phylogeny_encoder.png'), show_shapes=True,
                                   show_dtype=True,
                                   show_layer_names=True,
                                   rankdir='TB',
                                   expand_nested=True,
                                   dpi=300
                                   )

    return encoder_model, pd.DataFrame(encoded, index=distance_data.index)


if __name__ == '__main__':
    example_data = pd.read_csv('../analysis/data/simulations/binary/1/tree_distances.csv', index_col=0)
    autoencode_pairwise_distances(example_data, 0.1, 'example_out', plot=True)
