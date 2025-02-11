from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

from orca_gym.robomimic.dataset_util import DatasetReader, DatasetWriter


class OrcaGymDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation.',
                        ),
                        # 'wrist_image': tfds.features.Image(
                        #     shape=(64, 64, 3),
                        #     dtype=np.uint8,
                        #     encoding_format='png',
                        #     doc='Wrist camera RGB observation.',
                        # ),
                        'state': tfds.features.Tensor(
                            shape=(31,),
                            dtype=np.float32,
                            doc='Robot & object state, consists of '
                                'qpos[7x robot joint angles, 2x gripper position, 7x object position] +'
                                'qvel[7x robot joint angle speed, 2x gripper speed, 6x object speed].',
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(8,),
                        dtype=np.float32,
                        doc='Robot action, consists of [6x 6dof end-effector positon, 2x gripper postion].'
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, '
                            'For sparse reward, 1 for success, 0 otherwise.'
                            'For dense reward, see the reward function in the environment.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, False if it is a tuncated episode.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Kona language embedding. '
                            'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(path='data/*.hdf5', filter_key='train'),
            'val': self._generate_examples(path='data/*.hdf5', filter_key='valid'),
        }

    def _generate_examples(self, path : str, filter_key : str) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(episode_path):
            # load raw data --> this should change for your dataset
            # data = np.load(episode_path, allow_pickle=True)     # this is a list of dicts in our case
            reader = DatasetReader(episode_path)
            demo_names = reader.get_demo_names(filter_key)

            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            episode = []
            for demo_name in demo_names:
                demo_data = reader.get_demo_data(demo_name)
                                
                # compute Kona language embedding
                language_instruction = demo_data['language_instruction']
                language_embedding = self._embed([language_instruction])[0].numpy()

                demo_len = len(demo_data['actions'])
                for i in range(demo_len):
                    action = demo_data['actions'][i]
                    action[6] = np.clip(action[6], -1.0, 1.0)  # clip gripper action to [-1, 1]
                    action[7] = np.clip(action[7], 0.0, 1.0)  # clip door open to [0, 1]
                    episode.append({
                        'observation': {
                            'image': demo_data['camera_frames']['camera_primary'][i],
                            # 'wrist_image': step['wrist_image'],
                            'state': demo_data['states'][i],
                        },
                        'action': action,
                        'discount': 1.0,
                        'reward': demo_data['rewards'][i], #float(i == (len(demo_data) - 1)),
                        'is_first': i == 0,
                        'is_last': i == (demo_len - 1),
                        'is_terminal': i == (demo_len - 1),
                        'language_instruction': language_instruction,
                        'language_embedding': language_embedding,
                    })

            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path
                }
            }

            # if you want to skip an example for whatever reason, simply return None
            return episode_path, sample

        # create list of all examples
        episode_paths = glob.glob(path)

        # for smallish datasets, use single-thread parsing
        for sample in episode_paths:
            yield _parse_example(sample)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )

