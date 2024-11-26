import os
import numpy as np
import functools
import tensorflow as tf
import json
import argparse
import pickle

# Global debug flag
DEBUG = False

_FEATURE_DESCRIPTION = {
    'position': tf.io.VarLenFeature(tf.string),
}

_FEATURE_DESCRIPTION_WITH_GLOBAL_CONTEXT = _FEATURE_DESCRIPTION.copy()
_FEATURE_DESCRIPTION_WITH_GLOBAL_CONTEXT['step_context'] = tf.io.VarLenFeature(
    tf.string)

_FEATURE_DTYPES = {
    'position': {
        'in': np.float32,
        'out': tf.float32
    },
    'step_context': {
        'in': np.float32,
        'out': tf.float32
    }
}

_CONTEXT_FEATURES = {
    'key': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'particle_type': tf.io.VarLenFeature(tf.string)
}

def _read_metadata(data_path):
    with open(os.path.join(data_path, 'metadata.json'), 'rt') as fp:
        return json.loads(fp.read())

def convert_to_tensor(value, encoded_dtype):
    if DEBUG:
        print(f"Converting value to tensor with dtype {encoded_dtype}")
    tensor = tf.convert_to_tensor(value, dtype=encoded_dtype)
    if DEBUG:
        print(f"Converted tensor: {tensor}")
    return tensor

def parse_serialized_simulation_example(example_proto, metadata):
    if 'context_mean' in metadata:
        feature_description = _FEATURE_DESCRIPTION_WITH_GLOBAL_CONTEXT
    else:
        feature_description = _FEATURE_DESCRIPTION
    context, parsed_features = tf.io.parse_single_sequence_example(
        example_proto,
        context_features=_CONTEXT_FEATURES,
        sequence_features=feature_description)
    for feature_key, item in parsed_features.items():
        if DEBUG:
            print(f"Parsing feature: {feature_key}")
        convert_fn = functools.partial(
            convert_to_tensor, encoded_dtype=_FEATURE_DTYPES[feature_key]['in'])
        parsed_features[feature_key] = tf.py_function(
            convert_fn, inp=[item.values], Tout=_FEATURE_DTYPES[feature_key]['out'])
        if DEBUG:
            print(f"Parsed feature {feature_key}: {parsed_features[feature_key]}")

    position_shape = [metadata['sequence_length'] + 1, -1, metadata['dim']]
    parsed_features['position'] = tf.reshape(parsed_features['position'],
                                             position_shape)
    if DEBUG:
        print(f"Reshaped position: {parsed_features['position']}")

    sequence_length = metadata['sequence_length'] + 1
    if 'context_mean' in metadata:
        context_feat_len = len(metadata['context_mean'])
        parsed_features['step_context'] = tf.reshape(
            parsed_features['step_context'],
            [sequence_length, context_feat_len])
        if DEBUG:
            print(f"Reshaped step context: {parsed_features['step_context']}")

    context['particle_type'] = tf.py_function(
        functools.partial(convert_to_tensor, encoded_dtype=np.int64),
        inp=[context['particle_type'].values],
        Tout=[tf.int64])
    context['particle_type'] = tf.reshape(context['particle_type'], [-1])
    if DEBUG:
        print(f"Reshaped particle type: {context['particle_type']}")

    return context, parsed_features

def parse_arguments():
    parser = argparse.ArgumentParser("Transform options.")
    parser.add_argument('--dataset', default='Water', type=str, help='dataset to transform.')
    parser.add_argument('--split', default='train', choices=['train', 'test', 'valid'], help='split to transform.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode for detailed output.')

    args = parser.parse_args()

    global DEBUG
    DEBUG = args.debug

    return args

def transform(args):
    metadata = _read_metadata(f"datasets/{args.dataset}")
    ds = tf.data.TFRecordDataset([f"datasets/{args.dataset}/{args.split}.tfrecord"])
    ds = ds.map(functools.partial(
        parse_serialized_simulation_example, metadata=metadata
    ))

    positions = []
    step_context = []
    particle_types = []
    for context, parsed_features in ds:
        particle_types.append(context['particle_type'].numpy())
        positions.append(parsed_features['position'].numpy())
        if 'step_context' in parsed_features:
            step_context.append(parsed_features['step_context'].numpy())

    if DEBUG:
        print(f"Saving positions to datasets/{args.dataset}/{args.split}/positions.pkl")
    position_file = open(f'datasets/{args.dataset}/{args.split}/positions.pkl', 'wb')
    pickle.dump(positions, position_file)
    position_file.close()

    if DEBUG:
        print(f"Saving particle types to datasets/{args.dataset}/{args.split}/particle_types.pkl")
    particle_file = open(f'datasets/{args.dataset}/{args.split}/particle_types.pkl', 'wb')
    pickle.dump(particle_types, particle_file)
    particle_file.close()

    if len(step_context) != 0:
        if DEBUG:
            print(f"Saving step context to datasets/{args.dataset}/{args.split}/step_context.pkl")
        context_file = open(f'datasets/{args.dataset}/{args.split}/step_context.pkl', 'wb')
        pickle.dump(step_context, context_file)
        context_file.close()

if __name__ == '__main__':
    transform(parse_arguments())