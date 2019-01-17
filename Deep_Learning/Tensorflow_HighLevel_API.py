#%%
# Import Tensorflow
import tensorflow as tf

# Start Eager Execution for prototyping
tf.enable_eager_execution()

#%%
# Path to the Csv file
file = "/home/esssfff/Documents/Git/Examples/Datasets/covtype.data"

# Column type
defaults = [tf.int32] * 55

dataset = tf.data.experimental.CsvDataset(filenames=[file], 
                                     record_defaults=defaults,
                                     header=False)

# Print column types
print(list(dataset.take(3)))

#%%
col_names = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
            'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
            'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
            'Horizontal_Distance_To_Fire_Points', 'Soil_Type', 'Cover_Type']

def _parse_csv_row(*vals):
    soil_type_t = tf.convert_to_tensor(vals[14:54])
    feat_values = vals[:10] + (soil_type_t, vals[54])
    features = dict(zip(col_names, feat_values))

    class_label = tf.math.argmax(vals[10:14], axis=0)

    return features, class_label

dataset = dataset.map(_parse_csv_row).batch(64)

print(list(dataset.take(1)))

#%%
# Cover_Type / integer / 1 to 7
cover_type = tf.feature_column.categorical_column_with_identity(
    'Cover_Type', num_buckets=8)

cover_embedding = tf.feature_column.embedding_column(
    cover_type, dimension=10)


numeric_features = [
    tf.feature_column.numeric_column(feat) for feat in numeric_cols]

# Soil_type (40 binary columns)
soil_type = tf.feature_column.numeric_column(soil_type, shape=(40,))


columns = numeric_features + [soil_type, cover_embedding]

feature_layer = tf.keras.layers.DenseFeatures(columns)

model = tf.keras.Sequential([
    feature_layer,
    tf.keras.layers.Dense(256),
    tf.keras.layers.Dense(16),
    tf.keras.layers.Dense(8),
    tf.keras.layers.Dense(4, activation=tf.nn.softmax)
])

