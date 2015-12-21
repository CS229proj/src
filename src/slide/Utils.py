import pandas as pd

def vectorize_xy(X,Y, tokenizer, label_encoder):
    X_matrix = vectorize_x(X, tokenizer)
    Y_vector = vectorize_y(Y, label_encoder)
    print(tokenizer.n_features)
    print("Data vectorized.")
    return [X_matrix, Y_vector]

def vectorize_x(X, tokenizer):
    X_matrix = tokenizer.fit_transform(X)
    return X_matrix

def vectorize_y(Y, label_encoder):
    Y_vector = label_encoder.fit_transform(Y)
    return Y_vector

def devectorize_y(Y_vector, label_encoder):
    Y = label_encoder.nverse_transform(Y_vector)
    return Y

def get_y(fl):
    data = pd.read_csv(fl, encoding='utf-8', sep=r'\t+', header=None, names=['text', 'label'])
    trY = data['label'].values
    return trY

def most_common(lst):
    return max(set(lst), key=lst.count)

def save_output(X, Y, output_filename):
    raw_data = {'text': X, 'classes': Y}
    data = pd.DataFrame(data=raw_data)
    data.to_csv(output_filename, sep='\t', index=False, header=None)